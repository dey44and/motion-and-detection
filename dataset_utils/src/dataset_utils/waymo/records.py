"""Seekable TFRecord framing with CRC32C checks and bounded identity reads."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from google.protobuf.message import DecodeError

from .models import NativeWaymoError
from .proto import ContextProto

MAX_RECORD_BYTES = 128 * 1024 * 1024
MAX_CONTEXT_BYTES = 1024 * 1024


def _crc_table():
    values = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        values.append(value)
    return tuple(values)


_CRC_TABLE = _crc_table()


def crc32c(data: bytes) -> int:
    try:
        import google_crc32c
    except ImportError:
        value = 0xFFFFFFFF
        for byte in data:
            value = _CRC_TABLE[(value ^ byte) & 255] ^ (value >> 8)
        return value ^ 0xFFFFFFFF
    return google_crc32c.value(data)


def masked_crc32c(data: bytes) -> int:
    value = crc32c(data)
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


@dataclass(frozen=True)
class RecordOffset:
    offset: int
    length: int


def index_records(path: str | Path) -> tuple[RecordOffset, ...]:
    path = Path(path)
    result = []
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            offset = 0
            while offset < file_size:
                stream.seek(offset)
                header = stream.read(12)
                if len(header) != 12:
                    raise NativeWaymoError(f"Truncated TFRecord header at byte {offset} in {path}.")
                length = struct.unpack_from("<Q", header)[0]
                if struct.unpack_from("<I", header, 8)[0] != masked_crc32c(header[:8]):
                    raise NativeWaymoError(f"TFRecord length CRC mismatch at byte {offset} in {path}.")
                if not 0 < length <= MAX_RECORD_BYTES:
                    raise NativeWaymoError(f"TFRecord length {length} exceeds the allowed record size.")
                if offset + 16 + length > file_size:
                    raise NativeWaymoError(f"Truncated TFRecord payload at byte {offset} in {path}.")
                result.append(RecordOffset(offset, length))
                offset += 16 + length
    except OSError as exc:
        raise NativeWaymoError(f"Cannot index native TFRecord {path}: {exc}") from exc
    if not result:
        raise NativeWaymoError(f"Native TFRecord contains no frames: {path}")
    return tuple(result)


def read_record(path: str | Path, record: RecordOffset) -> bytes:
    try:
        with Path(path).open("rb") as stream:
            stream.seek(record.offset)
            header = stream.read(12)
            if len(header) != 12 or struct.unpack_from("<Q", header)[0] != record.length:
                raise NativeWaymoError("TFRecord changed after indexing; reload the native reader.")
            if struct.unpack_from("<I", header, 8)[0] != masked_crc32c(header[:8]):
                raise NativeWaymoError("TFRecord length CRC mismatch.")
            payload, checksum = stream.read(record.length), stream.read(4)
            if len(payload) != record.length or len(checksum) != 4:
                raise NativeWaymoError("Truncated TFRecord payload.")
            if struct.unpack("<I", checksum)[0] != masked_crc32c(payload):
                raise NativeWaymoError("TFRecord payload CRC mismatch.")
            return payload
    except OSError as exc:
        raise NativeWaymoError(f"Cannot read native TFRecord {path}: {exc}") from exc


def _stream_varint(stream, end):
    value = 0
    for shift in range(0, 70, 7):
        if stream.tell() >= end:
            raise NativeWaymoError("Truncated native protobuf identity.")
        data = stream.read(1)
        if not data:
            raise NativeWaymoError("Truncated native protobuf identity.")
        byte = data[0]
        value |= (byte & 127) << shift
        if not byte & 128:
            if shift == 63 and byte > 1:
                break
            return value
    raise NativeWaymoError("Invalid native protobuf identity varint.")


def read_identity(stream, record: RecordOffset) -> tuple[str, int]:
    """Read context and timestamp while seeking past unrelated protobuf fields.

    Payload CRC is checked by read_record when a frame is selected. Header
    indexing checks length CRCs; this metadata pass reads only small identity
    fields and never decompresses images or range matrices.
    """
    position, end = record.offset + 12, record.offset + 12 + record.length
    stream.seek(position)
    context_name, timestamp = None, None
    for _ in range(10000):
        if stream.tell() >= end:
            break
        tag = _stream_varint(stream, end)
        number, wire = tag >> 3, tag & 7
        if not number:
            raise NativeWaymoError("Invalid native protobuf field zero.")
        if wire == 0:
            value = _stream_varint(stream, end)
            if number == 2:
                timestamp = value
        elif wire == 2:
            size = _stream_varint(stream, end)
            if size > end - stream.tell():
                raise NativeWaymoError("Truncated native protobuf identity field.")
            if number == 1:
                if size > MAX_CONTEXT_BYTES:
                    raise NativeWaymoError("Native context metadata exceeds the size limit.")
                context = ContextProto()
                try:
                    context.ParseFromString(stream.read(size))
                except DecodeError as exc:
                    raise NativeWaymoError(f"Invalid native context protobuf: {exc}") from exc
                context_name = context.name
            else:
                stream.seek(size, 1)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            if size > end - stream.tell():
                raise NativeWaymoError("Truncated native protobuf scalar.")
            stream.seek(size, 1)
        else:
            raise NativeWaymoError(f"Unsupported native protobuf wire type {wire}.")
        if context_name and timestamp is not None:
            if timestamp >= 1 << 63:
                raise NativeWaymoError("Native frame timestamp must be a nonnegative int64.")
            return context_name, timestamp
    raise NativeWaymoError("Native Frame is missing context.name or timestamp_micros.")
