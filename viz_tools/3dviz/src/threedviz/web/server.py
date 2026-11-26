"""Loopback-only HTTP transport for the 3dviz browser frontend."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import ipaddress
from pathlib import Path
from threading import Thread
from typing import Annotated, Literal
from urllib.parse import urlsplit
import webbrowser

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from .service import BrowserError, LaunchOptions, SessionManager


class JobRequest(BaseModel):
    """Reject coercion and accidental options instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["connect", "select", "prepare", "detection", "segmentation"]
    dataset: Annotated[StrictStr, Field(max_length=256)] | None = None
    root: Annotated[StrictStr, Field(max_length=8192)] | None = None
    version: Annotated[StrictStr, Field(max_length=256)] | None = None
    config_path: Annotated[StrictStr, Field(max_length=8192)] | None = None
    split: Annotated[StrictStr, Field(max_length=256)] | None = None
    scene_id: Annotated[StrictStr, Field(max_length=2048)] | None = None
    frame_index: Annotated[StrictInt, Field(ge=0)] | None = None
    rgb: StrictBool | None = None
    raw: StrictBool | None = None
    revision: Annotated[StrictInt, Field(ge=1)] | None = None
    prediction_path: Annotated[StrictStr, Field(max_length=8192)] | None = None
    prediction_format: Annotated[StrictStr, Field(max_length=256)] | None = None
    filters: dict | None = None
    segmentation_mode: Literal["off", "ground_truth", "prediction"] | None = None
    segmentation_path: Annotated[StrictStr, Field(max_length=8192)] | None = None
    segmentation_format: Literal["auto", "official", "sonata"] | None = None


def _local_host(host: str, *, tests=False) -> bool:
    if host.lower() == "localhost" or (tests and host.lower() == "testserver"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(*, manager: SessionManager | None = None, factory=None, **launch_options) -> FastAPI:
    """Create the local API without Open3D, glfw or a dataset SDK.

    ``manager``/``factory`` injection is useful for research integrations and
    transport tests. A supplied manager owns its own launch configuration.
    """
    if manager is not None and (factory is not None or launch_options):
        raise ValueError("Supply a session manager or launch options, rather than both.")
    if manager is None:
        options = LaunchOptions(**launch_options)
        manager = SessionManager(options, **({"factory": factory} if factory is not None else {}))

    @asynccontextmanager
    async def lifespan(app):
        async def expire_sessions():
            while True:
                await asyncio.sleep(30)
                manager.expire()

        janitor = asyncio.create_task(expire_sessions())
        try:
            yield
        finally:
            janitor.cancel()
            try:
                await janitor
            except asyncio.CancelledError:
                pass
            await asyncio.to_thread(manager.close)

    app = FastAPI(title="3dviz local viewer", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.sessions = manager

    @app.middleware("http")
    async def local_requests(request, call_next):
        raw_host = request.headers.get("host", "")
        try:
            host = urlsplit("//" + raw_host)
            hostname = host.hostname or ""
            host.port  # Validate a supplied port without trusting a proxy header.
        except ValueError:
            return JSONResponse({"detail": "Invalid local viewer host."}, status_code=400)
        if not _local_host(hostname, tests=True) or host.username is not None or host.password is not None:
            return JSONResponse({"detail": "This viewer only accepts local host requests."}, status_code=400)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin is not None:
                expected = f"{request.url.scheme}://{raw_host}"
                if origin.lower() != expected.lower():
                    return JSONResponse({"detail": "Cross-origin viewer operations are not allowed."}, status_code=403)
            if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
                return JSONResponse({"detail": "Cross-site viewer operations are not allowed."}, status_code=403)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(BrowserError)
    async def browser_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    @app.get("/api/bootstrap")
    def bootstrap():
        return manager.bootstrap()

    @app.post("/api/sessions", status_code=201)
    def create_session():
        return {"id": manager.create().id}

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        manager.delete(session_id)
        return Response(status_code=204)

    @app.get("/api/sessions/{session_id}")
    def session_state(session_id: str):
        return manager.get(session_id).snapshot()

    @app.post("/api/sessions/{session_id}/jobs", status_code=202)
    def start_job(session_id: str, values: JobRequest):
        # Null optional fields mean "unspecified", retaining the last display
        # request during navigation. Pydantic still rejects non-boolean flags.
        return manager.get(session_id).start(values.model_dump(exclude_none=True, exclude_unset=True))

    @app.get("/api/sessions/{session_id}/jobs/{job_id}")
    def job_status(session_id: str, job_id: str):
        return manager.get(session_id).job(job_id)

    @app.get("/api/sessions/{session_id}/frame.bin")
    def frame_packet(session_id: str, revision: Annotated[int, Query(ge=1)]):
        return Response(manager.get(session_id).packet(revision), media_type="application/octet-stream")

    @app.get("/api/sessions/{session_id}/segmentation.bin")
    def semantic_packet(session_id: str, revision: Annotated[int, Query(ge=1)],
                        generation: Annotated[int, Query(ge=0)], native: bool = False):
        labels = manager.get(session_id).semantic_labels(revision, generation, native=native)
        return Response(labels, media_type="application/octet-stream")

    # Only bundled assets are served. Dataset paths are never mounted.
    static = Path(__file__).with_name("static")
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    return app


def run_server(*, host="127.0.0.1", port=8080, open_browser=False, **launch_options):
    """Serve this process's datasets on a loopback URL until interrupted."""
    if not isinstance(host, str) or not _local_host(host):
        raise ValueError("The browser viewer must bind to a loopback host.")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Port must be an integer between 1 and 65535.")
    import uvicorn

    app = create_app(**launch_options)
    url_host = f"[{host}]" if ":" in host else host
    url = f"http://{url_host}:{port}"
    print(f"3dviz browser viewer: {url}", flush=True)
    config = uvicorn.Config(app, host=host, port=port, proxy_headers=False, log_level="info")
    server = uvicorn.Server(config)
    if open_browser:
        def open_when_ready():
            import time
            while not server.started and not server.should_exit:
                time.sleep(0.1)
            if server.started:
                webbrowser.open(url)

        Thread(target=open_when_ready, daemon=True, name="3dviz-open-browser").start()
    server.run()
