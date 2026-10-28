"""Select a shared dataset loader or the local synthetic demonstration."""

from pathlib import Path

from dataset_utils import (
    Capability, DatasetConfig, DatasetConfigurationError, DatasetLoader,
    UnsupportedCapabilityError, available_loaders, create_loader,
)


def available_datasets() -> tuple[str, ...]:
    """List installed providers without importing their payload readers."""
    names = available_loaders()
    if "demo" in names:
        raise DatasetConfigurationError("The viewer uses 'demo' for its synthetic data; choose another provider name.")
    return tuple(sorted((*names, "demo")))


def create_dataset(
    name: str, root: str | Path | None = None, version: str = "v1.0-mini", *,
    config: DatasetConfig | None = None,
) -> DatasetLoader:
    """Open a point-cloud loader; the caller owns and closes the result."""
    if not isinstance(name, str) or not name.strip():
        raise DatasetConfigurationError("Dataset name must be a nonempty string.")
    if config is not None and (root is not None or version != "v1.0-mini"):
        raise DatasetConfigurationError("Pass either DatasetConfig or root/version, not both.")
    key = name.strip().lower()
    available_datasets()
    if key == "demo":
        if config is not None:
            raise DatasetConfigurationError("Demo data does not need a dataset configuration.")
        from .demo import DemoLoader
        return DemoLoader(DatasetConfig())
    release = None if version in (None, "", "v1.0-mini") else version
    loader = create_loader(key, config if config is not None else DatasetConfig(root=root, release=release))
    if Capability.POINT_CLOUD not in loader.metadata.capabilities:
        error = UnsupportedCapabilityError("The selected dataset does not provide point-cloud observations.")
        try:
            loader.close()
        except Exception as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(f"Provider cleanup also failed: {cleanup_error}")
        raise error
    return loader
