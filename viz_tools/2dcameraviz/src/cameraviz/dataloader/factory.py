"""Camera source selection: the shared registry plus the generated example."""

from pathlib import Path

from visual_utils import (
    Capability, DatasetConfig, DatasetConfigurationError, DatasetLoader,
    UnsupportedCapabilityError, available_loaders, create_loader,
)


def available_datasets() -> tuple[str, ...]:
    names = available_loaders()
    if "demo" in names:
        raise DatasetConfigurationError("The camera viewer reserves 'demo' for its generated example.")
    return tuple(sorted((*names, "demo")))


def create_dataset(
    name: str, root: str | Path | None = None, version: str = "v1.0-mini", *,
    config: DatasetConfig | None = None,
) -> DatasetLoader:
    """Open a shared camera loader. The caller owns and closes the result."""
    if config is not None and (root is not None or version != "v1.0-mini"):
        raise DatasetConfigurationError("Pass either DatasetConfig or root/version, not both.")
    key = name.strip().lower()
    available_datasets()
    if key == "demo":
        if config is not None:
            raise DatasetConfigurationError("Demo data do not take a dataset configuration.")
        from .demo import DemoDataset
        return DemoDataset()
    release = None if version in (None, "", "v1.0-mini") else version
    loader = create_loader(key, config or DatasetConfig(root=root, release=release))
    if Capability.CAMERAS not in loader.metadata.capabilities:
        loader.close()
        raise UnsupportedCapabilityError("The selected dataset does not provide camera observations.")
    return loader
