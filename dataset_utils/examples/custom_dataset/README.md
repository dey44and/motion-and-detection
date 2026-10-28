# Research dataset example

An executable `DatasetLoader` extension using NPZ point clouds, PNG camera
images and a JSON manifest. It exposes the `research` loader through the
`dataset_utils.loaders` entry-point group. Install `dataset-utils` first, then
install this directory with `python -m pip install -e .`.

```bash
python -m research_dataset.generate /tmp/research-dataset-example
3dviz --dataset research --root /tmp/research-dataset-example --split review
```

The generator requires a new directory. Install the applicable viewer to run
`2dcameraviz`, `3dviz` or `3ddetviz`; this example depends only on the shared
dataset package. See the repository's
[custom dataset guide](../../docs/custom_datasets.md) for the file schema,
scientific assumptions, source code walkthrough and all three launch commands.
