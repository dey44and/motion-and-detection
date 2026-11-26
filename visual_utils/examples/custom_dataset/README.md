# 🧪 Research dataset example

Try a complete `DatasetLoader` extension with NPZ point clouds, PNG camera
images, and a JSON manifest. It exposes the `research` loader through the
`visual_utils.loaders` entry-point group so `3dviz` can open the data by name.

**[Step-by-step tutorial](../../docs/custom_datasets.md)** ·
**[Interface reference](../../docs/loader_contract.md)** ·
**[Loader source](src/research_dataset/__init__.py)**

## 🚀 Generate and view a dataset

From the repository root, in an active Python 3.10–3.12 environment:

```bash
python -m pip install -e './visual_utils' \
  -e './visual_utils/examples/custom_dataset' -e './viz_tools/3dviz'
python -m research_dataset.generate /tmp/research-dataset-example
3dviz --dataset research --root /tmp/research-dataset-example --split review
```

The generator requires a new directory. It creates two deterministic frames
with sparse IDs, 1000 points each, a calibrated camera, and one annotated box
per frame. Click **Image** in the 3D scene to inspect the generated camera
image and projected box.

## 🧭 Adapt it to your data

The example package depends on `visual_utils`. The tutorial explains the file
schema, sensor conventions, and how to read the same observations in Python.
[Custom dataset tutorial](../../docs/custom_datasets.md)
