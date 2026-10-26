"""Keep the public 3D adapter imports compatible with the shared backend."""


def test_legacy_nuscenes_utility_modules_refer_to_shared_implementations():
    from dataset_utils.nuscenes import motion, projection, splits
    from threedviz.processing import motion as legacy_motion
    from threedviz.processing import projection as legacy_projection
    from threedviz.dataloader import splits as legacy_splits

    assert legacy_motion is motion
    assert legacy_projection is projection
    assert legacy_splits is splits


def test_dataset_errors_are_shared_with_application_adapters():
    from dataset_utils.errors import DatasetError
    from threedviz.dataloader import DatasetError as ViewerDatasetError

    assert ViewerDatasetError is DatasetError


def test_public_frame_models_are_shared_without_changing_adapter_imports():
    from dataset_utils.models import BoundingBox3D, FrameInfo, PointCloudFrame, SceneInfo
    from threedviz.dataloader import models

    assert models.BoundingBox3D is BoundingBox3D
    assert models.FrameInfo is FrameInfo
    assert models.FrameData is PointCloudFrame
    assert models.SceneInfo is SceneInfo
