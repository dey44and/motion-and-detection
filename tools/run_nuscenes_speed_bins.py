"""Score cached nuScenes validation detections by ground-truth speed on CPU.

Provide one --model NAME RESULT_JSON per run. Each result JSON must have its
original metrics_summary.json beside it for the unrestricted score check.
"""

import argparse
import csv
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt, load_prediction
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox

from eval_utils.nuscenes_speed_bins import evaluate_speed_bins


SUMMARY_FIELDS = ('mean_ap', 'nd_score', 'label_aps', 'mean_dist_aps', 'label_tp_errors', 'tp_errors', 'tp_scores')


def check_reference(actual, expected, field):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f'Official metric keys differ for {field}')
        for key, value in expected.items():
            check_reference(actual[key], value, f'{field}.{key}')
        return
    if expected is None or math.isnan(expected):
        matches = actual is None
    else:
        matches = actual is not None and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
    if not matches:
        raise ValueError(f'Official score mismatch at {field}: computed {actual}, saved {expected}')


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def score_model(nusc, gt, config, name, prediction_path, reference_path, verify_official):
    with reference_path.open() as stream:
        reference = json.load(stream)
    if reference['cfg'] != config.serialize():
        raise ValueError(f'Saved evaluation config differs from detection_cvpr_2019 for {name}')

    pred, meta = load_prediction(str(prediction_path), config.max_boxes_per_sample, DetectionBox, verbose=True)
    if set(gt.sample_tokens) != set(pred.sample_tokens):
        raise ValueError(f'{name} prediction sample tokens differ from the full validation split')
    pred = filter_eval_boxes(nusc, add_center_dist(nusc, pred), config.class_range, verbose=True)
    result = evaluate_speed_bins(gt, pred, config, verify_official=verify_official)
    for field in SUMMARY_FIELDS:
        check_reference(result['bins']['all'][field], reference[field], f'{name}.{field}')
    result.update(
        model_key=name,
        prediction_path=str(prediction_path),
        prediction_sha256=sha256_file(prediction_path),
        prediction_meta=meta,
        official_reference_summary=str(reference_path),
        official_reference_verified=True,
        eval_set='val',
        nuscenes_devkit_version=version('nuscenes-devkit'),
    )
    return result


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', nargs=2, action='append', metavar=('NAME', 'RESULT_JSON'), required=True)
    parser.add_argument('--dataroot', type=Path, default=repo_root / 'data' / 'nuscenes' / 'v1.0-trainval')
    parser.add_argument('--output-dir', type=Path, required=True, help='destination for model JSONs and summary.csv')
    parser.add_argument('--verify-official', action='store_true', help='also compare metric arrays with SDK accumulate')
    args = parser.parse_args()

    models = {}
    for name, path in args.model:
        if name in ('', '.', '..') or Path(name).name != name or name in models:
            parser.error(f'--model requires unique file-safe names, got {name!r}')
        prediction_path = Path(path).resolve()
        reference_path = prediction_path.with_name('metrics_summary.json')
        for input_path in (prediction_path, reference_path):
            if not input_path.is_file():
                parser.error(f'Missing input for {name}: {input_path}')
        models[name] = (prediction_path, reference_path)

    output_dir = args.output_dir.absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'Preserving existing evaluation at {output_dir}')

    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.dataroot), verbose=True)
    config = config_factory('detection_cvpr_2019')
    gt = load_gt(nusc, 'val', DetectionBox, verbose=True)
    gt = filter_eval_boxes(nusc, add_center_dist(nusc, gt), config.class_range, verbose=True)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent, prefix=f'.{output_dir.name}-') as directory:
        stage = Path(directory)
        with (stage / 'summary.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('model', 'speed_bin', 'label', 'num_gt', 'num_classes', 'mean_ap',
                             'nd_score', 'score_kind', 'mave_mps'))
            for name, (prediction_path, reference_path) in models.items():
                print(f'Evaluating {name}', flush=True)
                result = score_model(nusc, gt, config, name, prediction_path, reference_path, args.verify_official)
                (stage / f'{name}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
                for key, row in result['bins'].items():
                    writer.writerow((name, key, row['label'], row['num_gt'], len(row['macro_classes']), row['mean_ap'],
                                     row['nd_score'], row['score_kind'], row['tp_errors']['vel_err']))
        stage.rename(output_dir)
    print(f'Published speed-bin metrics to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
