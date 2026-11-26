"""GT-speed diagnostics ported from pc-corrector's nuScenes speed-bin evaluator.

Inputs must already have undergone the official DetectionEval eligibility
filters. Matching is performed against *all* GT before selecting a speed bin.
A match to another bin is ignored. Globally unmatched detections (including
duplicates) are false positives in every bin. Selection uses GT speed.
Only the unrestricted result uses the official nuScenes averaging rules.

Matching and metric interpolation follow the nuScenes devkit's ``accumulate``:
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py
The SDK's error functions, cummean, calc_ap and calc_tp are reused directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from nuscenes.eval.common.utils import (
    attr_acc, center_distance, cummean, scale_iou, velocity_l2, yaw_diff,
)
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionMetricData, DetectionMetrics


BIN_KEYS = ('0_1', '1_4', 'gt4', 'unknown')
BIN_LABELS = {
    'all': 'All eligible objects (official aggregate)',
    '0_1': '0 <= GT speed <= 1 m/s',
    '1_4': '1 < GT speed <= 4 m/s',
    'gt4': 'GT speed > 4 m/s',
    'unknown': 'Unknown GT planar speed',
}


def _speed_bin(box) -> int:
    vx, vy = box.velocity
    if not math.isfinite(vx) or not math.isfinite(vy):
        return 3
    speed = math.hypot(vx, vy)
    if speed <= 1.0:
        return 0
    return 1 if speed <= 4.0 else 2


def _json_number(value) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(value) else None


def _applicable(class_name: str, metric_name: str) -> bool:
    if class_name == 'traffic_cone' and metric_name in ('attr_err', 'vel_err', 'orient_err'):
        return False
    if class_name == 'barrier' and metric_name in ('attr_err', 'vel_err'):
        return False
    return True


@dataclass
class _Matching:
    scores: np.ndarray
    # -1 marks an unmatched prediction; 0..3 marks its matched GT speed bin.
    matched_bins: np.ndarray
    # Error vectors include only globally matched predictions, in score order.
    errors: dict[str, np.ndarray]


def _match_class(ordered_predictions, sample_gt, dist_fcn, dist_th, class_name):
    scores = np.asarray([box.detection_score for box in ordered_predictions], dtype=float)
    matched_bins = np.full(len(ordered_predictions), -1, dtype=np.int8)
    errors = {key: [] for key in TP_METRICS}
    taken = set()
    period = np.pi if class_name == 'barrier' else 2 * np.pi

    for pred_index, pred in enumerate(ordered_predictions):
        min_dist = np.inf
        match_gt_idx = None
        candidates = sample_gt.get(pred.sample_token, ())
        # Retaining original GT order preserves the SDK's distance-tie rule.
        for gt_idx, (gt, _) in enumerate(candidates):
            if (pred.sample_token, gt_idx) in taken:
                continue
            distance = dist_fcn(gt, pred)
            if distance < min_dist:
                min_dist = distance
                match_gt_idx = gt_idx
        if min_dist < dist_th:  # The official criterion is strictly less than.
            taken.add((pred.sample_token, match_gt_idx))
            gt, speed_bin = candidates[match_gt_idx]
            matched_bins[pred_index] = speed_bin
            errors['trans_err'].append(center_distance(gt, pred))
            errors['vel_err'].append(velocity_l2(gt, pred))
            errors['scale_err'].append(1 - scale_iou(gt, pred))
            errors['orient_err'].append(yaw_diff(gt, pred, period=period))
            errors['attr_err'].append(1 - attr_acc(gt, pred))

    return _Matching(scores, matched_bins, {
        name: np.asarray(values, dtype=float) for name, values in errors.items()
    })


def _bin_metric_data(matching: _Matching, num_gt: int, bin_index: int | None):
    """Select a cohort after matching, then use the SDK interpolation exactly."""
    globally_matched = matching.matched_bins >= 0
    if bin_index is None:
        keep = np.ones(len(matching.scores), dtype=bool)
        positive = globally_matched
        keep_match_errors = np.ones(int(globally_matched.sum()), dtype=bool)
    else:
        positive = matching.matched_bins == bin_index
        keep = positive | ~globally_matched
        keep_match_errors = matching.matched_bins[globally_matched] == bin_index

    num_tp = int(positive.sum())
    num_fp = int((~globally_matched).sum())
    counts = {
        'tp': num_tp,
        'fp': num_fp,
        'ignored': int((~keep).sum()),
        'scored_predictions': num_tp + num_fp,
        'recall': float(num_tp / num_gt) if num_gt else None,
    }
    if num_gt == 0 or num_tp == 0:
        return DetectionMetricData.no_predictions(), counts

    tp = np.cumsum(positive[keep]).astype(float)
    fp = np.cumsum((~globally_matched)[keep]).astype(float)
    confidence = matching.scores[keep]
    precision = tp / (tp + fp)
    recall = tp / float(num_gt)
    recall_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    precision = np.interp(recall_interp, recall, precision, right=0)
    confidence = np.interp(recall_interp, recall, confidence, right=0)
    match_confidence = matching.scores[positive]
    errors = {}
    for name, values in matching.errors.items():
        means = cummean(values[keep_match_errors])
        errors[name] = np.interp(
            confidence[::-1], match_confidence[::-1], means[::-1],
        )[::-1]
    return DetectionMetricData(
        recall=recall_interp, precision=precision, confidence=confidence, **errors,
    ), counts


def _assert_same_metric_data(actual, expected, context):
    for key in expected.serialize():
        np.testing.assert_allclose(
            getattr(actual, key), getattr(expected, key), rtol=0, atol=0,
            equal_nan=True, err_msg=f'{context}: {key}',
        )


def _finish_bin(result, cfg, is_official):
    classes = list(cfg.class_names)
    result['supported_classes'] = [c for c in classes if result['per_class'][c]['num_gt'] > 0]
    macro_classes = classes if is_official else result['supported_classes']
    result['macro_classes'] = macro_classes
    result['num_gt'] = int(sum(entry['num_gt'] for entry in result['per_class'].values()))
    result['mean_dist_aps'] = {c: result['per_class'][c]['mean_ap'] for c in classes}
    result['label_aps'] = {c: result['per_class'][c]['ap_by_distance'] for c in classes}
    result['label_tp_errors'] = {c: result['per_class'][c]['tp_errors'] for c in classes}
    result['mean_ap'] = (
        float(np.mean([result['per_class'][c]['mean_ap'] for c in macro_classes]))
        if macro_classes else None
    )
    result['tp_errors'] = {}
    result['tp_error_macro_classes'] = {}
    raw_macro_errors = {}
    for metric in TP_METRICS:
        # Match SDK nanmean: NaNs are omitted; infinity must still contribute
        # its zero TP score. JSON null conversion is only a presentation step.
        eligible = [c for c in macro_classes if not math.isnan(result['per_class'][c]['_raw_tp_errors'][metric])]
        result['tp_error_macro_classes'][metric] = eligible
        raw_macro_errors[metric] = (
            float(np.mean([result['per_class'][c]['_raw_tp_errors'][metric] for c in eligible]))
            if eligible else None
        )
        result['tp_errors'][metric] = _json_number(raw_macro_errors[metric])
    result['tp_scores'] = {
        name: max(0.0, 1.0 - value) if value is not None else None
        for name, value in raw_macro_errors.items()
    }
    if result['mean_ap'] is not None and all(v is not None for v in result['tp_scores'].values()):
        result['nd_score'] = float(
            (cfg.mean_ap_weight * result['mean_ap'] + np.sum(list(result['tp_scores'].values())))
            / (cfg.mean_ap_weight + len(TP_METRICS))
        )
    else:
        result['nd_score'] = None
    result['score_kind'] = 'official_aggregate' if is_official else 'custom_gt_speed_diagnostic'
    for entry in result['per_class'].values():
        entry['tp_error_nonfinite'] = {
            name: ('infinity' if math.isinf(value) else None)
            for name, value in entry.pop('_raw_tp_errors').items()
        }


def evaluate_speed_bins(gt_boxes, pred_boxes, cfg, verify_official: bool = False) -> dict[str, Any]:
    """Evaluate already-filtered EvalBoxes without disk I/O or mutation.

    Returns JSON-safe summaries in ``result['bins']`` under keys ``all``,
    ``0_1``, ``1_4``, ``gt4``, and ``unknown``. The three finite-speed bins use
    supported-class macros. Their absent classes have null metrics. ``all``
    retains official ten-class averaging, including the SDK's absent-class
    penalties, so it can reproduce an ordinary DetectionEval evaluation.

    The unknown group has no velocity error or NDS-like composite: its GT
    velocity is unavailable. Other unknown-group metrics are diagnostic.

    ``verify_official=True`` independently invokes SDK ``accumulate`` for every
    class and threshold and asserts exact equality of all 101-point metric
    arrays and aggregate scalars. This costs an additional official evaluation.
    """
    if set(gt_boxes.sample_tokens) != set(pred_boxes.sample_tokens):
        raise ValueError('GT and prediction sample tokens must match exactly.')
    if cfg.dist_th_tp not in cfg.dist_ths:
        raise ValueError('The TP error threshold must occur in dist_ths.')

    classes = list(cfg.class_names)
    gt_by_class = {name: {} for name in classes}
    pred_by_class = {name: [] for name in classes}
    gt_counts = {name: np.zeros(len(BIN_KEYS), dtype=np.int64) for name in classes}
    # Flatten once, preserving exactly EvalBoxes.all order for SDK score ties.
    for token in gt_boxes.sample_tokens:
        for gt in gt_boxes[token]:
            speed_bin = _speed_bin(gt)
            gt_by_class[gt.detection_name].setdefault(token, []).append((gt, speed_bin))
            gt_counts[gt.detection_name][speed_bin] += 1
    for token in pred_boxes.sample_tokens:
        for pred in pred_boxes[token]:
            pred_by_class[pred.detection_name].append(pred)

    bins = {
        key: {'label': BIN_LABELS[key], 'per_class': {}}
        for key in ('all', *BIN_KEYS)
    }
    sdk_metrics = DetectionMetrics(cfg) if verify_official else None
    checked_arrays = 0
    for class_name in classes:
        predictions = pred_by_class[class_name]
        # SDK sorts (confidence, original_index), then reverses the whole list.
        order = [i for _, i in sorted((box.detection_score, i) for i, box in enumerate(predictions))][::-1]
        ordered_predictions = [predictions[i] for i in order]
        for key, entry in bins.items():
            count = int(gt_counts[class_name].sum()) if key == 'all' else int(gt_counts[class_name][BIN_KEYS.index(key)])
            entry['per_class'][class_name] = {
                'num_gt': count,
                'num_predictions_before_bin_selection': len(predictions),
                'ap_by_distance': {}, 'counts_by_distance': {},
                'tp_errors': {}, '_raw_tp_errors': {}, 'tp_error_valid_matches': {},
            }

        for distance in cfg.dist_ths:
            distance_key = str(float(distance))
            matching = _match_class(
                ordered_predictions, gt_by_class[class_name],
                cfg.dist_fcn_callable, distance, class_name,
            )
            for key, entry in bins.items():
                stats = entry['per_class'][class_name]
                bin_index = None if key == 'all' else BIN_KEYS.index(key)
                md, counts = _bin_metric_data(matching, stats['num_gt'], bin_index)
                stats['counts_by_distance'][distance_key] = counts
                metric_defined = key == 'all' or stats['num_gt'] > 0
                ap = calc_ap(md, cfg.min_recall, cfg.min_precision)
                stats['ap_by_distance'][distance_key] = float(ap) if metric_defined else None
                if distance == cfg.dist_th_tp:
                    stats['tp_count_at_tp_threshold'] = counts['tp']
                    stats['recall_at_tp_threshold'] = counts['recall']
                    stats['tp_error_uses_low_recall_penalty'] = (
                        bool(md.max_recall_ind < round(100 * cfg.min_recall) + 1)
                        if metric_defined else None
                    )
                    globally_matched = matching.matched_bins >= 0
                    selected_matches = (
                        np.ones(int(globally_matched.sum()), dtype=bool) if key == 'all'
                        else matching.matched_bins[globally_matched] == bin_index
                    )
                    for metric in TP_METRICS:
                        defined = metric_defined and _applicable(class_name, metric)
                        if key == 'unknown' and metric == 'vel_err':
                            defined = False
                        tp = calc_tp(md, cfg.min_recall, metric) if defined else None
                        stats['_raw_tp_errors'][metric] = float(tp) if tp is not None else float('nan')
                        stats['tp_errors'][metric] = _json_number(tp)
                        stats['tp_error_valid_matches'][metric] = (
                            int(np.isfinite(matching.errors[metric][selected_matches]).sum())
                            if _applicable(class_name, metric) else 0
                        )
                if key == 'all' and verify_official:
                    official_md = accumulate(
                        gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, distance,
                    )
                    _assert_same_metric_data(md, official_md, f'{class_name}/{distance}')
                    checked_arrays += len(official_md.serialize())
                    sdk_metrics.add_label_ap(class_name, distance, calc_ap(
                        official_md, cfg.min_recall, cfg.min_precision,
                    ))
                    if distance == cfg.dist_th_tp:
                        for metric in TP_METRICS:
                            value = calc_tp(official_md, cfg.min_recall, metric) if _applicable(class_name, metric) else np.nan
                            sdk_metrics.add_label_tp(class_name, metric, value)

        for key, entry in bins.items():
            stats = entry['per_class'][class_name]
            aps = list(stats['ap_by_distance'].values())
            stats['mean_ap'] = float(np.mean(aps)) if all(v is not None for v in aps) else None

    for key, entry in bins.items():
        _finish_bin(entry, cfg, is_official=key == 'all')

    verification = {'requested': bool(verify_official), 'passed': None, 'metric_arrays_checked': 0}
    if verify_official:
        all_result = bins['all']
        for actual, expected in (
            (all_result['mean_ap'], sdk_metrics.mean_ap),
            (all_result['nd_score'], sdk_metrics.nd_score),
            *((all_result['tp_errors'][name], value) for name, value in sdk_metrics.tp_errors.items()),
        ):
            assert _json_number(actual) == _json_number(expected), (actual, expected)
        verification.update(passed=True, metric_arrays_checked=checked_arrays)

    total_gt = bins['all']['num_gt']
    assert sum(bins[key]['num_gt'] for key in BIN_KEYS) == total_gt
    return {
        'protocol': {
            'name': 'nuscenes_global_matching_then_gt_speed_bins_v1',
            'gt_speed': 'hypot(vx, vy) in m/s from the SDK GT annotation velocity; no ego-speed subtraction',
            'speed_boundaries': {'0_1': '[0, 1]', '1_4': '(1, 4]', 'gt4': '(4, infinity)', 'unknown': 'any nonfinite velocity component'},
            'input_requirement': 'Official DetectionEval range, points and bicycle-rack filters already applied',
            'matching': 'Official greedy confidence order, same class, closest unmatched GT, strict distance threshold; all GT participate before bin selection',
            'out_of_bin_match': 'Ignored in that bin; matched GT is still consumed globally',
            'unmatched_prediction': 'False positive in every bin, including duplicates and wrong-class predictions',
            'predicted_speed_selection': False,
            'bin_macro': 'Classes with at least one GT in that bin; unsupported classes and errors are null',
            'all_macro': 'Official SDK class set, including absent-class penalties',
            'unknown_velocity_error': 'Undefined for the unknown group; no NDS-like composite for this group',
            'comparability': 'Custom bin diagnostics share a common unmatched-FP pool and do not aggregate to official mAP/NDS; compare models within a bin',
            'tp_error_low_recall': 'SDK calc_tp returns 1 when supported-class recall does not reach above min_recall; this is a score penalty, not a measured mean error',
            'error_units': {'trans_err': 'm', 'scale_err': '1 - scale IoU', 'orient_err': 'rad', 'vel_err': 'm/s', 'attr_err': '1 - attribute accuracy'},
        },
        'config': cfg.serialize(),
        'num_samples': len(gt_boxes.sample_tokens),
        'num_predictions': sum(len(v) for v in pred_by_class.values()),
        'bins': bins,
        'verification': verification,
    }
