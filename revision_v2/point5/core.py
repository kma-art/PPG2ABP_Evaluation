"""Statistical primitives for revision point 5 (Python 3.6 compatible)."""

from __future__ import division, print_function

import math

import numpy as np


SEED = 20260908
BOOTSTRAP_REPLICATES = 10000
LOWER_LIMIT = 20.0
UPPER_LIMIT = 300.0
METRICS = ("waveform", "sbp_abs", "dbp_abs", "map_abs",
           "sbp_signed", "dbp_signed", "map_signed")
ABS_METRICS = ("waveform", "sbp_abs", "dbp_abs", "map_abs")
SCENARIOS = ("raw", "clip", "reject", "oracle_top5", "gt_50_200")


def descriptive(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty vector")
    return {"mean": float(np.mean(values)), "sd": float(np.std(values)),
            "n": int(len(values))}


def percentile_interval(values):
    values = np.asarray(values, dtype=np.float64)
    return [float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5))]


def weighted_quantile(values, weights, quantiles):
    """Empirical weighted quantiles using the first CDF value at or above q."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights):
        raise ValueError("values and weights must be equal vectors")
    if len(values) == 0 or np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("positive total weight is required")
    if np.any(quantiles < 0) or np.any(quantiles > 1):
        raise ValueError("quantiles must be in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    targets = quantiles * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    indices = np.minimum(indices, len(sorted_values) - 1)
    return sorted_values[indices]


def cluster_layout(clusters):
    clusters = np.asarray(clusters)
    names = sorted(set(str(value) for value in clusters))
    lookup = dict((name, index) for index, name in enumerate(names))
    indices = np.asarray([lookup[str(value)] for value in clusters], dtype=np.int64)
    counts = np.bincount(indices, minlength=len(names)).astype(np.float64)
    return names, indices, counts


def cluster_means(values, clusters):
    values = np.asarray(values, dtype=np.float64)
    names, indices, counts = cluster_layout(clusters)
    totals = np.bincount(indices, weights=values, minlength=len(names))
    return names, totals / counts


def bootstrap_mean_ci(cluster_values, seed=SEED, replicates=BOOTSTRAP_REPLICATES):
    values = np.asarray(cluster_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("cluster_values must be a non-empty vector")
    rng = np.random.RandomState(int(seed))
    boot = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), 128):
        stop = min(start + 128, int(replicates))
        indices = rng.randint(0, len(values), size=(stop - start, len(values)))
        boot[start:stop] = np.mean(values[indices], axis=1)
    return percentile_interval(boot)


def metric_statistics(values, clusters, seed=SEED,
                      replicates=BOOTSTRAP_REPLICATES):
    _, grouped = cluster_means(values, clusters)
    result = {"segment_weighted": descriptive(values),
              "cluster_equal": descriptive(grouped)}
    result["cluster_equal"]["clusters"] = int(len(grouped))
    result["cluster_equal"]["ci95"] = bootstrap_mean_ci(
        grouped, seed=seed, replicates=replicates)
    return result


def scenario_masks(columns):
    count = len(columns["raw_waveform"])
    raw = np.ones(count, dtype=bool)
    outside = np.asarray(columns["pred_outside_20_300"], dtype=bool)
    waveform = np.asarray(columns["raw_waveform"], dtype=np.float64)
    threshold = float(np.percentile(waveform, 95))
    gt_valid = ((np.asarray(columns["gt_min"], dtype=np.float64) >= 50.0) &
                (np.asarray(columns["gt_max"], dtype=np.float64) <= 200.0))
    return ({"raw": raw, "clip": raw.copy(), "reject": ~outside,
             "oracle_top5": waveform <= threshold, "gt_50_200": gt_valid},
            threshold)


def _tail_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(values, [25, 50, 75, 90, 95, 99, 99.5])
    result = {
        "q1": float(percentiles[0]), "median": float(percentiles[1]),
        "q3": float(percentiles[2]), "p90": float(percentiles[3]),
        "p95": float(percentiles[4]), "p99": float(percentiles[5]),
        "p99_5": float(percentiles[6]), "maximum": float(np.max(values)),
        "counts_above": {},
    }
    for threshold in (30, 50, 100, 200, 500):
        result["counts_above"][str(threshold)] = int(np.sum(values > threshold))
    return result


def _failure_concentration(event, clusters):
    event = np.asarray(event, dtype=bool)
    clusters = np.asarray(clusters)
    total = int(np.sum(event))
    if total == 0:
        return {"events": 0, "affected_clusters": 0,
                "largest_cluster_events": 0, "top_10_clusters_share": 0.0}
    _, indices, _ = cluster_layout(clusters)
    counts = np.bincount(indices[event])
    counts = np.sort(counts[counts > 0])[::-1]
    return {"events": total, "affected_clusters": int(len(counts)),
            "largest_cluster_events": int(counts[0]),
            "top_10_clusters_share": float(np.sum(counts[:10]) / total)}


def scenario_statistics(columns, scenario, mask, seed=SEED,
                        replicates=BOOTSTRAP_REPLICATES):
    prefix = "clip_" if scenario == "clip" else "raw_"
    mask = np.asarray(mask, dtype=bool)
    included = np.asarray(columns["cluster_included"], dtype=bool)
    cluster_mask = mask & included
    all_clusters = np.asarray(columns["cluster_id"])[included]
    selected_clusters = np.asarray(columns["cluster_id"])[cluster_mask]
    total_cluster_count = len(set(str(value) for value in all_clusters))
    selected_cluster_count = len(set(str(value) for value in selected_clusters))
    if not np.any(mask) or not np.any(cluster_mask):
        raise ValueError("scenario leaves no observations")
    result = {
        "coverage": {
            "segments_total": int(len(mask)), "segments_retained": int(np.sum(mask)),
            "segment_fraction": float(np.mean(mask)),
            "clusters_total": int(total_cluster_count),
            "clusters_retained": int(selected_cluster_count),
            "cluster_fraction": float(selected_cluster_count / total_cluster_count),
            "cluster_excluded_segments": int(np.sum(mask & ~included)),
        },
        "metrics": {}, "bhs": {}, "tails": {},
    }
    vectors, descriptors = [], []
    for metric in METRICS:
        vectors.append(np.asarray(columns[prefix + metric], dtype=np.float64))
        descriptors.append(("metric", metric, None))
    for pressure in ("sbp", "dbp", "map"):
        result["bhs"][pressure] = {}
        values = np.asarray(columns[prefix + pressure + "_abs"], dtype=np.float64)
        for threshold in (5, 10, 15):
            vectors.append((values <= threshold).astype(np.float64) * 100.0)
            descriptors.append(("bhs", pressure, "le{0}".format(threshold)))
    matrix = np.column_stack([vector[cluster_mask] for vector in vectors])
    names, cluster_index, cluster_counts = cluster_layout(selected_clusters)
    cluster_matrix = np.zeros((len(names), matrix.shape[1]), dtype=np.float64)
    for column in range(matrix.shape[1]):
        cluster_matrix[:, column] = (np.bincount(
            cluster_index, weights=matrix[:, column], minlength=len(names)) /
            cluster_counts)
    rng = np.random.RandomState(int(seed))
    boot = np.empty((int(replicates), matrix.shape[1]), dtype=np.float64)
    probabilities = np.full(len(names), 1.0 / len(names), dtype=np.float64)
    for start in range(0, int(replicates), 64):
        stop = min(start + 64, int(replicates))
        multiplicity = rng.multinomial(len(names), probabilities, size=stop - start)
        boot[start:stop] = np.dot(multiplicity, cluster_matrix) / len(names)
    for column, descriptor in enumerate(descriptors):
        original = vectors[column]
        stat = {"segment_weighted": descriptive(original[mask]),
                "cluster_equal": descriptive(cluster_matrix[:, column])}
        stat["cluster_equal"]["clusters"] = int(len(names))
        stat["cluster_equal"]["ci95"] = percentile_interval(boot[:, column])
        if descriptor[0] == "metric":
            result["metrics"][descriptor[1]] = stat
        else:
            result["bhs"][descriptor[1]][descriptor[2]] = stat
    for metric in ABS_METRICS:
        values = np.asarray(columns[prefix + metric], dtype=np.float64)
        result["tails"][metric] = _tail_statistics(values[mask])
    outside = np.asarray(columns["pred_outside_20_300"], dtype=bool)
    if scenario == "clip":
        outside = np.zeros(len(outside), dtype=bool)
    selected_outside = outside & mask
    result["prediction_outside_20_300"] = _failure_concentration(
        selected_outside[cluster_mask], selected_clusters)
    waveform = np.asarray(columns[prefix + "waveform"], dtype=np.float64)
    result["waveform_failures"] = {}
    for threshold in (30, 50, 100, 200, 500):
        result["waveform_failures"][str(threshold)] = _failure_concentration(
            (waveform[cluster_mask] > threshold), selected_clusters)
    return result


def pair_statistics(reference, prediction):
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    pearson, r2, slope, intercept = _pair_stats(reference, prediction)
    spearman = _pair_stats(_rankdata(reference), _rankdata(prediction))[0]
    return {"pearson_r": float(pearson), "spearman_rho": float(spearman),
            "r2": float(r2), "calibration_slope": float(slope),
            "calibration_intercept": float(intercept),
            "segments": int(len(reference))}


def _pair_stats(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean, y_mean = np.mean(x), np.mean(y)
    dx, dy = x - x_mean, y - y_mean
    ssx, ssy = np.sum(dx * dx), np.sum(dy * dy)
    covariance = np.sum(dx * dy)
    pearson = covariance / math.sqrt(ssx * ssy) if ssx > 0 and ssy > 0 else np.nan
    slope = covariance / ssx if ssx > 0 else np.nan
    intercept = y_mean - slope * x_mean
    r2 = 1.0 - np.sum((y - x) ** 2) / ssx if ssx > 0 else np.nan
    return pearson, r2, slope, intercept


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def correlation_statistics(reference, prediction, seed=SEED,
                           replicates=BOOTSTRAP_REPLICATES):
    """Pair statistics and a deterministic cluster-pair bootstrap."""
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != 1 or len(reference) < 2:
        raise ValueError("equal non-trivial vectors required")
    pearson, r2, slope, intercept = _pair_stats(reference, prediction)
    ranks_x, ranks_y = _rankdata(reference), _rankdata(prediction)
    spearman = _pair_stats(ranks_x, ranks_y)[0]
    rng = np.random.RandomState(int(seed))
    boot = np.empty((int(replicates), 5), dtype=np.float64)
    count = len(reference)
    probabilities = np.full(count, 1.0 / count, dtype=np.float64)
    features = np.column_stack((reference, prediction, reference ** 2,
                                prediction ** 2, reference * prediction,
                                (prediction - reference) ** 2,
                                ranks_x, ranks_y, ranks_x ** 2,
                                ranks_y ** 2, ranks_x * ranks_y))
    for start in range(0, int(replicates), 128):
        stop = min(start + 128, int(replicates))
        multiplicity = rng.multinomial(count, probabilities, size=stop - start)
        sums = np.dot(multiplicity, features)
        mx, my = sums[:, 0] / count, sums[:, 1] / count
        ssx = sums[:, 2] - count * mx * mx
        ssy = sums[:, 3] - count * my * my
        covariance = sums[:, 4] - count * mx * my
        boot[start:stop, 0] = covariance / np.sqrt(ssx * ssy)
        boot[start:stop, 2] = 1.0 - sums[:, 5] / ssx
        boot[start:stop, 3] = covariance / ssx
        boot[start:stop, 4] = my - boot[start:stop, 3] * mx
        rmx, rmy = sums[:, 6] / count, sums[:, 7] / count
        rssx = sums[:, 8] - count * rmx * rmx
        rssy = sums[:, 9] - count * rmy * rmy
        rcov = sums[:, 10] - count * rmx * rmy
        boot[start:stop, 1] = rcov / np.sqrt(rssx * rssy)
    names = ("pearson_r", "spearman_rho", "r2", "calibration_slope",
             "calibration_intercept")
    values = (pearson, spearman, r2, slope, intercept)
    result = {}
    for index, name in enumerate(names):
        result[name] = {"estimate": float(values[index]),
                        "ci95": percentile_interval(boot[:, index])}
    result["clusters"] = int(len(reference))
    result["bootstrap_method"] = (
        "cluster-pair percentile bootstrap; Spearman uses fixed full-sample midranks")
    return result


def calibration_bins(reference, prediction, clusters, minimum_clusters=10):
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    clusters = np.asarray(clusters)
    rows = []
    for lower in range(20, 300, 20):
        upper = lower + 20
        mask = ((reference >= lower) &
                ((reference < upper) if upper < 300 else (reference <= upper)))
        names = sorted(set(str(value) for value in clusters[mask]))
        ref_means, pred_means = [], []
        for name in names:
            local = mask & (np.asarray([str(value) for value in clusters]) == name)
            ref_means.append(float(np.mean(reference[local])))
            pred_means.append(float(np.mean(prediction[local])))
        rows.append({"lower": float(lower), "upper": float(upper),
                     "segments": int(np.sum(mask)), "clusters": int(len(names)),
                     "shown": bool(len(names) >= minimum_clusters),
                     "reference_mean": (float(np.mean(ref_means)) if names else None),
                     "prediction_mean": (float(np.mean(pred_means)) if names else None)})
    return rows


def bland_altman_statistics(differences, clusters, seed=SEED,
                            replicates=BOOTSTRAP_REPLICATES):
    differences = np.asarray(differences, dtype=np.float64)
    clusters = np.asarray(clusters)
    names, cluster_index, counts = cluster_layout(clusters)
    cluster_mean = (np.bincount(cluster_index, weights=differences,
                                minlength=len(names)) / counts)
    cluster_second = (np.bincount(cluster_index, weights=differences ** 2,
                                  minlength=len(names)) / counts)
    weights = 1.0 / (len(names) * counts[cluster_index])
    bias = float(np.mean(cluster_mean))
    second = float(np.mean(cluster_second))
    sd = math.sqrt(max(0.0, second - bias * bias))
    traditional = [bias - 1.96 * sd, bias + 1.96 * sd]
    nonparametric = weighted_quantile(differences, weights, [0.025, 0.975])

    order = np.argsort(differences, kind="mergesort")
    sorted_values = differences[order]
    sorted_cluster_index = cluster_index[order]
    rng = np.random.RandomState(int(seed))
    boot = np.empty((int(replicates), 5), dtype=np.float64)
    probabilities = np.full(len(names), 1.0 / len(names), dtype=np.float64)
    for start in range(0, int(replicates), 32):
        stop = min(start + 32, int(replicates))
        size = stop - start
        multiplicity = rng.multinomial(len(names), probabilities, size=size)
        means = np.sum(multiplicity * cluster_mean, axis=1) / len(names)
        seconds = np.sum(multiplicity * cluster_second, axis=1) / len(names)
        sds = np.sqrt(np.maximum(0.0, seconds - means * means))
        boot[start:stop, 0] = means
        boot[start:stop, 1] = means - 1.96 * sds
        boot[start:stop, 2] = means + 1.96 * sds
        sorted_weights = (multiplicity[:, sorted_cluster_index] /
                          counts[sorted_cluster_index])
        cumulative = np.cumsum(sorted_weights, axis=1)
        totals = cumulative[:, -1]
        for target_column, quantile in ((3, 0.025), (4, 0.975)):
            positions = np.argmax(cumulative >= (totals * quantile)[:, None], axis=1)
            boot[start:stop, target_column] = sorted_values[positions]
    return {
        "clusters": int(len(names)), "bias": bias,
        "bias_ci95": percentile_interval(boot[:, 0]),
        "traditional_limits": [float(traditional[0]), float(traditional[1])],
        "traditional_lower_ci95": percentile_interval(boot[:, 1]),
        "traditional_upper_ci95": percentile_interval(boot[:, 2]),
        "nonparametric_limits": [float(nonparametric[0]), float(nonparametric[1])],
        "nonparametric_lower_ci95": percentile_interval(boot[:, 3]),
        "nonparametric_upper_ci95": percentile_interval(boot[:, 4]),
        "weighting": "equal total weight per cluster; equal weight within cluster",
    }
