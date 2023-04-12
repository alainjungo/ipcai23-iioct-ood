import numpy as np

from . import metrics


AVAIL_DIST_METRICS = ('ACC', 'AE', 'SE', 'ERR', 'AET')


class Evaluator:

    def __init__(self) -> None:
        super().__init__()
        self.aggregation = {}

    def calculate_metrics(self, predictions, targets):
        pass

    @property
    def summary(self) -> dict:
        return {}

    @property
    def case_results(self) -> dict:
        return {}


class DistanceEvaluator(Evaluator):

    def __init__(self, label_names, to_eval=AVAIL_DIST_METRICS, nan_as=0.0) -> None:
        super().__init__()
        if isinstance(label_names, (tuple, list)):
            label_names = {i: k for i, k in enumerate(label_names)}
        self.label_dict = label_names
        self.to_eval = to_eval
        self.nan_as = nan_as
        self._summary = {}

    def calculate_metrics(self, predicted_dists, target_dists):
        case_results = calculate_distance_metrics(predicted_dists, target_dists, self.label_dict, self.to_eval, self.nan_as)
        self.aggregation = case_results
        mean_metrics = ('ACC', 'AE', 'SE', 'ERR', 'AET')
        std_metrics = ('AE', 'SE', 'AET')
        to_summarize = self.label_dict.values()
        self._summary = summarize_metrics(case_results, mean_metrics, std_metrics, to_summarize)
        metrics_to_combine = ('ACC', 'AE', 'SE', 'AET')
        combined = combine_metrics(case_results, metrics_to_combine, to_summarize)
        self._summary.update(combined)

    @property
    def summary(self) -> dict:
        return self._summary

    @property
    def case_results(self) -> dict:
        return self.aggregation


def calculate_distance_metrics(predicted_dists, target_dists, label_names, to_eval=AVAIL_DIST_METRICS, nan_as=0.0):
    if not set(to_eval).issubset(AVAIL_DIST_METRICS):
        raise ValueError(f'metric {set(to_eval) - set(AVAIL_DIST_METRICS)} is not in {AVAIL_DIST_METRICS}')

    if isinstance(label_names, (tuple, list)):
        label_names = {i: k for i, k in enumerate(label_names)}

    aggregation = {}
    for label_idx, label_name in label_names.items():
        predicted_dist, target_dist = predicted_dists[:, label_idx], target_dists[:, label_idx]

        label_results = calculate_distance_metrics_single(predicted_dist, target_dist, to_eval, label_name, nan_as)
        aggregation.update(label_results)

    return aggregation


def calculate_distance_metrics_single(predicted_dist, target_dist, to_eval=AVAIL_DIST_METRICS, label_name=None, nan_as=0.0):
    if not set(to_eval).issubset(AVAIL_DIST_METRICS):
        raise ValueError(f'metric {set(to_eval) - set(AVAIL_DIST_METRICS)} is not in {AVAIL_DIST_METRICS}')

    results = {}
    postfix = f'_{label_name}' if label_name else ''

    pred_has_label, target_has_label = ~np.isnan(predicted_dist), ~np.isnan(target_dist)
    if 'ACC' in to_eval:
        detect = metrics.equality(pred_has_label, target_has_label).astype(np.float)
        results[f'ACC{postfix}'] = detect

    prediction, target = np.nan_to_num(predicted_dist, nan=nan_as), np.nan_to_num(target_dist, nan=nan_as)
    if 'AE' in to_eval:
        ae = metrics.absolute_error(prediction, target)
        results[f'AE{postfix}'] = ae
    if 'SE' in to_eval:
        se = metrics.squared_error(prediction, target)
        results[f'SE{postfix}'] = se
    if 'ERR' in to_eval:
        error = metrics.error(prediction, target)
        results[f'ERR{postfix}'] = error
    if 'AET' in to_eval:
        ae_at_target = metrics.absolute_error(prediction, target)
        ae_at_target[~target_has_label] = np.nan
        results[f'AET{postfix}'] = ae_at_target

    return results


def summarize_metrics(case_results: dict, mean_metrics, std_metrics=(), labels_to_summarize=None):
    summary = {}
    for m in mean_metrics:
        keys = _get_keys(m, labels_to_summarize)
        to_add = {f'{k}_avg': np.nanmean(case_results[k]) for k in keys}  # nan are ignored
        summary.update(to_add)

    for m in std_metrics:
        keys = _get_keys(m, labels_to_summarize)
        to_add = {f'{k}_std': np.nanstd(case_results[k]) for k in keys}
        summary.update(to_add)

    return summary


def combine_metrics(case_results: dict, metrics_to_combine, labels_to_combine=None):
    combined = {}
    for m in metrics_to_combine:
        # nan are ignored
        keys = _get_keys(m, labels_to_combine)
        combined[f'{m}_all_avg'] = np.nanmean([np.nanmean(case_results[k]) for k in keys])
    return combined


def _get_keys(metric: str, label_names=None):
    if label_names is None:
        return [metric]
    return [f'{metric}_{label_name}' for label_name in label_names]
