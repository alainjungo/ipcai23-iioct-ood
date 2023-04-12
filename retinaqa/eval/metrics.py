import numpy as np
import sklearn.covariance as sk_cov
import sklearn.metrics as m


def squared_error(predictions: np.ndarray, targets: np.ndarray):
    return (predictions - targets) ** 2


def error(predictions: np.ndarray, targets: np.ndarray):
    return predictions - targets


def absolute_error(predictions: np.ndarray, targets: np.ndarray):
    return np.abs(predictions - targets)


def equality(predictions: np.ndarray, targets: np.ndarray):
    return predictions == targets


def maha_distance_2d(features, mean, cov_inv):
    feat_no_mean = features - mean
    dists_sq = (feat_no_mean[:, np.newaxis] @ cov_inv[np.newaxis] @ feat_no_mean[:, :, np.newaxis]).squeeze()  # (N, 1, F) x (1, F, F) x (N, F, 1)  -> (N, 1, 1)
    dists = np.sqrt(dists_sq)
    return dists


def get_maha_prams(features, cov_method='lw', inv_method='linalg'):
    mean = features.mean(axis=0)

    if cov_method == 'lw':
        ledoit_wolf = sk_cov.LedoitWolf()
        ledoit_wolf.fit(features)
        cov = ledoit_wolf.covariance_
    elif cov_method == 'emp':
        cov = np.cov(features.T)
    else:
        raise ValueError(f'cov_method = {cov_method} is unknown. Supported are (lw, emp)')

    if inv_method == 'linalg':
        cov_inv = np.linalg.inv(cov)
    elif inv_method == 'lw':
        if cov_method != 'lw':
            raise ValueError(f'inv_mentod = lw requires cov_method = lw as well')
        cov_inv = ledoit_wolf.precision_
    else:
        raise ValueError(f'inv_method = {cov_method} is unknown. Supported are (linalg, lw)')

    return mean, cov_inv


def evaluate_ood(targets, scores):
    fpr, tpr, thres = m.roc_curve(targets, scores)
    auc_roc = m.auc(fpr, tpr)
    ap = m.average_precision_score(targets, scores)
    return {'auc': auc_roc, 'ap': ap, 'fpr': fpr, 'tpr': tpr, 'thres': thres}
