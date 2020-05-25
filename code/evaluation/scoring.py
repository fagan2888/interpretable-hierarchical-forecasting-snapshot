#!/usr/bin/env python
# coding: utf-8

# # Scoring Rules

import numpy as np
import timeit

from scipy.stats import norm


# ## Interval Score


def interval_score(
    observations, alpha, q_dict=None, q_left=None, q_right=None, check_consistency=True
):
    """
    Compute interval scores (aka quantile scores) for an array of observations and predicted intervals.
    
    Either a dictionary with the respective (alpha/2) and (1-(alpha/2)) quantiles via q_dict needs to be
    specified or the quantiles need to be specified via q_left and q_right.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alpha : numeric
        Alpha level for (1-alpha) interval.
    q_dict : dict, optional
        Dictionary with predicted quantiles for all instances in `observations`.
    q_left : array_like, optional
        Predicted (alpha/2)-quantiles for all instances in `observations`.
    q_right : array_like, optional
        Predicted (1-(alpha/2))-quantiles for all instances in `observations`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total interval scores.
    sharpness : array_like
        Sharpness component of interval scores.
    calibration : array_like
        Calibration component of interval scores.
    """

    if q_dict is None:
        if q_left is None or q_right is None:
            raise ValueError(
                "Either quantile dictionary or left and right quantile must be supplied."
            )
    else:
        if q_left is not None or q_right is not None:
            raise ValueError(
                "Either quantile dictionary OR left and right quantile must be supplied, not both."
            )
        q_left = q_dict.get(alpha / 2)
        if q_left is None:
            raise ValueError(f"Quantile dictionary does not include {alpha/2}-quantile")

        q_right = q_dict.get(1 - (alpha / 2))
        if q_right is None:
            raise ValueError(
                f"Quantile dictionary does not include {1-(alpha/2)}-quantile"
            )

    if check_consistency and np.any(q_left > q_right):
        raise ValueError("Left quantile must be smaller than right quantile.")

    sharpness = q_right - q_left
    calibration = (
        (
            np.clip(q_left - observations, a_min=0, a_max=None)
            + np.clip(observations - q_right, a_min=0, a_max=None)
        )
        * 2
        / alpha
    )
    total = sharpness + calibration
    return total, sharpness, calibration


# ## Weighted Interval Score


def weighted_interval_score(
    observations, alphas, q_dict, weights=None, check_consistency=True
):
    """
    Compute weighted interval scores for an array of observations and a number of different predicted intervals.
    
    This function implements the new WIS-score by Johannes Bracher. A dictionary with the respective (alpha/2)
    and (1-(alpha/2)) quantiles for all alpha levels given in `alphas` needs to be specified.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alphas : iterable
        Alpha levels for (1-alpha) intervals.
    q_dict : dict
        Dictionary with predicted quantiles for all instances in `observations`.
    weights : iterable, optional
        Corresponding weights for each interval. If `None`, `weights` is set to `alphas`, yielding the WIS^alpha-score.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total weighted interval scores.
    sharpness : array_like
        Sharpness component of weighted interval scores.
    calibration : array_like
        Calibration component of weighted interval scores.
    """
    if weights is None:
        weights = alphas

    def weigh_scores(tuple_in, weight):
        return tuple_in[0] * weight, tuple_in[1] * weight, tuple_in[2] * weight

    interval_scores = [
        i
        for i in zip(
            *[
                weigh_scores(
                    interval_score(
                        observations,
                        alpha,
                        q_dict=q_dict,
                        check_consistency=check_consistency,
                    ),
                    weight,
                )
                for alpha, weight in zip(alphas, weights)
            ]
        )
    ]

    total = np.sum(np.vstack(interval_scores[0]), axis=0) / sum(weights)
    sharpness = np.sum(np.vstack(interval_scores[1]), axis=0) / sum(weights)
    calibration = np.sum(np.vstack(interval_scores[2]), axis=0) / sum(weights)

    return total, sharpness, calibration


def weighted_interval_score_fast(
    observations, alphas, q_dict, weights=None, check_consistency=True
):
    """
    Compute weighted interval scores for an array of observations and a number of different predicted intervals.
    
    This function implements the new WIS-score by Johannes Bracher. A dictionary with the respective (alpha/2)
    and (1-(alpha/2)) quantiles for all alpha levels given in `alphas` needs to be specified.
    
    This is a more efficient implementation using array operations instead of repeated calls of `interval_score`.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alphas : iterable
        Alpha levels for (1-alpha) intervals.
    q_dict : dict
        Dictionary with predicted quantiles for all instances in `observations`.
    weights : iterable, optional
        Corresponding weights for each interval. If `None`, `weights` is set to `alphas`, yielding the WIS^alpha-score.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total weighted interval scores.
    sharpness : array_like
        Sharpness component of weighted interval scores.
    calibration : array_like
        Calibration component of weighted interval scores.
    """
    if weights is None:
        weights = alphas

    if not all(alphas[i] <= alphas[i + 1] for i in range(len(alphas) - 1)):
        raise ValueError("Alpha values must be sorted in ascending order.")

    reversed_weights = list(reversed(weights))

    lower_quantiles = [q_dict.get(alpha / 2) for alpha in alphas]
    upper_quantiles = [q_dict.get(1 - (alpha / 2)) for alpha in reversed(alphas)]
    if any(q is None for q in lower_quantiles) or any(
        q is None for q in upper_quantiles
    ):
        raise ValueError(
            f"Quantile dictionary does not include all necessary quantiles."
        )

    lower_quantiles = np.vstack(lower_quantiles)
    upper_quantiles = np.vstack(upper_quantiles)

    # Check for consistency
    if check_consistency and np.any(
        np.diff(np.vstack((lower_quantiles, upper_quantiles)), axis=0) < 0
    ):
        raise ValueError("Quantiles are not consistent.")

    lower_q_alphas = (2 / np.array(alphas)).reshape((-1, 1))
    upper_q_alphas = (2 / np.array(list(reversed(alphas)))).reshape((-1, 1))

    # compute score components for all intervals
    sharpnesses = np.flip(upper_quantiles, axis=0) - lower_quantiles

    lower_calibrations = (
        np.clip(lower_quantiles - observations, a_min=0, a_max=None) * lower_q_alphas
    )
    upper_calibrations = (
        np.clip(observations - upper_quantiles, a_min=0, a_max=None) * upper_q_alphas
    )
    calibrations = lower_calibrations + np.flip(upper_calibrations, axis=0)

    totals = sharpnesses + calibrations

    # weigh scores
    weights = np.array(weights).reshape((-1, 1))

    sharpnesses_weighted = sharpnesses * weights
    calibrations_weighted = calibrations * weights
    totals_weighted = totals * weights

    # normalize and aggregate all interval scores
    weights_sum = np.sum(weights)

    sharpnesses_final = np.sum(sharpnesses_weighted, axis=0) / weights_sum
    calibrations_final = np.sum(calibrations_weighted, axis=0) / weights_sum
    totals_final = np.sum(totals_weighted, axis=0) / weights_sum

    return totals_final, sharpnesses_final, calibrations_final


# ## Outside-Interval Count


def outside_interval(observations, lower, upper, check_consistency=True):
    """
    Indicate whether observations are outside a predicted interval for an array of observations and predicted intervals.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    lower : array_like, optional
        Predicted lower interval boundary for all instances in `observations`.
    upper : array_like, optional
        Predicted upper interval boundary for all instances in `observations`.
    check_consistency: bool, optional
        If `True`, interval boundaries are checked for consistency. Default is `True`.
        
    Returns
    -------
    Out : array_like
        Array of zeroes (False) and ones (True) counting the number of times observations where outside the interval.
    """
    if check_consistency and np.any(lower > upper):
        raise ValueError("Lower border must be smaller than upper border.")

    return ((lower > observations) + (upper < observations)).astype(int)


# ## Interval Consistency Score


def interval_consistency_score(
    lower_old, upper_old, lower_new, upper_new, check_consistency=True
):
    """
    Compute interval consistency scores for an old and a new interval.
    
    Adapted variant of the interval score which measures the consistency of updated intervals over time.
    Ideally, updated predicted intervals would always be within the previous estimates of the interval, yielding
    a score of zero (best).
    
    Parameters
    ----------
    lower_old : array_like
        Previous lower interval boundary for all instances in `observations`.
    upper_old : array_like, optional
        Previous upper interval boundary for all instances in `observations`.
    lower_new : array_like
        New lower interval boundary for all instances in `observations`. Ideally higher than the previous boundary.
    upper_new : array_like, optional
        New upper interval boundary for all instances in `observations`. Ideally lower than the previous boundary.
    check_consistency: bool, optional
        If interval boundaries are checked for consistency. Default is `True`.
        
    Returns
    -------
    scores : array_like
        Interval consistency scores.
    """
    if check_consistency and (
        np.any(lower_old > upper_old) or np.any(lower_new > upper_new)
    ):
        raise ValueError("Left quantile must be smaller than right quantile.")

    scores = np.clip(lower_old - lower_new, a_min=0, a_max=None) + np.clip(
        upper_new - upper_old, a_min=0, a_max=None
    )
    return scores


# ## MAPE and sMAPE


def mape_score(observations, point_forecasts):
    return 100 * np.abs(point_forecasts - observations) / np.abs(observations)


def smape_score(observations, point_forecasts):
    return 100 * (
        2
        * np.abs(point_forecasts - observations)
        / (np.abs(observations) + np.abs(point_forecasts))
    )


# ## Parametric Wrappers


def confidence_to_predictive(
    point_forecasts, lower_ci, upper_ci, alpha, quantiles, dist="norm"
):
    if dist == "norm":
        norm_sd_lower = (point_forecasts - lower_ci) / norm.ppf(1 - alpha / 2)
        norm_sd_upper = (upper_ci - point_forecasts) / norm.ppf(1 - alpha / 2)
        norm_sd = (norm_sd_lower + norm_sd_upper) / 2  # average

        return np.array(
            [
                [norm.ppf(q, loc=point, scale=stdev) for q in quantiles]
                for point, stdev in zip(point_forecasts, norm_sd)
            ]
        )
    else:
        raise ValueError(f"Distribution {norm} not implemented.")


# ## Tests

if __name__ == "__main__":
    observations_test = np.array([4, 7, 4, 6, 2, 1, 3, 8])
    q_10_test = np.array([2, 3, 5, 9, 1, -3, 0.2, 8.7])
    q_90_test = np.array([5, 5, 7, 13, 5, -1, 3, 9])
    alpha_test = 0.2
    quantile_dict_test = {
        0.1: np.array([2, 3, 5, 9, 1, -3, 0.2, 8.7]),
        0.2: np.array([2, 4.6, 5, 9.4, 1.4, -2, 0.4, 8.8]),
        0.8: np.array([4, 4.8, 5.7, 12, 4.3, -1.5, 2, 8.9]),
        0.9: np.array([5, 5, 7, 13, 5, -1, 3, 9]),
    }


# ### Interval Score

if __name__ == "__main__":
    print(
        interval_score(
            observations_test, alpha_test, q_left=q_10_test, q_right=q_90_test
        )
    )
    print(interval_score(observations_test, alpha_test, q_dict=quantile_dict_test))


# ### Weighted Interval Score

if __name__ == "__main__":
    print(
        weighted_interval_score(
            observations_test,
            alphas=[0.2, 0.4],
            weights=[2, 5],
            q_dict=quantile_dict_test,
        )
    )

    print(
        weighted_interval_score_fast(
            observations_test,
            alphas=[0.2, 0.4],
            weights=[2, 5],
            q_dict=quantile_dict_test,
        )
    )

    # WIS^alpha score
    print(
        weighted_interval_score_fast(
            observations_test,
            alphas=[0.2, 0.4],
            weights=None,
            q_dict=quantile_dict_test,
        )
    )


# #### Compare runtimes of weighted interval score methods

if __name__ == "__main__":
    print(
        {
            n: min(
                timeit.repeat(
                    lambda: weighted_interval_score(
                        observations_test,
                        alphas=[0.2] * n,
                        weights=None,
                        q_dict=quantile_dict_test,
                    ),
                    repeat=100,
                    number=100,
                )
            )
            for n in [2, 5, 10, 20]
        }
    )


if __name__ == "__main__":
    print(
        {
            n: min(
                timeit.repeat(
                    lambda: weighted_interval_score_fast(
                        observations_test,
                        alphas=[0.2] * 2,
                        weights=None,
                        q_dict=quantile_dict_test,
                    ),
                    repeat=100,
                    number=100,
                )
            )
            for n in [2, 5, 10, 20, 40, 80]
        }
    )


# As can be seen, the fast implementation is in fact quicker for more than 2 or 3 intervals. It stays almost constant.

# ### Outside Interval Count

if __name__ == "__main__":
    print(outside_interval(observations_test, lower=q_10_test, upper=q_90_test))


# ### Interval Consistency Score

if __name__ == "__main__":
    interval_consistency_score(
        quantile_dict_test[0.1],
        quantile_dict_test[0.9],
        quantile_dict_test[0.1],
        quantile_dict_test[0.8],
    )


if __name__ == "__main__":
    interval_consistency_score(
        quantile_dict_test[0.1],
        quantile_dict_test[0.8],
        quantile_dict_test[0.1],
        quantile_dict_test[0.9],
    )


# ## Parametric Wrappers

if __name__ == "__main__":
    points = np.array([309, 327, 452, 496, 764, 994])
    lower = np.array([293, 309, 433, 475, 741, 968])
    upper = np.array([328, 343, 474, 523, 798, 1029])
    confidence_to_predictive(
        points, lower, upper, alpha=0.025, quantiles=np.arange(0.05, 1, 0.05)
    )
