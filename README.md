# Interpretable Hierarchical Forecasting of Infectious Diseases
The main objective of this work is to develop an approach for *interpretable hierarchical forecasting*, which allows to combine forecast reconciliation methods (e.g. optimal combination via OLS) with feature attribution methods (e.g. SHapley Additive Explanations) in order to obtain coherent model explanations for hierarchical forecasts. The practicality of the concept is demonstrated by producing interpretable and hierarchical forecasts for case numbers of infectious diseases. For a detailed motivation and background, see the [thesis exposé](./thesis/exposé.md).

Hierarchical forecasting is a well-established discipline and interpretable machine learning is an emerging research stream. However, the combination of both, as done in this work, is a novelty. The following conceptual overview puts the different steps and components of interpretable hierarchical forecasting in relation to each other:

<img src="./resources/Process%20Overview.svg" height="100%"/>

----
## Technical Report for May
### Summary of Theory
The existing body of knowledge in hierarchical forecasting (forecast reconciliation) and in interpretable machine learning (feature attribution) has been assessed in a structured and reproducible literature research (for details on the approach see the [research documentation](./literature/research%20process)).

An approach has been developed to transfer forecast reconciliation to local model explanations in order to obtain coherent hierarchical explanations. A next step is to formally prove the hypothesis that desired properties of feature attribution methods (see [section on interpretation](#interpretation)) can be preserved under the required transformations.

### Summary of Practice
As a first case study, the nowcasting\* of Covid-19 infections in Germany has been chosen. Preprocessing of individual case data (especially missing value imputation), time series forecasting using quantile regression and cross-validation of probabilistic forecasts using proper scoring rules has been implemented.

Next steps are the implementation of forecast reconciliation methods and the adaption of model explainers (like SHAP or LIME) to hierarchical forecasts, as well as the implementation of further quantile regressors. Later, more case studies will be conducted with other infectious diseases for which more training data is available.

\**The goal of nowcasting is to estimate the true number of current new infections by adjusting for the reporting delay inherent in today's public health reporting systems. It is usually performed by estimation of the reporting delay distribution using a reverse-time survival function (for a good introduction, see ([Höhle, an der Heiden, 2014](https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.12194))).*

### Detailed Updates
#### Preprocessing
Nowcasting should be performed for the time of *disease onset*, not the time of *report*. However, in almost 40% of reported Covid-19 cases, the disease onset date is missing.
Several model-free approaches for *multiple imputation* of onset dates (i.e. each missing value is imputed by drawing from a predictive distribution several times) have been implemented (see [this code](./notebook/imputation/imputation_methods.ipynb)) and compared using k-fold cross-validation. The Kullback-Leibler divergence was used as a performance criterion to compare the true and estimated distribution of reporting delay. Predictive mean matching with random forests produced the best imputations (see [this code](./notebook/imputation/Cross-validation%20of%20imputation%20methods.ipynb)). The missing-at-random-assumption for imputation has been tested by showing that missingness itself cannot be predicted based on other features (see [this code](./notebook/imputation/Missing-At-Random%20Analysis.ipynb)).

#### Forecasting
*Quantile Regression*<br>
Infectious disease forecasting is associated with considerable uncertainty. Thus, instead of point forecasts, probabilistic forecasts are required. This work will focus on quantile regression because it does not rely on a-priori assumptions about the predictive distribution. Also, one can interpret the predictions for each quantile separately to assess the varying impact of features on different regions of the distribution.

The following methods for quantile regression shall be explored:
- Linear quantile regression
- Quantile regression trees
- Quantile regression with gradient boosting trees
- Quantile regression with RNNs using pinball loss / check loss

So far, linear quantile regression and quantile regression with gradient boosters (LightGBM, CatBoost and sklearnGradientBoosting) have been implemented using wrappers which jointly estimate several quantiles in one routine. For the gradient boosters, an additional cascading scheme was implemented: The model for a higher quantile uses the predicted value for the next lower quantile as input feature in order to improve the coherency of the quantiles (see [this code](./notebook/quantile%20regression)).

*Time series normalization and cross-learning*<br>
Time series of infectious diseases often exhibit seasonality and strong trends (e.g. during epidemic phases). In order to counteract non-stationarity and to allow for cross-learning on several time series, the potential of time series normalization has been tested. Based on the findings from the [M4 forecasting competition](https://www.sciencedirect.com/science/article/abs/pii/S0169207018300785), exponential smoothing has been tested as a promising normalization technique.
First test results indicate that normalization can help to overcome the deficiency of tree-based learners to extrapolate the exponential epidemic curve (see [nowcasting notebook](./notebook/nowcasting/Nowcasting.ipynb)). However, normalization also introduces problems with interpretability and the sharpness of prediction intervals which still must be addressed.

*Nowcasting as a time series prediction* task<br>
In order to tackle nowcasting using standard regression tools, it was reformulated as a time series forecasting task with exogenous information (see [nowcasting notebook](./notebook/nowcasting/Nowcasting.ipynb)). Roughly speaking, nowcasting the case numbers for today is equivalent to forecasting the case numbers 30 days ago with a forecast horizon of 30 days and additional exogenous information about the future.

#### Interpretation
Interpretable machine learning is still an emerging field without a consolidated set of theories/terminologies and with few peer-reviewed publications. Hence, for the literature research on feature attribution methods, conference proceedings and preprints were included and forward-search of influential authors was conducted (see [literature research](./literature/research%20process/Feature%20Attribution)).

Main insight: Several desirable, axiomatic properties for feature attributions have been proposed, for example local accuracy/completeness, symmetry, consistency or linearity. Shapley values seem to be the most theoretically grounded approach to feature attribution, although there remains debate on their operationalization (e.g. choice of reference distributions). Nonetheless, SHapley Additive Explanations will be used as a sound and accurate method for which efficient model-specific explainers exist. The existing tool will have to be adjusted so that it operates on sets of quantile forecasts too. Other feature attribution methods such as integrated gradients or LIME may be tried out as well.

#### Reconciliation
A literature research across several databases on peer-reviewed journals has been conducted to review the body of knowledge on hierarchical forecasting and different methods have been identified to reconcile incoherent forecasts through matrix projections to a coherent subspace (see [literature research](./literature/research%20process/Hierarchical%20Forecasting)).
A remaining step is to implement forecast reconciliation as a standalone functionality in python (suitable tools readily exists in R, but the experiments will be mostly conducted in python).

#### Hierarchical Explanation
The methodological novelty of this work is to apply forecast reconciliation to feature attributions. The general approach is as follows: First, incoherent predictions are produced by forecasting each time series of a hierarchy individually without consideration of the underlying aggregation constraints. For these predictions, local explanations are obtained using a feature attribution method. Next, the incoherent predictions are made coherent using a reconciliation method of choice. Finally, the feature attributions for the incoherent models must be transformed analogously in order to obtain explanations for the coherent forecasts. The transformation will consist of two steps: (1) the matrix projection used by the reconciliation step has to be applied to the feature attributions as well and (2) the features of higher level forecasts have to be disaggregated to the features of lower level forecasts in order to make them comparable.

If easily interpretable models are used for the incoherent forecasts, this method will be considerably more efficient than a naive end-of-pipe feature attribution of the reconciled forecasts (the latter would need to apply a model-agnostic attribution method to the full ensemble of hierarchical forecasts).

It will be shown formally that if desirable properties of feature attribution are fulfilled by the explanations for the incoherent forecasts, they can be preserved under the projections of forecast reconciliation. Implementation-wise, existing model explainers for regression (e.g. SHAP or LIME) have to be extended by the proposed functionality.

#### Cross-Validation
For point forecasts, well-known measures such as MAE, MAPE and MASE are used. For probabilistic forecasts, after a review of the literature on proper scoring rules, interval-based scoring rules were identified as most suitable because they are compatible with all types of probabilistic predictions (parametric, samples, quantiles) and have desirable scoring properties. No software for interval scoring is currently available for quantile forecasts, therefore the scoring rules had to be implemented from scratch (see [this code](./notebook/evaluation/scoring.ipynb)).

To evaluate the generalization error of time series models, a time-aware cross-validation scheme should be used (ignoring the time-dependency of the data generating process violates the i.i.d-assumption of cross-validation and will lead to biased scores). Hence, a validation scheme form scikit-learn has been adapted to perform walk-forward cross-validation (see [nowcasting notebook](./notebook/nowcasting/Nowcasting.ipynb)).

At a later stage, the cross-validation and performance scores will have to be transferred to the hierarchical setting, which is mostly a challenge of efficient implementation.
