# Exposé: Interpretable Hierarchical Forecasting of Infectious Diseases

## Motivation
Forecasts for infectious diseases can provide valuable decision support in public health, especially during epidemics (Nsoesie et al., 2013; Lutz et al., 2019). Decision makers often require the predicted case numbers of a disease to be partitioned by region and demographic features such as age or sex, as well as coherently aggregated at various levels, e.g. county, state and nation (Biggerstaff et al., 2016, 2018; Lutz et al., 2019). This constitutes a highly relevant and illustrative application of hierarchical forecasting, where the task is to predict a set of time series with hierarchical aggregation structure across one or several dimensions, while the aggregated time series are a linear combination of the disaggregated series (Hyndman and Athanasopoulos, 2018).

The requirements of hierarchical forecasting can be met by making independent predictions at the smallest resolution and aggregating results for higher levels (bottom-up approach), which however does not take the correlation structure between the time series into account and thus can lead to inaccurate forecasts at higher levels (Hyndman et al., 2011). An alternative approach is to produce forecasts for all aggregation levels separately, which will likely yield incoherent results, as the hierarchical aggregation constraints are ignored by the individual forecasts. Fortunately, forecasts at different levels can be harmonized ex post through appropriate transformations, a method known as forecast reconciliation (Hyndman et al., 2011; Taieb, Taylor and Hyndman, 2017; Wickramasuriya, Athanasopoulos and Hyndman, 2019). In infectious diseases forecasting, such a technique may even improve overall forecast accuracy (Gibson et al., 2019).

Meanwhile, researchers have called for more interpretability of data-driven predictions in infectious disease surveillance, especially as complex statistical methods such as machine learning are employed (Flahault and Paragios, 2016; Moss et al., 2018; Lutz et al., 2019). It has been suggested that the lack of interpretability hampers the widespread use of machine learning among epidemiologists (Yuan et al., 2019). Black box models contradict the basic principle of human judgement and comprehensibility in health sciences, where decision making can have a significant impact on individuals or populations and therefore requires explanations to those affected (Flahault and Paragios, 2016). On these grounds, interpretability also concerns infectious disease forecasting as carried out in public health surveillance.

Interpretability in data-driven forecasting can either be ensured through the use of statistical methods with limited complexity (e.g. linear models) or by providing simplified explanations for predictions from complex models via methods from the recently emerging field of interpretable machine learning (Berrada and Adadi, 2018; Mohseni, Zarei and Ragan, 2018; Molnar, 2019). A common objective of most explanations is to relate the inputs of a prediction process (i.e. features of a statistical model) to its outcome, this concept is known as feature attribution (Lundberg and Lee, 2017; Molnar, 2019).
Unfortunately, feature attribution is not straightforward when hierarchical forecasts are obtained through forecast reconciliation. Even if simple models are used for the individual forecasts, the ex-post transformations carried out to ensure consistency between hierarchy levels pose an additional layer of obfuscation between input data, models and predictions. Moreover, forecast reconciliation does not explicitly reveal the sources of incoherence between predictions in the first place, although such insights could help to improve forecasting models. On the other hand, if feature attribution could be applied to reconciled forecasts, it would provide model explanations for predictions at all levels, making hierarchical forecasting interpretable and thus more suitable for infectious disease forecasting.

## Formulation of Objectives
The main objective of this work is to develop an approach for interpretable hierarchical forecasting which allows to combine forecast reconciliation methods with feature attribution in order to obtain model explanations for hierarchical forecasts. The practicality of the concept will be demonstrated by producing interpretable and hierarchical forecasts for case numbers of infectious diseases.

## State of Research / Practice
Methodologically, hierarchical forecasting is a long-known task with a wide range of applications in public health, econometrics and business (Fliedner, 2015). Early research mostly focused on the bottom-up or top-down construction of forecasts (or mixtures of both) and the performance of these methods in different forecasting settings (Schwarzkopf et al., 1988; Gross and Sohl, 1990; Shlifer and Wolff, 2017). Later, forecast reconciliation was proposed as a more general approach, which is based on matrix projections of incoherent forecasts to a coherent subspace (Hyndman, R.J. Athanasopoulos, 2018). Under this framework, further theoretical development yielded a number of reconciliation approaches which aim to minimize the overall forecast error and provide unbiased predictions (given the assumption that individual predictions are unbiased) (Hyndman et al., 2011; Wickramasuriya, Athanasopoulos and Hyndman, 2019). Recently, forecast reconciliation has been extended from the original focus on point forecasts to probabilistic forecasts (Taieb, Taylor and Hyndman, 2017; Gamakumara et al., 2019; Gibson et al., 2019).

Forecasting of infectious diseases has been mostly applied to well-known diseases such as influenza, where substantial historical data is available (Lutz et al., 2019). Aside from mechanistic approaches which predict the spread of epidemics through explicit modeling, statistical approaches such as time-series modeling, Bayesian regression, generalized linear models and machine learning have been employed to predict the case counts of infectious diseases (Unkel et al., 2012; Nsoesie et al., 2013; Biggerstaff et al., 2016, 2018; Brooks et al., 2018; Stojanović et al., 2019). Forecast reconciliation has only sparsely been applied to epidemic forecasts but seems to have the potential of improving the accuracy of predictions (Shang and Smith, 2013; Shang and Hyndman, 2017; Biggerstaff et al., 2018).

Interpretable machine learning (IML), being a very recent research stream, has not yet built a consolidated set of theories on interpretability in machine learning (Berrada and Adadi, 2018; Mohseni, Zarei and Ragan, 2018). Nevertheless, the concept of local model explanation through feature attribution, where the result of a prediction is represented as a sum of contributions of the different input features (Lundberg and Lee, 2017), has received considerable attention. Explanations may be straightforwardly obtained from translucent models (e.g. linear models), where feature attribution is directly given through the model coefficients and inputs, but there exist also non-interpretable machine learning models (e.g. tree-based ensembles), where sophisticated techniques have been proposed to approximate feature attributions locally (Lundberg et al., 2019; Molnar, 2019).

To date, no research on the interpretability of forecast reconciliation and explanation of hierarchical forecasts is known.

## Research Approach
### Review
To motivate the research of this work, an application-focused, representative literature review on the forecasting of infectious diseases is conducted which addresses the hierarchical nature of disease forecasts and provides an overview over data-driven forecasting methods and evaluation approaches. For a theoretical basis, two further literature reviews will be conducted to summarize the current body of knowledge, a representative review of hierarchical forecasting and a central review on feature attribution, each with a focus on theories (c.f. COOPER for classification).
The full review process will be structured according to VOM BROCKE ET AL., with reproducible literature search following the guidelines by LEVY AND ELLIS and a concept-centric synthesis and organization as proposed by WEBSTER AND WATSON.

### Theory
Based on the findings from the theoretical review, the concept of model explanation through feature attribution shall be transferred to hierarchical forecasting:
First, a general framework for interpretable hierarchical forecasting will be proposed, which bridges feature attribution and hierarchical forecasting by providing a unified and compatible notation. Second, it will be shown formally that under certain preconditions, the transformations from forecast reconciliation cannot only be applied to the individual forecasts but also to their corresponding feature attributions without introducing bias to the explanations. This offers an efficient way of obtaining feature attributions for the reconciled forecasts.

### Experiments
In order to demonstrate the practicality of the approach, a case study in infectious disease forecasting is conducted:
First, based on the findings of the application-focused literature review on infectious disease forecasting, a suitable experimental setting is chosen by selecting one or several disease time series, one or several statistical approaches for infectious disease forecasting, several approaches to forecast reconciliation, several performance measures and several feature attribution methods.
Next, hierarchical forecasting of diseases is conducted, which involves preprocessing of disease data, modeling of forecasts and reconciliation, model fitting and finally performance evaluation via suitable cross-validation. Next, in order to obtain model explanations, feature attribution methods are applied to the hierarchical forecasts using the newly proposed framework for interpretable hierarchical forecasting.

### Evaluation
Given the obtained model explanations, a case-based evaluation of the usefulness of feature attributions for practice will be conducted by describing possible insights about the forecasting models, assessing the consistency of the explanations and addressing their general plausibility for describing infectious disease spreading.

## Design of the Thesis
### Introduction (topic, motivation, objectives, outline)

### Part I: Literature Review

(1) Materials and Methods (literature research methodology)

(2) Forecasting of Infectious Diseases (application-focused review)

(3) Hierarchical Forecasting (theoretical review)

(4) Interpretability and Feature Attribution (theoretical review)

Part II: Interpretable Hierarchical Forecasting

(5) A Framework for Interpretable Hierarchical Forecasting

(6) Feature Attribution Approaches for Reconciled Forecasts

### Part III: Application of Interpretable Hierarchical Forecasting to Infectious Diseases

(7) Methods (forecast, reconciliation, performance evaluation, feature attribution)

(8) Experiments (modeling, training, validation, feature attribution)

(9) Results (performance evaluation, model interpretation)

### Discussion

### Conclusion and Outlook

## Literature
Berrada, M. and Adadi, A. (2018) ‘Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)’, IEEE Access. IEEE, 6, pp. 52138–52160. doi: 10.1109/ACCESS.2018.2870052.

Biggerstaff, M. et al. (2016) ‘Results from the centers for disease control and prevention’s predict the 2013 – 2014 Influenza Season Challenge’, BMC Infectious Diseases. BMC Infectious Diseases, 16(357), pp. 1–10. doi: 10.1186/s12879-016-1669-x.

Biggerstaff, M. et al. (2018) ‘Results from the second year of a collaborative effort to forecast influenza seasons in the United States’, Epidemic, 24(February), pp. 26–33. doi: 10.1016/j.epidem.2018.02.003.
Vom Brocke, J. et al. (2009) ‘Reconstructing the Giant: On the Importance of Rigour in Documenting the Literature Search Process’, ECIS 2009 Proceedings.

Brooks, L. C. et al. (2018) ‘Nonmechanistic forecasts of seasonal influenza with iterative one-week-ahead distributions’, PLoS Comput Biol, 14(6), pp. 1–30.

Cooper, H. M. (1986) ‘Organizing Knowledge Syntheses : A Taxonomy of Literature Reviews’.

Flahault, A. and Paragios, N. (2016) ‘Public Health and Epidemiology Informatics’, IMIA Yearbook of Medical Informatics 2016, pp. 240–246.

Fliedner, G. (2015) ‘Hierarchical forecasting: issues and use guidelines’, Industrial Management & Data Systems, 101(1), pp. 5–12.

Gamakumara, P. et al. (2019) ‘Probabilisitic Forecasts in Hierarchical Time Series’, Monash University, Work Paper XX/19., (August).

Gibson, G. C. et al. (2019) ‘Improving Probabilistic Infectious Disease Forecasting Through Coherence’, Centers for Disease Control, pp. 1–21.

Gross, C. W. and Sohl, J. E. (1990) ‘Disaggregation methods to expedite product line forecasting’, Journal of Forecasting, 9(3), pp. 233–254. doi: 10.1002/for.3980090304.

Hyndman, R.J. Athanasopoulos, G. (2018) Forecasting: Principles and Practice. 2nd Editio. Melbourne. Available at: OTexts.com/fpp2.

Hyndman, R. J. et al. (2011) ‘Optimal combination forecasts for hierarchical time series’, Computational Statistics and Data Analysis. Elsevier B.V., 55(9), pp. 2579–2589. doi: 10.1016/j.csda.2011.03.006.

Hyndman, R. J. and Athanasopoulos, G. (2018) Forecasting: Principles and practice. 2nd Editio. OTexts.

Levy, Y. and Ellis, T. J. (2006) ‘A Systems Approach to Conduct an Effective Literature Review in Support of Information Systems Research’, Informing Science Journal.

Lundberg, S. M. et al. (2019) ‘Explainable AI for Trees: From Local Explanations to Global Understanding’, Nature Machine Intelligence, 2, pp. 56–67. Available at: http://arxiv.org/abs/1905.04610.

Lundberg, S. M. and Lee, S. I. (2017) ‘A unified approach to interpreting model predictions’, Advances in Neural Information Processing Systems, 2017-Decem(Section 2), pp. 4766–4775.

Lutz, C. S. et al. (2019) ‘Applying infectious disease forecasting to public health : a path forward using influenza forecasting examples’, BMC Public health. BMC Public Health, 19(1659), pp. 1–12.

Mohseni, S., Zarei, N. and Ragan, E. D. (2018) ‘A Multidisciplinary Survey and Framework for Design and Evaluation of Explainable AI Systems’, ACM Trans. interact. intell. syst., 1(1), pp. 1–37. Available at: http://arxiv.org/abs/1811.11839.

Molnar, C. (2019) Interpretable Machine Learning. Lulu.com. Available at: https://books.google.de/books?id=jBm3DwAAQBAJ.

Moss, R. et al. (2018) ‘Epidemic forecasts as a tool for public health: interpretation and (re)calibration’, Infectious and communicable diseases, 42(1), pp. 69–76. doi: 10.1111/1753-6405.12750.

Nsoesie, E. O. et al. (2013) ‘A systematic review of studies on forecasting the dynamics of influenza
outbreaks’, Influenza and other respiratory viruses, 8(8), pp. 309–317. doi: 10.1111/irv.12226.

Schwarzkopf, A. B. et al. (1988) ‘Top-down versus bottom-up forecasting strategies’, International Journal of Production Research, 26(11), pp. 1833–184. doi: 10.1080/00207548808947995.

Shang, H. L. and Hyndman, R. J. (2017) ‘Grouped Functional Time Series Forecasting : An Application to Age-Specific Mortality Rates Grouped Functional Time Series Forecasting : An Application to Age-Specific’, Journal of Computational and Graphical Statistics. Taylor & Francis, 26(2), pp. 330–343. doi: 10.1080/10618600.2016.1237877.

Shang, H. L. and Smith, P. W. F. (2013) Grouped time-series forecasting with an application to regional infant mortality counts, Centre for Population Change.

Shlifer, E. and Wolff, R. W. (2017) ‘Aggregation and Proration in Forecasting’, Management Science, 25(6), pp. 594–603.

Stojanović, O. et al. (2019) ‘A Bayesian Monte Carlo approach for predicting the spread of infectious diseases’, PLoS ONE, 14(12), pp. 1–20. doi: 10.1371/journal.pone.0225838.

Taieb, S. Ben, Taylor, J. W. and Hyndman, R. J. (2017) ‘Coherent probabilistic forecasts for hierarchical time series’, 34th International Conference on Machine Learning, ICML 2017, 7, pp. 5143–5155.

Unkel, S. et al. (2012) ‘Statistical methods for the prospective detection of infectious disease outbreaks: a review’, Journal of the Royal Statistical Society A, 175(1), pp. 49–82.

Webster, J. G. and Watson, R. T. (2002) ‘Analyzing the Past to Prepare for the Future: Writing a Literature Review’, MIS Quarterly, 26(2), pp. 13–23.

Wickramasuriya, S. L., Athanasopoulos, G. and Hyndman, R. J. (2019) ‘Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization’, Journal of the American Statistical Association. Taylor & Francis, 114(526), pp. 804–819. doi: 10.1080/01621459.2018.1448825.

Yuan, M. et al. (2019) ‘A systematic review of aberration detection algorithms used in public health surveillance’, Journal of Biomedical Informatics. Elsevier Inc., 94, p. 103181. doi: 10.1016/j.jbi.2019.103181.