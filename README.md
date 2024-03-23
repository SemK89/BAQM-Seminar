# BAQM-Seminar
Code, paper and presentation for the Master in Econometrics and Management Science: BAQM Seminar for a.s.r. / Aegon. The used dataset is proprietary.

The file preprocessing.py contains all preprocessing utility functions that are used across separate analyses.

The file Clustering.py contains all considered clustering configurations and evaluation metrics. It generates the file with Aegon clusters and relevant plots.

The file Bayesian-causal.py contains the Hierarchical Bayesian Model. It generates files with parameter draws, predicted outcomes and evaluation scores, as well as relevant plots.

The file logit_model.py introduces the discrete-time survival model, which uses sm.logit regression. The AUC and Brier Score (function Brier_and_AUC) are calculated. 

The file Random survival forest.py runs the ML survival model on our data and also calulates the AUC and Brier score.

The file plot.py generates plots using the results from the survival models and the Bayesian model.
Both survival models' data are preprocessed in the same way as those included in the logit_model file; the imputation of total premium is also included.
