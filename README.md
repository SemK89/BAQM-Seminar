# BAQM-Seminar
Code and scripts for the Econometrics BAQM Seminar for a.s.r. / Aegon

The file preprocessing.py contains all preprocessing utility functions that can be used across separate analyses.

The used dataset is proprietary.


The data set is cleaned and manipulated in this case. Most functions are included in the preprocessing file. Survival models' data are preprocessed in the same way as those included in the logit_model file; the imputation of total premium is also included. In the logit_model.py, the discrete-time survival model is introduced, and the AUC and Brier Score (including the one for RSF) are calculated. 
