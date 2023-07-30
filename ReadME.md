miRNAs have shown to be significant in the development of cancer tumors. Currently,
pancreatic cancer’s (PC’s) early diagnostic rate is just 9% as screening methods are unattainable,
making it the fourth leading cause of cancer death. Many studies have achieved low accuracy
(70-75%) as they use methods that do not take into account the 33% misdiagnosis rate of PC
with other cancers. As a result, feature selection, ensemble algorithms, and interpretability
techniques were used to find significant miRNAs to construct an early diagnostic tool for PC. In
the first phase, recursive feature elimination algorithms were used to find 200 differentially
expressed miRNAs in PC and no PC samples as well as early and late stage PC samples. In the
second phase, an ensemble algorithm was constructed out of K-Nearest Neighbor, Naive Bayes,
neural network, and Logistic Regression models to diagnose PC and distinguish between early
and late stages. In the third phase, XGBoost, SHAP, and Skater interpretability methods were
used to find which miRNAs were significant in model predictions. In the fourth phase, a user
interface, PanCan Diagnosis, was designed to test if a person had no, early, or late stage PC and
also displayed the patient’s most differentially expressed miRNAs. This novel tool is the first in
literature to receive a PC diagnostic accuracy of above 90%, seek miRNAs that can lead to
personalized treatment of early and late stage PC samples, offers a ten-fold improvement in
monetary costs, and is two times faster than current methods.