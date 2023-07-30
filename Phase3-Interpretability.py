#Phase 3: Different interpretability techniques used to calculate feature importance

#Program to calculate feature importance using XGBoost(Extreme Gradient Boosting)
%reset

#import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
shap.initjs()

#import data and define X and y values
Data_try=pd.read_csv("C:/files/micro_rna-200.csv")
X = Data_try.iloc[1:,1:191].values
yval = Data_try.iloc[1:,191]

#Uses XGBoost classifier to fit a model to the data and then visualizes the feature importances using different graphs.
xgc = xgb.XGBClassifier(n_estimators=10, max_depth=5, base_score=0.5,
                        objective='binary:logistic')
xgc.fit(X, yval)
fig = plt.figure(figsize = (16, 12))
title = fig.suptitle("Default Feature Importances from XGBoost", fontsize=14)
ax1 = fig.add_subplot(2,2, 1)
xgb.plot_importance(xgc, importance_type='weight', ax=ax1,max_num_features=20)
t=ax1.set_title("Feature Importance - Feature Weight")
ax2 = fig.add_subplot(2,2, 2)
xgb.plot_importance(xgc, importance_type='gain', ax=ax2,max_num_features=20)
t=ax2.set_title("Feature Importance - Split Mean Gain")
ax3 = fig.add_subplot(2,2, 3)
xgb.plot_importance(xgc, importance_type='cover', ax=ax3,max_num_features=20)
t=ax3.set_title("Feature Importance - Sample Coverage")

#################################
#Program to calculate feature importance using SHAP (SHapley Additive exPlanations)
%reset

#import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')
shap.initjs()

#import data and define X and y values
Data_try=pd.read_csv("C:/files/micro_rna-200.csv")
X = Data_try.iloc[1:,1:191].values
yval = Data_try.iloc[1:,191]

#Makes predictions on data using XGBoost algorithm and creates a custom function called booster
xgc = xgb.XGBClassifier(n_estimators=10, max_depth=5, base_score=0.5,
                        objective='binary:logistic')
mymodel = xgc.fit(X, yval)
mybooster = mymodel.get_booster()
model_bytearray = mybooster.save_raw()[4:]
def myfun(self=None):
    return model_bytearray
mybooster.save_raw = myfun

#Uses SHAP to explain the predictions of the XGBoost model by computing and plotting Shapley values
shap_values = shap.TreeExplainer(mybooster).shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar")

####################################################
# Program to calculate feature importance using Skater algorithm
%reset

#import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
import xgboost as xgb
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
import warnings
warnings.filterwarnings('ignore')
shap.initjs()

#import data and define X and y values
Data_try=pd.read_csv("C:/files/micro_rna-200.csv")
X = Data_try.iloc[1:,1:191].astype(float)
yval = Data_try.iloc[1:,191].astype(int)

#Makes predictions on data using XGBoost algorithm and makes a model called the im_model using the InMemoryModel class
xgc = xgb.XGBClassifier(n_estimators=10, max_depth=5, base_score=0.5,
                        objective='binary:logistic')
xgc.fit(X, yval)
im_model = InMemoryModel(xgc.predict_proba, examples=X)

#Interpretation class created and data X is loaded into the interpreter
interpreter = Interpretation()
interpreter.load_data(X)

#Feature importances are computed using the interpreter for the im_model and the 20 features with the highest importance are printed
importances=interpreter.feature_importance.feature_importance(im_model, ascending=True)
print(importances[170:190])
