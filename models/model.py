# importing python libraries
import pandas as pd
import pickle as pkl
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

# loading diabetes data into variable data
data = pd.read_csv("./dataset/diabetes.csv")

# wrangling dataset.
data.chol_hdl_ratio = round(data.cholesterol / data.hdl_chol, 2)
data.waist_hip_ratio = round(data.waist / data.hip, 2)

# correcting comma separated number to decimal separated number.
data.bmi = pd.to_numeric(data.bmi.str.replace(",", "."))

print(data.head())
# encoding columns with object values using Ordinal Encoding
s = (data.dtypes == "object")
obj_col = s[s].index

print("Ordinal Encoding")
orde = OrdinalEncoder()
data[obj_col] = orde.fit_transform(data[obj_col])

print("Splitting features and target.")
# dropping off target and unnecessary columns (diabetes and patient number columns)
X = data.drop(["patient_number", "diabetes"], axis=1)
y = data.diabetes

print("Robust Scaling on X, y.")
# scaling data using RobustScaler
scale = RobustScaler()
scaled_X = scale.fit_transform(X, y)

print("Stratified Split.")
# StratifiedShuffleSplit on Data
split = StratifiedShuffleSplit(n_splits=4, random_state=42)

for train_index, test_index in split.split(scaled_X, y):
    X_train, X_test = scaled_X[train_index], scaled_X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Loading LightGBM classifier to be used for training model
lgbm = LGBMClassifier(n_estimators=200, max_depth=-2, random_state=42)
lgbm.fit(X_train, y_train)
pred = lgbm.predict(X_test)

f1 = f1_score(pred, y_test)
print(f"F1 Score for LightGBM: {f1}.")

# Using pickle to save model
lightgbm = open("../deployment/lightgbm.pickle", "wb")
pkl.dump(lgbm, lightgbm)
lightgbm.close()
