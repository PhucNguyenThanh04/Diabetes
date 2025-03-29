import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

data = pd.read_csv("synthetic_diabetes_dataset.csv")
result = data.describe()
# profile = ProfileReport(data, title = "Diabetes_Report", explorative=True)
# profile.to_file("report_html")

#split
target = "Diabetes"
x = data.drop(target, axis = 1)
y = data[target]
bool_cols = ["Smoking", "Alcohol_Consumption", "Family_History", "Hypertension_History", "Gestational_Diabetes_History"]
num_cols = ["Age", "BMI", "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", "Cholesterol", "Fasting_Blood_Sugar", "HbA1c", "C_Peptide", "Insulin_Level", "Physical_Activity", "Diet_Score"]
nom_cols = ["Gender"]
x[bool_cols] = x[bool_cols].replace(["N/A", "nan", "p"], np.nan)
x[bool_cols] = x[bool_cols].replace({"Yes": True, "No": False}).astype("boolean")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#preprocession numerical
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(missing_values= np.nan, strategy="median")),
    ("scaler", StandardScaler())
])

bool_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=False)),
    ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False))
])


nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output= False))
])

#data processing
preprocessor =ColumnTransformer(transformers=[
    ("num_feature", num_transformer, num_cols),
    ("bool_feature", bool_transformer, bool_cols),
    ("nom_feature", nom_transformer, nom_cols)
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier())
])

params = {
    "model__n_estimators": [100, 200, 300],
    "model__criterion": ["gini", "entropy", "log_loss"]
}

gridsearch = GridSearchCV(estimator= cls, param_grid=params,cv = 5, scoring="recall", verbose=2)
gridsearch.fit(x_train, y_train)
print(gridsearch.best_params_)
print(gridsearch.best_score_)
y_predict =gridsearch.predict(x_test)
print(classification_report(y_test, y_predict))









