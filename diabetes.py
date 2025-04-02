

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from ydata_profiling.profile_report import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

data = pd.read_csv("diabetes_dataset_extended.csv")
data = data.drop("Patient_ID", axis = 1)
# profile = ProfileReport(data, title = "diabetes_report", explorative=True )
# profile.to_file("diabetes_statistical")

# split_data
target = "Outcome"
x = data.drop(target, axis = 1)
y = data[target]

bool_cols =["Smoker"]
num_cols = ["Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"]
nom_cols = ["Notes"]
ord_cols =["BMI_Category"]

x[bool_cols] = x[bool_cols].replace(["N/A", "p", "nan"], np.nan)
x[bool_cols] = x[bool_cols].astype(bool).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

bmi_category_value =["Underweight","Normal", "Overweight", "Obese"]

#chuan hoa bool
bool_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=False)),
    ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False))
])
#chuan hoa du lieu co thu tu
ord_transformer = Pipeline(steps=[
     ("imputer", SimpleImputer(strategy="most_frequent")),
     ("encoder", OrdinalEncoder(categories=[bmi_category_value]))
])
#chuan hoa du lieu k co thu tu
nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=True))
])
#chuan hoa dang so
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(missing_values= np.nan, strategy="median")),
    ("encoder", StandardScaler())
])

#preprocessor data
preprocessor =ColumnTransformer(transformers=[
    ("bool_feature", bool_transformer, bool_cols),
    ("ord_feature", ord_transformer, ord_cols),
    ("nom_feature", nom_transformer, nom_cols),
    ("num_feature", num_transformer, num_cols),
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42,
                                     bootstrap=False,
                                     max_depth=10,
                                     max_features='sqrt',
                                     min_samples_leaf=1,
                                     min_samples_split=5,
                                     n_estimators=100))
])
cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))

joblib.dump(cls, "diabetes_model.pkl")

#
# joblib.dump(cls, "diabetes_pipeline.pkl")
# print("Pipeline đã được lưu thành diabetes_pipeline.pkl")

#test