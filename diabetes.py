
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import pickle
from ydata_profiling.profile_report import ProfileReport

data = pd.read_csv("diabetes_dataset_extended.csv")
statistic = data.describe()
data = data.drop("Patient_ID", axis = 1)
data = data.drop("BMI_Category", axis = 1)
# profile = ProfileReport(data, title = "diabetes_report", explorative=True )
# profile.to_file("diabetes_statistical")

# chia dư lieu theo chieu ngan
target = "Outcome"
x = data.drop(target, axis = 1)
y = data[target]

# chia du lieu theo chiu doc


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

x[bool_cols] = x[bool_cols].replace(["N/A", "p", "nan"], np.nan)
x[bool_cols] = x[bool_cols].astype(bool).astype(int)

Q1 = data[num_cols].quantile(0.25)
Q3 = data[num_cols].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data[num_cols] < (Q1 - 1.5 * IQR)) |
                        (data[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


bmi_category_value =["Underweight","Normal", "Overweight", "Obese"]

#chuan hoa bool
bool_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=False)),
    ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False))
])
# bool_transformer = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False))
# ])

#chuan hoa du lieu co thu tu
# ord_transformer = Pipeline(steps=[
#      ("imputer", SimpleImputer(strategy="most_frequent")),
#      ("encoder", OrdinalEncoder(categories=[bmi_category_value]))
# ])
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
    ("nom_feature", nom_transformer, nom_cols),
    ("num_feature", num_transformer, num_cols),
])





cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
                                     random_state=42,
                                     bootstrap=True,
                                     max_depth=20,
                                     max_features='sqrt',
                                     min_samples_leaf=2,
                                     min_samples_split=5,
                                     n_estimators=200))
])
cls.fit(x_train, y_train)


# param_grid = {
#     'model__n_estimators': [100, 200, 300],
#     'model__max_depth': [10, 20, 30, None],
#     'model__min_samples_split': [2, 5, 10],
#     'model__min_samples_leaf': [1, 2, 4],
#     'model__max_features': ['sqrt', 'log2'],
#     'model__bootstrap': [True, False],
# }
# gridsearch = GridSearchCV(estimator=cls, param_grid=param_grid, cv = 5,scoring="recall", n_jobs=5, verbose=2)
# gridsearch.fit(x_train, y_train)
# print(gridsearch.best_params_)
# print(gridsearch.best_score_)
# y_predict =gridsearch.predict(x_test)
# print(classification_report(y_test, y_predict))

# {'model__bootstrap': True, 'model__max_depth': 20, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 2, 'model__min_samples_split': 5, 'model__n_estimators': 200}




inputs = pd.DataFrame({
    "Pregnancies": 3,
    "Glucose": 120,
    "BloodPressure": 80,
    "SkinThickness": 20,
    "Insulin": 100,
    "BMI": 30.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 45,
    "Smoker": 1,
    "Notes": ["No symptoms"]

    #
    # "Pregnancies": 2,
    # "Glucose": 90,
    # "BloodPressure": 75,
    # "SkinThickness": 18,
    # "Insulin": 80,
    # "BMI": 22.5,
    # "DiabetesPedigreeFunction": 0.2,
    # "Age": 35,
    # "Smoker": 0,
    # "Notes": ["No symptoms"]
})
predictions = cls.predict(inputs)[0]
print(predictions)


# y_predict = cls.predict(x_test)
# print(classification_report(y_test, y_predict))
#
# with open("model.pkl", "wb") as f:
#     pickle.dump(cls, f)