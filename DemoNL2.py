import pandas as pd
from ydata_profiling.profile_report import ProfileReport

profile = ProfileReport(data, title = "diabetes_report", explorative=True )
profile.to_file("diabetes_statistical")


numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
Q1 = data[numerical_columns].quantile(0.25)
Q3 = data[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
data_new = data[~((data[numerical_columns] < (Q1 - 1.5 * IQR)) |
                        (data[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]





