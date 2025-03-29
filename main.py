import pandas as pn
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier


data = pn.read_csv("diabetes.csv")

results = data.describe()
# results = data.info()

profile =  ProfileReport(data, title = "Diabetes Report", explorative= True)
profile.to_file("report_html")

target = "Outcome"
x_data = data.drop(target, axis = 1)
y_data = data[target]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=2004)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state = 2004)


#data processing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#model
# modle = RandomForestClassifier(n_estimators= 200, random_state=14, criterion= "gini")
# modle.fit(x_train, y_train)
#
# #test
# y_predict = modle.predict(x_test)
#
# print(classification_report(y_test, y_predict))
# print(len(y_predict))

# params = {
#     "n_estimators" : [100, 200, 300],
#     "criterion": ["gini", "entropy", "log_loss"]
# }
#
# gridsearch = GridSearchCV(estimator=RandomForestClassifier(random_state= 100), param_grid= params, cv = 5, scoring= "recall",  verbose= 2)
# gridsearch.fit(x_train, y_train)
# print(gridsearch.best_score_)
# print(gridsearch.best_params_)
# print(gridsearch.best_estimator_)
#
# y_predict = gridsearch.predict(x_test)
# print(classification_report(y_test, y_predict))


clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test )
print(models)

