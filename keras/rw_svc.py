from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv('openpose_data3.csv')
X = df.drop("category", axis=1)
y = df["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=False)

tuned_parameters = [
#  {'C': [1, 10, 100, 1000]},
#  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
#  {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3], 'gamma': [0.001, 0.0001]},
#  {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
   {'C': [100], 'kernel': ['poly'], 'degree': [2], 'gamma': [0.001]}
]

svc_estimator = SVC(C=10, gamma=0.0001, kernel='rbf')
svc_model_tuning = GridSearchCV(
  estimator = svc_estimator,     # 識別器
  param_grid= tuned_parameters,
  cv = 2,                    # Cross validationの分割数
  n_jobs = -1,
  verbose = 3
)
svc_estimator.fit(X_train, y_train)

#print(svc_model_tuning.best_estimator_)
#print(svc_model_tuning.best_params_)

#pred = svc_model_tuning.predict(X_test)
#print(classification_report(y_test, pred))

print("train data score")
print(svc_estimator.score(X_train, y_train))
print("test data score")
print(svc_estimator.score(X_test, y_test))
