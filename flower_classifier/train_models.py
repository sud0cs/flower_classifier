import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle, os

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

def custompredict(self, X, normalize=True):
    names = iris.target_names
    if normalize:
        X = sc.transform(X)
    prediction = self.predict(X)
    return [names[i] for i in prediction]
LogisticRegression.custompredict = custompredict
lr = LogisticRegression(C=50.0,
                        solver = 'lbfgs',
                        multi_class='ovr')

lr.fit(X_train_std, y_train)
SVC.custompredict = custompredict
svm = SVC(kernel='rbf', C=1, gamma=.1, random_state=1)
svm.fit(X_train_std, y_train)
DecisionTreeClassifier.custompredict = custompredict
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=1)
tree_model.fit(X_train_std, y_train)
RandomForestClassifier.custompredict = custompredict
f0rest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs = 16)
f0rest.fit(X_train_std, y_train)
KNeighborsClassifier.custompredict = custompredict
knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
print(tree_model.custompredict([[1.0, 3.0]]))

models = [lr, svm, f0rest, knn]
if 'models' not in os.listdir('..'):
    os.mkdir('../models')

for model in models:
    with open(f'../models/{model.__class__.__name__.lower().replace('classifier', '')}model.pck', 'wb') as file:
        pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)
