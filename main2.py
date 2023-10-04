from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def imput(dataset):
    # imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer.fit(dataset)
    return imputer.transform(dataset)


data = pd.read_csv('heart_data.csv')
data = data.replace('?', np.nan).astype(float)
data.head()

predictions = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
               'thal']
outcome = 'goal'

X = data.loc[1:, predictions]
Y = data.loc[1:, outcome]
X = imput(X)  # корректные данные без пропусков

# max_depth=5 аргумент максимальной глубины max_leaf_nodes=16 максимальное кол-во листьев
clf = DecisionTreeClassifier()
# clf.fit(X, Y)

for i in range(10):
    print(f"{i+1} splitting:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    clf.fit(X_train, Y_train)

    predictions = clf.predict(X_train)
    acc = accuracy_score(Y_train, predictions)
    print("Accuracy Train: {: .4%}".format(acc))


    plt.figure(figsize=(40, 15))
    tree.plot_tree(clf, rounded=True, filled=True, feature_names=list(data.columns))
    plt.savefig(f'tree{i}_train.png')

    predictions2 = clf.predict(X_test)
    acc2 = accuracy_score(Y_test, predictions2)
    print("Accuracy Test: {: .4%}".format(acc2))


    plt.figure(figsize=(40, 15))
    tree.plot_tree(clf, rounded=True, filled=True, feature_names=list(data.columns))
    plt.savefig(f'tree{i}_test.png')

