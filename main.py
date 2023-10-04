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
clf.fit(X, Y)

predictions = clf.predict(X)
# clf.get_deph() get_n_leaves()
acc = accuracy_score(Y, predictions)
print("Accuracy: {: .4%}".format(acc))

plt.figure(figsize=(40, 15))
tree.plot_tree(clf, rounded=True, filled=True, feature_names=list(data.columns))
plt.savefig('tree.png')
plt.show()

print(' N \t\t На обучающей', 'На тестовой', sep=" ")
for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    clf.fit(X_train, Y_train)
    print(f'{(i + 1):2} {clf.score(X_train, Y_train) :15.6f} {clf.score(X_test, Y_test) :15.6f}')
