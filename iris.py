
# Iris flower species classification using DecisionTreeClassifier

import pandas as pd

dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder

labelencoder_species_name = LabelEncoder()
Y = labelencoder_species_name.fit_transform(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

species_predict = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
prediction_accuracy = accuracy_score(y_test, species_predict) * 100;
