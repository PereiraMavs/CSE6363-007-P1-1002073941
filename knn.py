import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,f1_score,confusion_matrix, accuracy_score, recall_score
    
# Importing the dataset
dataset = pd.read_csv('P1.csv')
X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 2].values
    
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting K-NN to the Training set where K=3
classifier_3 = KNeighborsClassifier(n_neighbors = 3, metric = 'euclidean', p = 2)
classifier_3.fit(X_train, y_train)

# Predicting the Test set results
y_pred_3 = classifier_3.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_3)

accuracy = accuracy_score(y_test, y_pred_3)

recall = recall_score(y_test, y_pred_3)

report = classification_report(y_test, y_pred_3)

f1 = f1_score(y_test, y_pred_3)

#Report for 3 neigbours 7368421052631579
print("#---Classification Report for 3 neighbours---#")
print("---Confusion Matrix---")
print(cm)
print("---Accuracy---")
print(accuracy)
print("---Recall---")
print(recall)
print("---Predicted Values---")
print(y_pred_3)
print("---f1 Score---")
print(f1)

# Fitting K-NN to the Training set where K=5
classifier_5 = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean', p = 2)
classifier_5.fit(X_train, y_train)

# Predicting the Test set results
y_pred_5 = classifier_5.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_5)

accuracy = accuracy_score(y_test, y_pred_5)

recall = recall_score(y_test, y_pred_5)

report = classification_report(y_test, y_pred_5)

f1 = f1_score(y_test, y_pred_5)

#Report for 3 neigbours
print("#---Classification Report for 5 neighbours---#")
print("---Confusion Matrix---")
print(cm)
print("---Accuracy---")
print(accuracy)
print("---Recall---")
print(recall)
print("---Predicted Values---")
print(y_pred_5)
print("---f1 Score---")
print(f1)

# Fitting K-NN to the Training set where K=7
classifier_7 = KNeighborsClassifier(n_neighbors = 7, metric = 'euclidean')
classifier_7.fit(X_train, y_train)

# Predicting the Test set results
y_pred_7 = classifier_7.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_7)

accuracy = accuracy_score(y_test, y_pred_7)

recall = recall_score(y_test, y_pred_7)


f17 = f1_score(y_test, y_pred_7)

#Report for 7 neigbours
print("#---Classification Report for 7 neighbours---#")
print("---Confusion Matrix---")
print(cm)
print("---Accuracy---")
print(accuracy)
print("---Recall---")
print(recall)
print("---Predicted Values---")
print(y_pred_7)
print("---f1 Score---")
print(f17)