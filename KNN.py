#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Load the dataset
url = "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
# Load the data using pandas
data = pd.read_csv("diabetes.csv")

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[2]:


class StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
    def transform(self, X):
        scaled_X = (X - self.mean) / self.std
        return scaled_X


# In[3]:


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the scaler on training data
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[4]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut

# Determine the best K value using cross-validation
k_values = list(range(1, 21))
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    accuracy_scores.append(np.mean(scores))

best_k = k_values[np.argmax(accuracy_scores)]

# Plot accuracy vs K value
import matplotlib.pyplot as plt

plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K Value')
plt.grid()
plt.show()

# Train the KNN classifier with the best K value
knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(X_train_scaled, y_train)

# Evaluate on the test set
y_pred = knn_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Explanation
explanation = "The KNN classifier achieved an accuracy of {:.2f}% on the test set. The confusion matrix:\n\n".format(accuracy * 100)
explanation += str(conf_matrix)


# In[5]:


loo = LeaveOneOut()
loo_scores = cross_val_score(knn_classifier, X_train_scaled, y_train, cv=loo)
mean_loo_accuracy = np.mean(loo_scores)
std_loo_accuracy = np.std(loo_scores)


# In[6]:


# Mean and standard deviation of 5-fold cross-validation
mean_cv_accuracy = np.mean(accuracy_scores)
std_cv_accuracy = np.std(accuracy_scores)

# Print mean and standard deviation of 5-fold cross-validation
print("Mean accuracy of 5-fold cross-validation:", mean_cv_accuracy)
print("Standard deviation of 5-fold cross-validation:", std_cv_accuracy)

# Print mean and standard deviation of leave-one-out cross-validation
print("Mean accuracy of leave-one-out cross-validation:", mean_loo_accuracy)
print("Standard deviation of leave-one-out cross-validation:", std_loo_accuracy)


# In[7]:


from sklearn.metrics import confusion_matrix
y_pred = knn_classifier.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


# In[8]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the KNN classifier on the test set: {:.2f}%".format(accuracy * 100))


# In[ ]:




