# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

##importing dataset

df=pd.read_csv("D:\SCIT STUDY\smartbridge\carprice\car data.csv")
df.shape
print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())

##check missing values

df.isnull().sum()
df.describe()
final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
final_dataset.head()

##Calculating No_Of_Years

final_dataset['Current Year']=2023
final_dataset.head()
final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year']
final_dataset.head()

##Dropping unnecessary columns
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.head()
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
final_dataset=final_dataset.drop(['Current Year'],axis=1)
final_dataset.head()
final_dataset.corr()
sns.pairplot(final_dataset)

#get correlations of each features in dataset

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))


#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
X['Owner'].unique()

### Feature Importance

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

# Splitting data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Support Vector Machines Classifier
svm = SVC()
svm.fit(X_train, y_train)
print("The accuracy of SVM is",
      svm.score(X_train,y_train), svm.score(X_test,y_test))
svm = [svm.score(X_train,y_train), svm.score(X_test,y_test)]

#Decision Tree Classifier Model
dtclassifier = DecisionTreeClassifier(max_depth=7)
dtclassifier.fit(X_train,y_train)
print("The accuracy of Decision Tree Classifier is",
      dtclassifier.score(X_train,y_train),dtclassifier.score(X_test,y_test))
dt = [dtclassifier.score(X_train,y_train),dtclassifier.score(X_test,y_test)]

#Random Forest Classifier Model
rfclassifier = RandomForestClassifier(max_depth = 7)
rfclassifier.fit(X_train, y_train)
print("The accuracy of random forest Classifier is",
      rfclassifier.score(X_train,y_train), rfclassifier.score(X_test,y_test))
rf = [rfclassifier.score(X_train,y_train), rfclassifier.score(X_test,y_test)]

#Linear Regression Model
regr = LinearRegression()
regr.fit(X_train, y_train)
print("The accuracy of linear regressor is",
      regr.score(X_train,y_train), regr.score(X_test,y_test))
reg = [regr.score(X_train,y_train), regr.score(X_test,y_test)]


#Results table for comparison of accuracies
results1 = pd.DataFrame(data=[svm,dt,rf,reg],
                        columns = ['Training Accuracy ', 'Testing Accuracy '],
                        index = ['Support Vector Machine',
                                 'Decision Tree', 'Random Forest', 'linear regression'])

import pickle
# open a file, where you want to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rfclassifier, file)