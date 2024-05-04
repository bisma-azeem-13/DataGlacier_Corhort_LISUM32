import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify, render_template

iris=load_iris()
dir(iris)

iris.feature_names

iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df.head()

iris_df['target']=iris.target
iris_df

iris_df['flower_name']=iris_df.target.apply(lambda x : iris.target_names[x])
iris_df

iris_df=iris_df.drop(columns='target')

iris_df

X=iris_df.drop(columns='flower_name')
y=iris_df['flower_name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr=LogisticRegression()


lr.fit(X_train,y_train)

pickle.dump(lr,open('model.pkl','wb'))