#!/usr/bin/env python
# coding: utf-8

# ## <font color = 'red'> Project: Predicting if a patient has heart disease

# ### <mark>Imports</mark>

# https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

# In[1]:


# Import the necessary libraries for steamlit deployment
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


# ### <mark>Model Development</mark>

# In[3]:


# Choosing 4 most important features, separate the train and test variables and train the model
df = pd.read_csv('heart.csv')
encoder_columns = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_data = encoder.fit_transform(df[encoder_columns])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(encoder_columns))

df_encoded = pd.concat([df.drop(encoder_columns, axis=1), encoded_df], axis=1)

X = df_encoded[['Age','MaxHR','Cholesterol','ST_Slope_Up']]
y = df_encoded['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = LogisticRegression(C=4.281332398719396, max_iter=5000, solver='sag')
model.fit(X_train.values, y_train)


# ### <mark>Develop Streamlit Model</mark>

# In[7]:


import streamlit as st
st.title('Heart Disease Prediction')

# Input
Age = st.sidebar.slider("Age", float(X['Age'].min()), float(X['Age'].max()), float(X['Age'].mean()))
MaxHR = st.sidebar.slider("Max Heart Rate", float(X['MaxHR'].min()), float(X['MaxHR'].max()), float(X['MaxHR'].mean()))
Cholesterol = st.sidebar.slider("Cholesterol", float(X['Cholesterol'].min()), float(X['Cholesterol'].max()), float(X['Cholesterol'].mean()))
ST = st.sidebar.slider("ST Slope Up", float(X['ST_Slope_Up'].min()), float(X['ST_Slope_Up'].max()), float(X['ST_Slope_Up'].mean()))

# Predict the class 
input_data = (Age, MaxHR, Cholesterol, ST)
input_array = np.asarray(input_data,dtype=np.float64)
input_array_reshaped = input_array.reshape(1,-1)
prediction = model.predict(input_array_reshaped)
prediction_proba = model.predict_proba(input_array_reshaped)


# Display the prediction
st.subheader("Prediction")
st.write(f"Predicted Class: ",prediction[0])
st.write(f"No Disease Probability: {prediction_proba[0][0]:.2f}")
st.write(f"Disease Probability: {prediction_proba[0][1]:.2f}")
st.write(f"Accuracy: {model.score(X_test.values, y_test)}")


# In[ ]:




