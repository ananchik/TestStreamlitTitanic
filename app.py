import streamlit as st
import joblib
import pandas as pd
import numpy as np
from utils import PrepProcesor, columns 

model = joblib.load('xgbpipe.joblib')

st.title('Did they survive? :ship:')

passengerid = st.text_input("Input Passenger ID", '1234')
pclass = st.selectbox("Choose class", [1,2,3])
name = st.text_input("Input name", "John Smith")
parch = st.slider("Choose parch", 0,2)
ticket = st.text_input("Input Ticket ID", '1234')
sex = st.selectbox("chose sex", ["male", "female"])
sibsp = st.slider("Choose sibsp", 0,10)
age = st.slider("choose age", 0,100)
fare = st.number_input("Input fare", 0, 1000)
cabin = st.text_input("Input cabin", "C51")
embarked = st.select_slider("Did they Embarked", ['S', 'C', 'Q'])

# ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked']
def predict():
    row = np.array([passengerid, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked])
    X = pd.DataFrame([row], columns=columns)
    prediction = model.predict(X)
    if prediction[0] == 1:
        st.success('Passenger Survived :thumbsup:')
    else :
        st.error('Passenger didn"t Survive :thumbsdown:')

trigger = st.button('Predict', on_click=predict)
