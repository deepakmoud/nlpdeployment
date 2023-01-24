# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:20:35 2021

@author: deepak
"""

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('svmmodel.pkl', 'rb')) 
model_randomforest = pickle.load(open('randomforest.pkl', 'rb')) 
dataset= pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(UserID, Gender,Age,EstimatedSalary):
  output= model.predict(sc.transform([[Age,EstimatedSalary]]))
  print("Purchased", output)
  if output==[1]:
    prediction="Item will be purchased"
  else:
    prediction="Item will not be purchased"
  print(prediction)
  return prediction
def predict_random(UserID, Gender,Age,EstimatedSalary):
  output= model_randomforest.predict(sc.transform([[Age,EstimatedSalary]]))
  print("Purchased", output)
  if output==[1]:
    prediction="Item will be purchased"
  else:
    prediction="Item will not be purchased"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Six days FDP on Deep learningk and Machine Learning</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Item Purchase Prediction using SVM Algorithm")
    UserID = st.text_input("UserID","Type Here")
    Gender = st.selectbox(
    "Gender",
    ("Male", "Female", "Others")
    )
    
    Age = st.number_input('Insert a Age',0,100)
    #Age = st.text_input("Age","Type Here")
    EstimatedSalary = st.number_input("Insert EstimatedSalary",1000,1000000000)
    resul=""
    if st.button("SVM Prediction"):
      result=predict_note_authentication(UserID, Gender,Age,EstimatedSalary)
      st.success('SVM Model has predicted {}'.format(result))
    if st.button("Random Forest Prediction"):
      result=predict_random(UserID, Gender,Age,EstimatedSalary)
      st.success('Random forest Model  has predicted {}'.format(result))  
    if st.button("About"):
      st.header("Developed by Mr. Deepak Moud")
      st.subheader("Head , Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Experiment : Support Vector Machine and Random Forest</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()
