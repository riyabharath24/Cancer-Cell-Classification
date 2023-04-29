import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import time
import streamlit as st
import sklearn
from sklearn.svm import SVC
import itertools

model = pkl.load(open('rbfweights.pkl', 'rb'))


df = pd.read_csv('cancer_cell_dataset.csv')
df.head()


inputlist = []

html1 = """
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h1> Cancer Cell Classifier </h1>
    </div>
      """
st.markdown(html1, unsafe_allow_html=True)

html2 = """
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h2>Find out if the Tumour is Benign or Malignant <h2>
    </div>
      """
st.markdown(html2, unsafe_allow_html=True)

st.sidebar.title("Your Information")

Name = st.sidebar.text_input("Full Name")

Contact_Number = st.sidebar.text_input("Contact Number")

Email_address = st.sidebar.text_input("Email address")

if not Name and Email_address:
    st.sidebar.warning("Please fill out your name and EmailID")

if Name and Contact_Number and Email_address:
    st.sidebar.success("Thanks!")

st.write('Fill in your measurements here')


Clump = st.slider(
    'Clump Size', 0, 10, 1)
st.write(Clump)
inputlist.append(Clump)


UnifSize = st.slider(
    'UnifSize', 0, 10, 1)
st.write(UnifSize)
inputlist.append(UnifSize)

UnifShape = st.slider(
    'UnifShape', 0, 10, 1)
st.write(UnifShape)
inputlist.append(UnifShape)

MargAdh = st.slider(
    'MargAdh', 0, 10, 1)
st.write(MargAdh)
inputlist.append(MargAdh)

SingEpiSize = st.slider(
    'SingEpiSize', 0, 10, 1)
st.write(SingEpiSize)
inputlist.append(SingEpiSize)

BareNuc = st.slider(
    'BareNuc', 0, 10, 1)
st.write(BareNuc)
inputlist.append(BareNuc)

BlandChrom = st.slider(
    'BlandChrom', 0, 10, 1)
st.write(BlandChrom)
inputlist.append(BlandChrom)

NormNucl = st.slider(
    'NormNucl', 0, 10, 1)
st.write(NormNucl)
inputlist.append(NormNucl)

Mit = st.slider(
    'Mit', 0, 10, 1)
st.write(Mit)
inputlist.append(Mit)


if st.button("Predict"):
    result = model.predict([inputlist])

    if result == 2:
        st.write('The tumour is Benign - it is not cancerous!')
    if result == 4:
        st.write('The tumour is Malignant - Consult a doctor now')

    st.write('For more info click here👇')

    st.write("[Benign vs Malignant Tumours](https://www.cancercenter.com/community/blog/2023/01/whats-the-difference-benign-vs-malignant-tumors)")

    import seaborn as sns

    f, axes = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(df.corr(), cmap='coolwarm', cbar=True)
    st.subheader("Exploratory Data Analysis on the Dataset: ")
    st.text("Correlation Between Numerical Features")
    st.pyplot(f)
