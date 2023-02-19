import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Starting the app with a title
with st.sidebar:
    st.title("SWISS BANKNOTE COUNTERFEIT DETECTION")
    st.write("""### -A machine learning solution""")
    st.image("https://i.imgur.com/5ES391q.jpg")
    st.write("""
             **Data source on [Kaggle](https://www.kaggle.com/datasets/chrizzles/swiss-banknote-conterfeit-detection/code)**
    """)
    st.write("""
    ## CAPSTONE PROJECT
    ### By Team AWS
    #### FALL '22 cohort of Hamoye Internship
    """)
    st.info("This web app uses machine learning to determine whether a bank note is counterfeit or genuine."
            " This is specific to the SWISS "
            "bank note")
# Loading the model
cwd = os.getcwd()
# Getting the values for the prediction
st.info("All units are in mm")
col1, col2 = st.columns(spec=2, gap="medium")
with col1:
    length = st.number_input("Input the dimension of the length (Length)", step=1., format="%.1f")
    left = st.number_input("Input the dimension of the width of left edge (Left)", step=1., format="%.1f")
    right = st.number_input("Input the dimension of the width of right edge (Right)", step=1., format="%.1f")

with col2:
    bottom = st.number_input("Input the dimension of the bottom margin (Bottom)", step=1., format="%.1f")
    top = st.number_input("Input the dimension of the top margin (Top)", step=1., format="%.1f")
    diagonal = st.number_input("Input the dimension of the length of diagonal (Diagonal)", step=1., format="%.1f")


@st.cache_data()
def load_data(data_path):
    data = pd.read_csv(data_path)
    return data


@st.cache_resource()
def train_model(model_algo):
    data = load_data(os.path.join(cwd, "banknotes.csv"))
    x = data.copy()
    y = x.pop('conterfeit')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = model_algo()
    model = model.fit(x_train, y_train)
    return model


array = [length, left, right, bottom, top, diagonal]
full_array = [array]


def predict():
    if all(array):
        model = train_model(RandomForestClassifier)
        prediction = model.predict(full_array)
        if prediction == 0:
            return st.success("This is a genuine note")
        else:
            return st.error("This is a fake (counterfeit) note")
    else:
        st.error("Kindly fill all the appropriate columns")


st.button("PREDICT", on_click=predict)
