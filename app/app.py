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
    radio = st.radio("Choose your action from here", options=["Explore the dataset", "Test the ML model"])
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
    
cwd = os.getcwd()

@st.cache_data()
def load_data(data_path):
    data = pd.read_csv(data_path)
    return data


if radio == "Test the ML model":
    # Getting the values for the prediction
    st.title("Model testing section")
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


    @st.cache_resource()
    def train_model(model_algo):
        data = load_data(os.path.join(cwd, "app/banknotes.csv"))
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
else:
    st.title("Data understanding section")
    st.info("If this is your first time, you probably don't know what to do, kindly take time to go through this section before using the ML model")
    data = load_data(os.path.join(cwd, "app/banknotes.csv"))
    st.write("First five rows")
    st.dataframe(data.head())
    st.write("Last five rows")
    st.dataframe(data.tail())
    st.write(f"In total, there are {data.shape[0]} data points in the dataset, with {data.shape[1]} features")
    st.markdown("### Explanation to help understand the data")
    st.write("""
    In the dataframes above, the first one is showing the first five rows of the data, while the last five rows can be seen in the second
    dataframe. The reason for showing the two dataframes, being that, the first five shows cases of genuine notes and the second shows instances
    of counterfeit notes.

    ### Understanding each of the columns
    1. conterfeit: This is the target column we will be getting predictions for and it contains two unique classes (0 and 1)
    which translates directly into genuine and counterfeit notes respectively. This makes our task a binary classification task.
    2. Length: The dimension of the length of the note.
    3. Left: The height of the note measured from the left.
    4. Right: The height of the note measured from the right. Left and Right should be the same, except there are errors, which cannot be avoided.
    5. Bottom: Distance of the inner frame to the border of the note, at the bottom.
    6. Top: Distance of the inner frame to the border of the note, at the top.
    7. Diagonal: Dimension of the diagonal distance of the note.
    All dimensions are in millimeters (mm)
    The seven columns above are the columns in the dataset as one can see, and they should all make sense now.

    ### Guide to using the ML model
    To use the machine model the following informations should be kept in mind:
    1. There are 6 number input boxes for each of the six features in the dataset (all but the target column (conterfeit)).
       Each of the fields is correctly labelled with the associated column name.
    2. The 0.0 in the fields must be cleared out before you start typing your inputs.
      An easy way to do that is by double tapping the 0.0 and just start typing.
    3. The "+" and "-" sign beside each field is to add or subtract 1 from the input.
    4. Be sure to enter only numbers without adding the units.
    5. Make sure you have filled in all the fields before clicking on "Predict", else you will get a warning asking you to do the same.
     don't worry, you wont lose the data you entered before clicking on "Predict".
    6. Click on "Predict" to get the model to do it's magic and give you the output you are looking for.
    7. Check the sidebar on the left to see the radio buttons to switch between this section and the ML testing section
    8. Know that switching will make you lose whatever input you have in the model testing section. So, be careful when making switches  
    """)
    