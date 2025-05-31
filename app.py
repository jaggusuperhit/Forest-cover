import streamlit as st
import pickle
import numpy as np

# Load the model
try:
    rfc = pickle.load(open('rfc.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    rfc = None

# Define the cover type dictionary
COVER_TYPE_DICT = {
    1: {"name": "Spruce/Fir"},
    2: {"name": "Lodgepole Pine"},
    3: {"name": "Ponderosa Pine"},
    4: {"name": "Cottonwood/Willow"},
    5: {"name": "Aspen"},
    6: {"name": "Douglas-fir"},
    7: {"name": "Krummholz"}
}

def extract_features(user_input):
    try:
        user_input = user_input.split(',')
        features = np.array([user_input], dtype=np.float64)
        return features
    except ValueError:
        st.error("Invalid input features")
        return None

def predict_cover_type(features):
    try:
        output = rfc.predict(features).reshape(1, -1)
        return int(output[0])
    except Exception as e:
        st.error(f"Error predicting cover type: {e}")
        return None

# Creating web app
st.title('Forest Cover Type Prediction')
user_input = st.text_input('Input Features')

if user_input and rfc is not None:
    features = extract_features(user_input)
    if features is not None:
        predicted_cover_type = predict_cover_type(features)
        if predicted_cover_type is not None:
            cover_type_info = COVER_TYPE_DICT.get(predicted_cover_type)
            if cover_type_info is not None:
                cover_type_name = cover_type_info["name"]
                st.write("Predicted Cover Type:")
                st.write(f"<h1 style='font-size: 40px; font-weight: bold;'>{cover_type_name}</h1>", unsafe_allow_html=True)
            else:
                st.write("Unable to make a prediction")