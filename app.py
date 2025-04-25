import os
import base64 #Encodes images into a format suitable for transmission over APIs
import pickle as pkl
import json
import re
import requests
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from io import BytesIO
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm
from pydantic import BaseModel #Validates and defines schemas for structured data.
from typing import List


# Initialize Groq client (Mock function for now, replace with actual Groq client initialization if available)
class Groq:
    def __init__(self, api_key):
        self.api_key = api_key

    def chat(self):
        pass
def initialize_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)

# Process image function using Groq API
def process_image(client, image_path: str, schema: dict):
    try:
        # Convert image to base64
        img = Image.open(image_path)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Mock response (replace with real API logic)
        response = {
            "clothing_category": "Jacket",
            "color": "Black",
            "style": "Casual",
            "occasion": "Winter",
            "suitable_for_weather": "Cold",
            "description": "Warm black jacket, ideal for cold weather."
        }
        return response
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# Fashion schema classes
class FashionBaseModel(BaseModel):
    clothing_category: str
    color: str
    style: str
    occasion: str
    suitable_for_weather: str
    description: str


class FashionModel(BaseModel):
    clothes: List[FashionBaseModel]

############################
# Tomorrow.io Weather API function
API_KEY = "0mtOZsjERX5mZ9tOfXHImEoVW650NZCa"
def get_weather_tomorrow_io(city):
    url = f'https://api.tomorrow.io/v4/weather/forecast?location={city}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    # Check if 'timelines' key exists
    #Extracts temperature and weather condition
    if 'timelines' in data:
        forecast = data['timelines']['daily'][0]  # Access daily forecast
        temperature = forecast['values']['temperatureAvg']  # Average temperature
        precipitation_type = forecast['values'].get('precipitationType', 'Clear')  # Default to Clear if not found
        return temperature, precipitation_type
    else:
        raise KeyError("Invalid response structure: 'timelines' key not found")


# Load and configure ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.models.Sequential([base_model, GlobalMaxPool2D()])

#datasets are used to recommend visually similar clothing items based on user input.
# # Load precomputed image features and corresponding images
dataset_choice = st.selectbox(
    "Choose Recommendation Source",
    ("Retail Dataset", "Database Dataset")
)

if dataset_choice == "Retail Dataset":
    file_img = pkl.load(open(r'image.pkl', 'rb'))
    feature_list = pkl.load(open(r'feature.pkl', 'rb'))
else:  # "Database Dataset"
    file_img = pkl.load(open(r'images.pkl', 'rb'))
    feature_list = pkl.load(open(r'fetaures.pkl', 'rb'))

# Nearest Neighbors for recommendations
#(KNN) with Euclidean distance is used to find the 5 most
# similar clothing items to the uploaded image.
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

# Streamlit app
st.header('Fashion Recommendation System')

# Weather-based recommendations
city = st.text_input("Enter your city for weather-based recommendations:", "Amman")

if city:
    try:
        # Fetch weather data
        temperature, condition = get_weather_tomorrow_io(city)
        # Display weather details
        st.subheader("Weather Information")
        st.write(f"City: {city}")
        st.write(f"Forecasted Temperature: {temperature}°C")
        st.write(f"Weather Condition: {condition}")
        #st.subheader("Weather-Based Outfit Recommendations")
        if temperature < 10:
            st.write("It's cold outside! Here are some warm outfit recommendations.")
        elif condition.lower() == "rain":
            st.write("It looks like rain! Here are some waterproof clothing recommendations.")
        else:
            st.write("Here are some general outfit recommendations for today's forecast.")
    except Exception as e:
        st.error("Error fetching weather data. Please check your city name or API key.")
        st.error(e)

# Image upload and recommendation
if not os.path.exists('upload'):
    os.makedirs('upload')

upload_file = st.file_uploader("Upload an Image for Fashion Recommendation")
if upload_file is not None:
    # Save and display uploaded image
    image_path = os.path.join('upload', upload_file.name)
    with open(image_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    st.subheader('Uploaded Image')
    st.image(upload_file)
    # Define schema for processing
    schema = FashionModel.model_json_schema()
    # Process uploaded image
    try:
        features = process_image(client=None, image_path=image_path, schema=schema)
        if features:
            st.subheader('Extracted Features')
            st.json(features)

            # Extract features using ResNet50
            input_img_features = preprocess_input(image.img_to_array(image.load_img(image_path, target_size=(224, 224))))
            input_img_features = np.expand_dims(input_img_features, axis=0)
            input_img_features = model.predict(input_img_features).flatten()
            input_img_features = input_img_features / norm(input_img_features)

            # Find the most similar images
            distances, indices = neighbors.kneighbors([input_img_features])
            st.subheader('Recommended Images')

            # Display recommended images in columns
            col1, col2, col3, col4, col5  = st.columns(5)
            with col1:
                st.image(file_img[indices[0][1]])
            with col2:
                st.image(file_img[indices[0][2]])
            with col3:
                st.image(file_img[indices[0][3]])
            with col4:
                st.image(file_img[indices[0][4]])
            with col5:
                st.image(file_img[indices[0][5]])
        else:
            st.write("No valid features extracted from the image.")
    except Exception as e:
        st.write("Error processing the image:")
        st.write(e)
