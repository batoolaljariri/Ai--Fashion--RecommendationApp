# Ai--Fashion--RecommendationApp
An intelligent fashion recommendation system that blends weather data, computer vision, and user preferences to suggest the perfect outfit — whether it's sunny, rainy, or snowing! Built using deep learning, vector similarity search, and real-time APIs.

🌟 Features
🧠 Deep Learning-Based Feature Extraction
Uses ResNet50 + GlobalMaxPooling2D to extract 2048-dimensional feature vectors from clothing images.

🧼 Image Processing Pipeline
Automatically resizes, normalizes, and embeds user-uploaded images for consistent, high-accuracy matching.

🌦️ Weather-Aware Outfit Suggestions
Integrates with Tomorrow.io API to personalize recommendations based on real-time temperature and conditions.

🤖 Style Analysis with Groq + LLaMA
Sends outfit images to a Groq-based LLaMA model for generating fashion metadata (category, style, occasion, etc.).

🛍️ Online Store Integration
Suggests complementary clothing pieces from retail datasets based on detected style and user preferences.

🔎 K-Nearest Neighbors (KNN)
Matches uploaded images with similar outfits using vector similarity (Euclidean distance).

🛠️ Tech Stack
Python 3.9+

TensorFlow / Keras – ResNet50 model

Scikit-learn – KNN similarity matching

Streamlit – Frontend interface

Tomorrow.io API – Weather integration

Groq + LLaMA – AI vision-based metadata extraction

FastAPI / Pydantic – Schema validation

Pickle – Feature storage

PIL, NumPy, Requests – Image & data processing
