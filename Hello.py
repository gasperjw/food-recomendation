import requests
import streamlit as st
import openai
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


# Hugging Face Classifier details
API_URL = "https://api-inference.huggingface.co/models/nateraw/food"
headers = {"Authorization": "Bearer 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"}

# OpenAI API key setup
openai.api_key = st.secrets["openapi"]

def query(image):
    response = requests.post(API_URL, headers=headers, data=image)
    return response.json()

def get_food_recommendations(food_item):
    prompt = f"Based on the food item '{food_item}', what are some recommended dishes or recipes?"
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=100)
    return response.choices[0].text.strip()

def main():
    st.title("Food Image Classifier")
    
    uploaded_image = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
        with st.spinner("Classifying..."):
            predictions = query(uploaded_image.getvalue())

        top_prediction = predictions[0]
        st.write("Top predicted food:", top_prediction["label"], "with confidence score:", top_prediction["score"])

        # Get food recommendations
        recommendations = get_food_recommendations(top_prediction["label"])
        st.write("Food Recommendations:", recommendations)

if __name__ == "__main__":
    main()