# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import requests


LOGGER = get_logger(__name__)

# Hugging Face Classifier details
API_URL = "https://api-inference.huggingface.co/models/nateraw/food"
headers = {"Authorization": "Bearer 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"}

def query(image):
    response = requests.post(API_URL, headers=headers, data=image)
    return response.json()

def main():
    st.title("Food Image Classifier")
    
    uploaded_image = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
        with st.spinner("Classifying..."):
            predictions = query(uploaded_image.getvalue())

        top_prediction = predictions[0]
        st.write("Top predicted food:", top_prediction["label"], "with confidence score:", top_prediction["score"])

        
if __name__ == "__main__":
    main()
