import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.model_utils import load_and_evaluate_model

st.set_page_config(page_title="Lung Disease Classification")

# Load model
model_path = r'Models/LungDiseaseModel-CNN.keras'



def load_model_for_prediction(model_path):
    """Load the trained model."""
    return load_model(model_path)

def preprocess_image(img, target_size):
    """Preprocess image for prediction."""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

def predict_image(model, img_array, class_names):
    """Predict the class of the image using the model."""
    prediction = model.predict(img_array, verbose=0)
    print(prediction)
    predicted_class = np.argmax(prediction)
    print(predicted_class)
    return class_names[predicted_class]

def main():
    st.title('Lung Disease Classification')
    st.write('Upload an X-ray image of a lung to get a prediction.')
    prediction = None
    # Upload image
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"],)
    cols = st.columns(2)
    with cols[0]:
        if uploaded_file is not None:
            img = image.load_img(uploaded_file,color_mode="grayscale")
            st.image(img, caption='Uploaded Image.')
            
            # Load model
            class_names = ["Corona Virus Disease", "Normal", "Pneumonia", "Tuberculosis"]
            model = load_model_for_prediction(model_path)
            
            # Preprocess image
            target_size = (150, 150)  # Ensure this matches the input size expected by your model
            img_array = preprocess_image(img, target_size)
        
        # Predict
            prediction = predict_image(model, img_array, class_names)

    with cols[1]:
        text_to_display = f'Prediction: {prediction}'
        if prediction is not None:
            if prediction == "Normal":
                st.success(text_to_display)
            else:
                st.error(text_to_display)
        
        # Display result
        

if __name__ == "__main__":
    main()
