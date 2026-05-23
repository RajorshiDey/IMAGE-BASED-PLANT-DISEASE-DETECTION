import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

# 1. Define the 38 classes
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Global dictionary to hold our model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Rebuild Architecture and Load Weights ---
    print("Rebuilding architecture to bypass config mismatch...")
    IMG_SHAPE = (224, 224, 3)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights=None 
    )

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(38, activation='softmax')(x) 

    model = tf.keras.Model(inputs, outputs)

    print("Loading weights from disk...")
    # Ensure this file is in the same directory as this script
    model.load_weights("plant_disease_model_final.keras") 
    print("Weights loaded successfully!\n")
    
    ml_models["classifier"] = model
    
    yield
    
    # --- Shutdown: Clean up resources ---
    ml_models.clear()

# Initialize FastAPI app
app = FastAPI(title="Plant Disease Classifier API", lifespan=lifespan)

# Add this CORS middleware block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all domains to access your API
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Validate that the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the uploaded file into memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to match the model's expected input shape
        image = image.resize((224, 224))
        
        # Preprocess the image
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        
        # Retrieve the model and predict
        model = ml_models["classifier"]
        predictions = model.predict(img_array, verbose=0)
        
        # Extract results
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100) # Cast to native Python float for JSON serialization
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
        return {
            "filename": file.filename,
            "prediction": predicted_class_name,
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")