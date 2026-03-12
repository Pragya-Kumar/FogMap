from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time

# 1. Initialize the API
app = FastAPI(title="Fog Detection API", description="Classifies images as Clear or Smog")

# 2. Load the Model ONCE at startup (Reduces processing latency!)
print("Loading fine-tuned model into memory...")
MODEL_PATH = "smog_classifier_finetuned.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Warm-up the model so the first prediction is fast
dummy_input = np.zeros((1, 224, 224, 3))
model.predict(dummy_input, verbose=0)
print("Model loaded and warmed up!")

# 3. Create the Prediction Endpoint
@app.post("/predict")
async def predict_fog(file: UploadFile = File(...)):
    # Start the latency timer
    start_time = time.time()
    
    try:
        # Read the uploaded image in memory 
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess (Resize to 224x224 to match our training pipeline)
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        
        # Model Inference
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Process Thresholds
        threshold = 0.50
        if prediction > threshold:
            label = "Smog"
            confidence = float(prediction)
        else:
            label = "Clear"
            confidence = float(1 - prediction)
            
        # Stop the timer and calculate latency
        process_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Return the JSON Response with Latency Report
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": label,
            "confidence_percent": round(confidence * 100, 2),
            "latency_ms": process_time_ms
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}