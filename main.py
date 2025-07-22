from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi.middleware.cors import CORSMiddleware 
from PIL import Image
import io
import numpy as np


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = load_model("cats_vs_dogs_new.keras")

@app.get("/")
def return_root():
    return{"message":"hiiii"}

@app.post("/predict")
async def return_prediction(file: UploadFile=File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((64, 64))  # Resize properly
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = float(prediction if prediction > 0.5 else (1 - prediction))

    return {
        "prediction":label,
        "confidence":round(confidence, 2)
    }

