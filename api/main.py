from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image

import uvicorn
import numpy as np
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/potatoes.h5")
CLASS_NAMES = ['Potato Early Blight', 'Potato Late Blight', 'Potato Healthy']

def read_file_as_image(file) -> np.ndarray:
    image = np.array(Image.open(BytesIO(file)))

    return image


@app.post("/predict")
async def predict(file: UploadFile):

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(float(np.max(predictions[0])) * 100, 2)
    
    return {'class': predicted_class, 'confidence': confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)