from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import traceback

from .schemas import PixelData
from . import crud
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(data: PixelData):
    try:
        probability, fruit_state = crud.predict(data)

        return {
            "status": 200,
            "message": "Prediction evaluated succesfully!",
            "data": {
                "probability": probability,
                "fruit_state": fruit_state,
            },
        }
    except Exception:
        traceback.print_exc() 
        
        return {
            "status": 500,
            "message": "An error occured, while evaluating model predictions!",
        }