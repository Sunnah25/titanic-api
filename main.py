from fastapi import FastAPI
from pydantic  import BaseModel
import joblib
import pandas as pd
import numpy as np
import os


# --- App Setup ---
app = FastAPI(
    title="Titanic Survival Predictor",
    description="Predicts survival probability for Titanic passengers",
    version="1.0.0"
)

# --- Load model once at startup ---
model = joblib.load("titanic_model.pkl")


# --- Define input schema ---
#Pydantic validates every incoming request against this
class PassengerInput(BaseModel):
    Pclass: int     #1, 2, or 3
    Sex_encoded: int    #0=male, 1=female
    Age: float      #passenger age
    Fare: float     #ticket fare
    IsAlone: int    #=traveling alone, 0=with family
    FamilySize: int #total family members
    Title_encoded: int  #Mr=0, Mrs=1, Miss=2, Master=3, Other=4



# --- Endpoints ---

@app.get("/")
def root():
    return {"message": "Titanic Survival Predictor API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "envirnment": os.environ.get("APP_ENV")}

app.post("/predict")
def predict (passenger: PassengerInput):
    #convert input to DataFrame - model expects this format
    input_df = pd.DataFrame([passenger.dict()])

    #Get prediction and probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "survived": bool(prediction),
        "survival_probability": round(float(probability), 4),
        "survival_percentage": f"{probability:.1%}",
        "passenger_details": passenger.dict()
    }

@app.post("/predict/batch")
def predict_batch(passengers: list[PassengerInput]):
    #predictfor multiple passengers at once
    input_df = pd.DataFrame([p.dict() for p in passengers])
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:,1]

    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            "passenger_index": i,
            "survived": bool(pred),
            "survived_probability": round(float(prob), 4)
        })


    return {"predictions": results, "total_passengers": len(results)}