from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictRequest
from model_loader import load_artifacts, preprocess_input

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

artifacts = load_artifacts()

@app.post("/predict")
def predict(request: PredictRequest):
    input_df = preprocess_input(request.dict(), artifacts)
    model = artifacts["model"]
    features = artifacts["features"]
    for c in features:
        if c not in input_df.columns:
            input_df[c] = 0
    input_df = input_df[features]
    pred = float(model.predict(input_df)[0])
    return {"prediction": round(pred, 2)}
