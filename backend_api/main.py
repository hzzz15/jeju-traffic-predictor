import sys
import os
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictRequest
from .model_loader import load_artifacts, preprocess_input
import traceback
from fastapi.responses import JSONResponse

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 모델, 인코더, feature list
artifacts = load_artifacts()

from fastapi.responses import JSONResponse

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        input_data = request.dict()
        print("[DEBUG] 요청 받은 데이터:", input_data)

        input_df = preprocess_input(input_data, artifacts)
        print("[DEBUG] 전처리된 데이터:", input_df)

        model = artifacts["model"]
        features = artifacts["features"]

        for c in features:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[features]

        input_df = input_df[features]
        print("[DEBUG] 모델 입력 최종 데이터:", input_df)

        pred = float(model.predict(input_df)[0])
        print("[DEBUG] 예측 결과:", pred)
        return {"prediction": round(pred, 2)}

    except Exception as e:
        print("[ERROR] 예측 중 오류 발생:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "FastAPI backend is running!"}
