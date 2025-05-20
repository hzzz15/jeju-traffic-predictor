import requests

def predict_speed_api(user_input: dict) -> float:
    #API_URL = "https://jeju-traffic-predictor.onrender.com/predict"
    API_URL = "http://127.0.0.1:8000/predict"
    try:
        response = requests.post(API_URL, json=user_input)
        response.raise_for_status()
        return response.json()["prediction"]
    except requests.RequestException as e:
        raise RuntimeError(f"API 요청 실패: {e}")
