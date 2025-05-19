import requests

def predict_speed_api(user_input: dict) -> float:
    API_URL = "http://localhost:8000/predict"  # 배포 시 URL 수정
    try:
        response = requests.post(API_URL, json=user_input)
        response.raise_for_status()
        return response.json()["prediction"]
    except requests.RequestException as e:
        raise RuntimeError(f"API 요청 실패: {e}")
