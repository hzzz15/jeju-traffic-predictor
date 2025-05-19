from pydantic import BaseModel

class PredictRequest(BaseModel):
    road_name: str
    weekday: str
    time_type: str
    season: str
    adjacent_august: str
