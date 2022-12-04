from typing import Optional
from pydantic import BaseModel


class BikeSharingData(BaseModel):
    FeelsLikeC: Optional[int]
    maxtempC: Optional[int]
    mintempC: Optional[int]
    windspeedKmph: Optional[int]
    cloudcover: Optional[int]
    humidity: Optional[int]
    pressure: Optional[int]
    visibility: Optional[int]
    is_holiday: Optional[int]
    is_weekend: Optional[int]
    year: Optional[int]
    season: Optional[int]
    month: Optional[int]
    hour: Optional[int]
    day: Optional[int]
    week_day: Optional[str]


class BikeSharingResponse(BaseModel):
    prediction: float
