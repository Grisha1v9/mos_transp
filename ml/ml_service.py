"""
ML-микросервис для Java
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import json
import holidays
from datetime import datetime, timedelta

app = FastAPI(title="ML Forecast Service")

with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts/metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

features_order = meta['features_order']
route_means = {int(k): v for k, v in meta['route_means'].items()}
route_hour_means = meta['route_hour_means']
bias_dict = {int(k): v for k, v in meta['bias_dict'].items()}
temp_map = {int(k): v for k, v in meta['temp_map'].items()}
precip_map = {int(k): v for k, v in meta['precip_map'].items()}
routes = meta['routes']

ru_holidays = holidays.RU(years=[2025])


def is_weekend(date):
    return 1 if date.weekday() >= 5 else 0

def is_holiday(date):
    return 1 if date.date() in ru_holidays else 0

def is_major_event(date):
    major_events = ['2025-01-01','2025-01-02','2025-01-07','2025-02-23','2025-03-08',
                    '2025-05-01','2025-05-09','2025-06-12','2025-09-13','2025-11-04','2025-12-31']
    return 1 if date.strftime('%Y-%m-%d') in major_events else 0

def is_school_break(date):
    date_str = date.strftime('%Y-%m-%d')
    ranges = [
        ('2025-10-04', '2025-10-12'),
        ('2025-10-25', '2025-11-02'),
        ('2025-11-15', '2025-11-23'),
        ('2025-12-31', '2025-12-31'),
    ]
    for start, end in ranges:
        if start <= date_str <= end:
            return 1
    return 0

def get_traffic_index(hour, is_we, is_hol):
    if is_we or is_hol:
        return 4.0 if 12 <= hour <= 19 else 2.0
    else:
        if 0 <= hour <= 6:   return 1.0
        if 7 <= hour <= 11:  return 8.0
        if 12 <= hour <= 15: return 5.0
        if hour == 16:       return 5.0
        if 17 <= hour <= 20: return 8.0
        if hour == 21:       return 4.0
        if 22 <= hour <= 23: return 2.0
        return 1.0


class ForecastRequest(BaseModel):
    route: int
    date_from: str
    date_to: str
    hour_from: int = 0
    hour_to: int = 23
    weather_coef: float = 1.0
    event_coef: float = 1.0
    season_coef: float = 1.0


def prepare_features(route: int, date_str: str, hour: int) -> List[float]:
    date = datetime.strptime(date_str, '%Y-%m-%d')
    dayofweek = date.weekday()
    month = date.month
    is_we = is_weekend(date)
    is_hol = is_holiday(date)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7)

    temp = temp_map[month]
    precip = precip_map[month]
    is_cold = 1 if temp <= -5 else 0
    is_snow = 1 if (precip > 0 and temp < 0) else 0

    route_mean = route_means.get(route, 0)
    route_hour_mean = route_hour_means.get(f"{route}|{hour}", 0)

    feature_dict = {
        'route': route,
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'is_weekend': is_we,
        'is_holiday': is_hol,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'temp': temp,
        'precipitation': precip,
        'route_mean_boardings': route_mean,
        'route_hour_mean': route_hour_mean,
        'is_cold': is_cold,
        'is_snow': is_snow,
        'is_major_event': is_major_event(date),
        'traffic_index': get_traffic_index(hour, is_we, is_hol),
        'is_school_break': is_school_break(date),
    }

    return [feature_dict[f] for f in features_order]


def apply_bias(prediction: float, dayofweek: int, hour: int) -> float:
    if hour <= 6:
        block = 'night'
    elif hour <= 11:
        block = 'morning'
    elif hour <= 16:
        block = 'day'
    else:
        block = 'evening'
    coef = bias_dict.get(dayofweek, {}).get(block, 1.0)
    return prediction * coef


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "n_features": len(features_order)}

@app.get("/debug")
def debug():
    test_feat = prepare_features(1, '2025-11-01', 8)
    raw = float(model.predict(np.array([test_feat], dtype=np.float64))[0])
    day = 5  # суббота
    hour = 8
    block = 'morning'
    bias = bias_dict.get(day, {}).get(block, 1.0)
    return {
        "features_sent": test_feat,
        "features_order": features_order,
        "raw_prediction": raw,
        "bias_applied": bias,
        "final": raw * bias,
        "n_features_expected": len(features_order),
        "n_features_sent": len(test_feat),
    }

@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    if req.route not in routes:
        raise HTTPException(400, f"Route {req.route} not supported")

    results = []
    date_from = datetime.strptime(req.date_from, '%Y-%m-%d')
    date_to = datetime.strptime(req.date_to, '%Y-%m-%d')

    current = date_from
    while current <= date_to:
        date_str = current.strftime('%Y-%m-%d')
        for hour in range(req.hour_from, req.hour_to + 1):
            features = np.array([prepare_features(req.route, date_str, hour)], dtype=np.float64)

            base_pred = max(0, float(model.predict(features)[0]))
            final_pred = apply_bias(base_pred, current.weekday(), hour)
            final_pred *= req.weather_coef * req.event_coef * req.season_coef

            if req.route == 5:
                final_pred = 0

            results.append({
                'route': req.route,
                'date': date_str,
                'hour': hour,
                'prediction': int(round(final_pred))
            })

        current += timedelta(days=1)

    return {'route': req.route, 'data': results}