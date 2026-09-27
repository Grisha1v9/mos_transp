from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import httpx
import pandas as pd
from io import BytesIO
import os

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8000/api/forecast")

app = FastAPI(title="Backend API Gateway")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_dashboard():
    return FileResponse("dashboard.html")

from typing import Optional

class ForecastRequest(BaseModel):
    route: int
    date_from: str
    date_to: str
    hour_from: int = 0
    hour_to: int = 23
    horizon: str = "DAY"
    stop_id: Optional[int] = None
    weather_coef: float = 1.0
    event_coef: float = 1.0
    season_coef: float = 1.0



@app.get("/api/routes")
async def get_routes():
    """Возвращает список маршрутов (заглушка, так как ML-сервис не имеет этого эндпоинта)"""
    return {"routes": [1, 7, 11, 12, 17, 25, 26, 28, 50]}

@app.post("/api/forecast")
async def proxy_forecast(req: ForecastRequest):
    async with httpx.AsyncClient() as client:
        ml_payload = {
            "route": req.route,
            "date_from": req.date_from,
            "date_to": req.date_to,
            "hour_from": req.hour_from,
            "hour_to": req.hour_to,
            "weather_coef": req.weather_coef,
            "event_coef": req.event_coef,
            "season_coef": req.season_coef,
        }
        try:
            response = await client.post(ML_SERVICE_URL, json=ml_payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            # Учет горизонта
            if req.horizon == "MONTH":
                daily = {}
                for row in data["data"]:
                    daily.setdefault(row["date"], 0)
                    daily[row["date"]] += row["prediction"]
                data["data"] = [
                    {"route": req.route, "date": d, "hour": 0, "prediction": v}
                    for d, v in sorted(daily.items())
                ]
            elif req.horizon == "YEAR":
                monthly = {}
                for row in data["data"]:
                    m = row["date"][:7]
                    monthly.setdefault(m, [])
                    monthly[m].append(row["prediction"])
                data["data"] = [
                    {"route": req.route, "date": m + "-01", "hour": 0,
                     "prediction": int(sum(v) / len(v))}
                    for m, v in sorted(monthly.items())
                ]

            # Учет остановки (пропорциональное распределение)
            if req.stop_id is not None:
                share = 1.0 / 10.0
                for row in data["data"]:
                    row["prediction"] = int(round(row["prediction"] * share))
                    row["stop_id"] = req.stop_id

            return data
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"ML Service Error: {e.response.text}")
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="ML Service is unavailable")
@app.post("/api/export/csv")
async def export_csv(req: ForecastRequest):
    """Запрашивает прогноз у ML и отдает CSV файл"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(ML_SERVICE_URL, json=req.dict(), timeout=30.0)
            response.raise_for_status()
            data = response.json()['data']
            
            df = pd.DataFrame(data)
            stream = BytesIO()
            df.to_csv(stream, index=False, encoding='utf-8')
            stream.seek(0)
            
            return StreamingResponse(
                iter([stream.getvalue()]), 
                media_type="text/csv", 
                headers={"Content-Disposition": f"attachment; filename=forecast_route_{req.route}.csv"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export/xlsx")
async def export_xlsx(req: ForecastRequest):
    """Запрашивает прогноз у ML и отдает Excel файл"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(ML_SERVICE_URL, json=req.dict(), timeout=30.0)
            response.raise_for_status()
            data = response.json()['data']
            
            df = pd.DataFrame(data)
            stream = BytesIO()
            with pd.ExcelWriter(stream, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Forecast')
            stream.seek(0)
            
            return StreamingResponse(
                iter([stream.getvalue()]), 
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename=forecast_route_{req.route}.xlsx"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
