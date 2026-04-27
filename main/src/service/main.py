from fastapi import FastAPI, UploadFile
import pandas as pd
import logging

from main.src.models.load_artifacts import load_artifacts
from main.src.models.predict import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bank Churn Prediction API")

# 🔥 Загружаем всё один раз при старте
model, preprocessor, feature_names, shap_explainer = load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


# --- 1. Предсказание для одного клиента ---
@app.post("/predict")
def predict_one(data: dict):
    try:
        df = pd.DataFrame([data])

        X = preprocessor.preprocess_for_inference(df)
        result = predict(model, X)

        prob = float(result["probabilities"][0])
        label = int(result["labels"][0])

        return {
            "churn_probability": prob,
            "churn_flag": label
        }

    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        return {"error": str(e)}


# --- 2. Batch предсказание (CSV) ---
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import io


@app.post("/predict_batch")
async def predict_batch(file: UploadFile):
    # 1. читаем CSV
    df = pd.read_csv(file.file)

    # 2. предобработка
    X = preprocessor.preprocess_for_inference(df)

    # 3. предсказание
    result = predict(model, X)

    # 4. добавляем колонки
    df["churn_probability"] = result["probabilities"]
    df["churn_flag"] = result["labels"]

    # 5. создаём CSV в памяти
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    filename = f"predictions_{len(df)}_rows.csv"
    # 6. возвращаем файл
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )