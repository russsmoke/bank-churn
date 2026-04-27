"""
Модуль для загрузки всех артефактов модели
"""

import joblib
from pathlib import Path
import logging

from main.src.data.preprocess import DataPreprocessor

logger = logging.getLogger(__name__)


class ModelArtifacts:
    """
    Класс-контейнер для всех артефактов модели
    """

    def __init__(self, artifacts_path="artifacts"):
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        self.artifacts_path = BASE_DIR / "artifacts"

        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.shap_explainer = None

    def load_all(self):
        """
        Загружает все артефакты
        """

        logger.info("Загрузка артефактов...")

        # --- 1. Модель ---
        model_path = self.artifacts_path / "models" / "xgb_final_v1.pkl"
        self.model = joblib.load(model_path)
        logger.info(f"Модель загружена: {model_path}")

        # --- 2. Препроцессор ---
        preprocessor_path = self.artifacts_path / "preprocessor.pkl"

        self.preprocessor = DataPreprocessor()
        self.preprocessor.load(preprocessor_path)

        logger.info(f"Препроцессор загружен: {preprocessor_path}")

        # --- 3. Feature names (если используешь отдельно) ---
        feature_names_path = self.artifacts_path / "feature_names_v1.pkl"

        if feature_names_path.exists():
            self.feature_names = joblib.load(feature_names_path)
            logger.info("Feature names загружены")

        # --- 4. SHAP explainer ---
        shap_path = self.artifacts_path / "shap_explainer_v1.pkl"

        if shap_path.exists():
            self.shap_explainer = joblib.load(shap_path)
            logger.info("SHAP explainer загружен")

        logger.info("Все артефакты успешно загружены")

        return self


def load_artifacts():
    """
    Удобная функция для быстрого использования в сервисе
    """
    artifacts = ModelArtifacts().load_all()

    return (
        artifacts.model,
        artifacts.preprocessor,
        artifacts.feature_names,
        artifacts.shap_explainer
    )