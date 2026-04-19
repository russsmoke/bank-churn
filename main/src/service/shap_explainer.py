"""
Модуль для SHAP интерпретации предсказаний модели

Используется в сервисе для объяснения причин оттока клиентов.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Определяет корневую директорию проекта.
    
    Ищет маркеры проекта:
    - папка artifacts на том же уровне
    - папка data на том же уровне
    - файл requirements.txt
    """
    current = Path(__file__).resolve()  # src/shap_explainer.py
    src_dir = current.parent.parent # src/
    
    # Корень проекта - это родитель src/
    project_root = src_dir.parent
    
    # Проверяем, что это действительно корень
    if (project_root / "artifacts").exists() or (project_root / "data").exists():
        return project_root
    
    # Если нет, поднимаемся выше
    for parent in src_dir.parents:
        if (parent / "artifacts").exists() or (parent / "data").exists():
            return parent
        if (parent / "requirements.txt").exists():
            return parent
    
    # Если не нашли, возвращаем родителя src
    return src_dir.parent


def get_artifacts_path() -> Path:
    """Возвращает путь к папке artifacts в корне проекта."""
    return get_project_root() / "artifacts"


def get_data_path() -> Path:
    """Возвращает путь к папке data в корне проекта."""
    return get_project_root() / "data"


# Определяем пути
PROJECT_ROOT = get_project_root()
ARTIFACTS_PATH = get_artifacts_path()
DATA_PATH = get_data_path()

logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"ARTIFACTS_PATH: {ARTIFACTS_PATH}")
logger.info(f"DATA_PATH: {DATA_PATH}")


class ShapExplainer:
    """
    Класс для SHAP интерпретации предсказаний модели.
    """
    
    def __init__(self):
        """Инициализация SHAP объяснителя."""
        self.explainer = None
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.expected_value = None
        self.optimal_threshold = 0.5
        self._loaded = False
        
        logger.info(f"ShapExplainer инициализирован")
    
    def load(self) -> 'ShapExplainer':
        """
        Загружает все необходимые артефакты.
        
        Returns:
        --------
        ShapExplainer
            self для цепочечных вызовов
        """
        try:
            # 1. Загружаем SHAP explainer
            explainer_path = ARTIFACTS_PATH / "shap_explainer.pkl"
            if explainer_path.exists():
                self.explainer = joblib.load(explainer_path)
                logger.info(f"✅ SHAP explainer загружен из {explainer_path}")
            else:
                logger.warning(f"SHAP explainer не найден в {explainer_path}")
                self.explainer = None
            
            # 2. Загружаем модель
            model_path = ARTIFACTS_PATH / "models" / "xgb_final_v1.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"✅ Модель загружена из {model_path}")
            else:
                raise FileNotFoundError(f"Модель не найдена: {model_path}")
            
            # 3. Загружаем препроцессор
            preprocessor_path = ARTIFACTS_PATH / "preprocessor.pkl"
            if preprocessor_path.exists():
                # Добавляем путь к src для импорта
                src_path = PROJECT_ROOT / "src"
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                
                from data.preprocess import DataPreprocessor
                
                self.preprocessor = DataPreprocessor()
                self.preprocessor.load(str(preprocessor_path))
                logger.info(f"✅ Препроцессор загружен из {preprocessor_path}")
            else:
                raise FileNotFoundError(f"Препроцессор не найден: {preprocessor_path}")
            
            # 4. Загружаем feature names
            self.feature_names = self.preprocessor.get_feature_columns()
            
            # 5. Загружаем оптимальный порог
            threshold_path = ARTIFACTS_PATH / "threshold_config.yaml"
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.optimal_threshold = config.get('optimal_threshold', 0.5)
                logger.info(f"✅ Оптимальный порог: {self.optimal_threshold}")
            else:
                logger.warning(f"threshold_config.yaml не найден, используем порог 0.5")
            
            self._loaded = True
            logger.info("✅ ShapExplainer полностью загружен и готов к работе")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке артефактов: {e}")
            raise
        
        return self
    
    def _preprocess_customer(self, customer_data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """Предобработка данных клиента."""
        if not self._loaded:
            raise RuntimeError("ShapExplainer не загружен. Вызовите .load() сначала.")
        
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()
        
        X = self.preprocessor.preprocess_for_inference(df)
        return X
    
    def _get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Получение SHAP values для одного или нескольких клиентов."""
        if self.explainer is None:
            raise RuntimeError("SHAP explainer не загружен")
        
        shap_values = self.explainer.shap_values(X)
        
        # Для XGBoost shap_values может быть списком [class0, class1]
        if isinstance(shap_values, list):
            return shap_values[1]  # Берем класс "отток"
        return shap_values
    
    def predict(self, customer_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """Предсказание для одного клиента (без SHAP объяснения)."""
        if not self._loaded:
            raise RuntimeError("ShapExplainer не загружен. Вызовите .load() сначала.")
        
        X = self._preprocess_customer(customer_data)
        proba = self.model.predict_proba(X)[0, 1]
        prediction = 1 if proba >= self.optimal_threshold else 0
        
        return {
            'churn_probability': float(proba),
            'is_risky': prediction == 1,
            'prediction': int(prediction)
        }
    
    def explain(self, customer_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """Предсказание с SHAP объяснением для одного клиента."""
        if not self._loaded:
            raise RuntimeError("ShapExplainer не загружен. Вызовите .load() сначала.")
        
        # Предобработка
        X = self._preprocess_customer(customer_data)
        
        # Предсказание
        proba = self.model.predict_proba(X)[0, 1]
        prediction = 1 if proba >= self.optimal_threshold else 0
        
        # SHAP values (если explainer доступен)
        shap_values_client = None
        top_factors = []
        explanation_text = "SHAP explainer не доступен"
        
        if self.explainer is not None:
            try:
                shap_values_client = self._get_shap_values(X)[0]
                
                # Анализируем факторы
                factors = []
                for i, (feature, shap_val) in enumerate(zip(self.feature_names, shap_values_client)):
                    if abs(shap_val) > 0.005:
                        direction = "увеличивает" if shap_val > 0 else "уменьшает"
                        factors.append({
                            'feature': feature,
                            'feature_ru': self._get_feature_name_ru(feature),
                            'impact': float(shap_val),
                            'direction': direction,
                            'abs_impact': abs(float(shap_val))
                        })
                
                # Сортируем по важности
                factors.sort(key=lambda x: x['abs_impact'], reverse=True)
                top_factors = factors[:4]
                
                # Формируем человеко-читаемое объяснение
                reasons = []
                for f in top_factors:
                    reason = f"{f['feature_ru']} {f['direction']} риск оттока"
                    reasons.append(reason)
                
                explanation_text = f"Основные факторы: {', '.join(reasons)}" if reasons else "Факторы риска не выявлены"
                
            except Exception as e:
                logger.warning(f"Ошибка при расчёте SHAP: {e}")
                explanation_text = "Ошибка при анализе факторов риска"
        
        # Формируем ответ
        result = {
            'churn_probability': float(proba),
            'is_risky': prediction == 1,
            'prediction': int(prediction),
            'explanation': explanation_text,
            'top_factors': top_factors
        }
        
        # Добавляем raw SHAP values для отладки (опционально)
        if shap_values_client is not None:
            result['shap_values'] = shap_values_client.tolist()
        
        return result
    
    def explain_batch(self, customers_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Пакетное предсказание с SHAP объяснением."""
        results = []
        for idx, row in customers_df.iterrows():
            try:
                result = self.explain(row.to_dict())
                result['row_id'] = idx
                results.append(result)
            except Exception as e:
                logger.error(f"Ошибка для клиента {idx}: {e}")
                results.append({
                    'row_id': idx,
                    'error': str(e),
                    'churn_probability': None,
                    'is_risky': None,
                    'prediction': None,
                    'explanation': "Ошибка при обработке"
                })
        return results
    
    def _get_feature_name_ru(self, feature: str) -> str:
        """Возвращает русское название признака для объяснений."""
        names = {
            'CreditScore': 'кредитный рейтинг',
            'Age': 'возраст',
            'Tenure': 'стаж',
            'Balance': 'баланс',
            'NumOfProducts': 'количество продуктов',
            'IsActiveMember': 'активность',
            'Age_Group': 'возрастная группа',
            'Tenure_Ratio': 'лояльность (стаж/возраст)',
            'Balance_Per_Product': 'баланс на продукт',
            'Active_Senior': 'активный пожилой',
            'Has_Balance': 'наличие баланса',
            'Is_Multi_Product': 'мульти-продукт',
            'Gender_Male': 'пол (мужской)'
        }
        return names.get(feature, feature)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Возвращает важность признаков по SHAP."""
        importance_path = ARTIFACTS_PATH / "shap_feature_importance.csv"
        if importance_path.exists():
            return pd.read_csv(importance_path)
        else:
            logger.warning("SHAP feature importance не найден")
            return pd.DataFrame()


# Глобальный экземпляр для использования в сервисе
_default_explainer = None


def get_shap_explainer() -> ShapExplainer:
    """
    Возвращает глобальный экземпляр ShapExplainer (синглтон).
    """
    global _default_explainer
    if _default_explainer is None:
        _default_explainer = ShapExplainer()
        _default_explainer.load()
    return _default_explainer


def explain_customer(customer_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
    """Быстрая функция для объяснения предсказания."""
    explainer = get_shap_explainer()
    return explainer.explain(customer_data)


# Для тестирования модуля
if __name__ == "__main__":
    print("="*60)
    print("ТЕСТИРОВАНИЕ SHAP EXPLAINER")
    print("="*60)
    
    print(f"\nPROJECT_ROOT: {PROJECT_ROOT}")
    print(f"ARTIFACTS_PATH: {ARTIFACTS_PATH}")
    print(f"DATA_PATH: {DATA_PATH}")
    
    # Проверяем наличие артефактов
    print("\nПроверка артефактов:")
    artifacts_to_check = [
        "preprocessor.pkl",
        "model.pkl",
        "shap_explainer.pkl",
        "threshold_config.yaml"
    ]
    
    for artifact in artifacts_to_check:
        path = ARTIFACTS_PATH / artifact
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {artifact}: {path}")
    
    # Пытаемся загрузить тестовые данные
    test_data_path = DATA_PATH / "raw" / "Churn_Modelling.csv"
    if test_data_path.exists():
        test_df = pd.read_csv(test_data_path)
        test_customer = test_df.iloc[0].to_dict()
        
        print(f"\nТестовый клиент (первые 5 полей):")
        for k, v in list(test_customer.items())[:5]:
            print(f"  - {k}: {v}")
        print("  ...")
        
        # Создаём и загружаем explainer
        explainer = ShapExplainer()
        
        try:
            explainer.load()
            
            # Получаем объяснение
            result = explainer.explain(test_customer)
            
            print(f"\nРезультат:")
            print(f"  - Вероятность оттока: {result['churn_probability']:.2%}")
            print(f"  - В зоне риска: {'ДА' if result['is_risky'] else 'НЕТ'}")
            print(f"  - Объяснение: {result['explanation']}")
            
            print(f"\nТоп-3 фактора:")
            for f in result['top_factors'][:3]:
                print(f"  - {f['feature_ru']}: {f['direction']} (impact: {f['impact']:.4f})")
            
            print("\n✅ Тест пройден успешно!")
            
        except Exception as e:
            print(f"\n❌ Ошибка при тестировании: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n❌ Тестовые данные не найдены: {test_data_path}")
        print("Пожалуйста, убедитесь, что файл data/raw/Churn_Modelling.csv существует")