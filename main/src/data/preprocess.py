"""
Модуль для предобработки данных банковского скоринга

Используется в:
- ноутбуках для обучения моделей
- сервисе для инференса
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Класс для предобработки данных клиентов банка.
    
    Выполняет:
    1. Создание новых признаков (Tenure_Ratio, Balance_Per_Product и др.)
    2. Кодирование категориальных признаков
    3. Масштабирование числовых признаков
    4. Обработку пропусков
    """
    
    def __init__(self, config_path=None):
        """
        Инициализация препроцессора.
        
        Parameters:
        -----------
        config_path : str, optional
            Путь к конфигурационному файлу YAML
        """
        # Загружаем конфигурацию
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Инициализируем scaler
        self.scaler = None
        
        # Параметры для Age_Group
        self.age_bins = self.config.get('age_bins')
        self.age_labels = self.config.get('age_labels')
        
        # Колонки, которые нужно удалить
        self.drop_cols = [
            'HasCrCard', 'EstimatedSalary', 'CustomerId', 
            'Gender', 'RowNumber', 'Surname', 'Geography'
        ]
        
        logger.info("DataPreprocessor инициализирован")
    
    def _get_default_config(self):
        """Возвращает конфигурацию по умолчанию"""
        return {
            'age_bins': [17, 30, 40, 50, 60, 100],
            'age_labels': [0, 1, 2, 3, 4],
            'random_state': 42,
            'test_size': 0.2
        }
    
    def create_features(self, df):
        """
        Создаёт новые признаки из сырых данных.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходные данные клиентов
            
        Returns:
        --------
        pd.DataFrame
            Данные с добавленными признаками
        """
        df = df.copy()
        
        # 1. Отношение стажа к возрасту (лояльность)
        df['Tenure_Ratio'] = df['Tenure'] / df['Age']
        
        # 2. Баланс на один продукт
        df['Balance_Per_Product'] = df['Balance'] / (df['NumOfProducts'] + 1)
        
        # 3. Активный пожилой клиент (возраст > 50 и активный)
        df['Active_Senior'] = (
            (df['Age'] > 50) & (df['IsActiveMember'] == 1)
        ).astype(int)
        
        # 4. Наличие ненулевого баланса
        df['Has_Balance'] = (df['Balance'] > 0).astype(int)
        
        # 5. Мульти-продукт (3 и более продуктов)
        df['Is_Multi_Product'] = (df['NumOfProducts'] >= 3).astype(int)
        
        # 6. Кодирование пола (Male = 1, Female = 0)
        if 'Gender' in df.columns:
            df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
        
        # 7. Age Group (категории возраста)
        df['Age_Group'] = pd.cut(
            df['Age'], 
            bins=self.age_bins, 
            labels=self.age_labels
        )
        
        logger.info(f"Создано {len(df.columns)} признаков")
        return df
    
    def clean_data(self, df):
        """
        Очищает данные от мусора.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Данные после create_features
            
        Returns:
        --------
        pd.DataFrame
            Очищенные данные
        """
        df = df.copy()
        
        # Удаляем ненужные колонки
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        df = df.drop(cols_to_drop, axis=1)
        
        # Обработка пропусков (если есть)
        if df.isnull().any().any():
            logger.warning(f"Обнаружены пропуски: {df.isnull().sum().sum()}")
            # Заполняем медианой
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].median())
        
        logger.info(f"Данные очищены. Форма: {df.shape}")
        return df
    
    def get_feature_columns(self):
        """
        Возвращает список признаков для модели.
        
        Returns:
        --------
        list
            Список названий признаков
        """
        return [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'IsActiveMember', 'Age_Group', 'Tenure_Ratio', 'Balance_Per_Product',
            'Active_Senior', 'Has_Balance', 'Is_Multi_Product', 'Gender_Male'
        ]
    
    def fit(self, X, y=None):
        X_scaled, _, _ = self.preprocess(X, fit_scaler=True)
        return self

    def transform(self, X):
        X_scaled, _, _ = self.preprocess(X, fit_scaler=False)
        return X_scaled

    def preprocess(self, df, fit_scaler=False):
        """
        Полный пайплайн предобработки данных.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходные сырые данные
        fit_scaler : bool
            Если True, обучает scaler на этих данных
            
        Returns:
        --------
        tuple
            (X_scaled, y, df_processed)
            - X_scaled: масштабированные признаки (numpy array)
            - y: целевая переменная (None если нет в данных)
            - df_processed: обработанный DataFrame
        """
        # 1. Создаём новые признаки
        df = self.create_features(df)
        
        # 2. Очищаем данные
        df = self.clean_data(df)
        
        # 3. Выделяем целевую переменную
        y = df['Exited'].copy() if 'Exited' in df.columns else None
        
        # 4. Выбираем признаки
        feature_cols = self.get_feature_columns()
        X = df[feature_cols].copy()
        
        # 5. Масштабирование
        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Scaler обучен и применён")
        else:
            if self.scaler is None:
                raise ValueError("Scaler не обучен. Сначала вызовите preprocess с fit_scaler=True")
            X_scaled = self.scaler.transform(X)
            logger.info("Scaler применён")
        
        return X_scaled, y, df
    
    def preprocess_for_inference(self, df):
        """
        Предобработка для одного клиента (инференс).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Данные одного клиента
            
        Returns:
        --------
        np.ndarray
            Масштабированные признаки
        """
        X_scaled, _, _ = self.preprocess(df, fit_scaler=False)
        return X_scaled
    
    def save(self, path="artifacts/preprocessor.pkl"):
        """Сохраняет препроцессор в файл"""
        # Сохраняем только необходимые атрибуты
        to_save = {
            'scaler': self.scaler,
            'age_bins': self.age_bins,
            'age_labels': self.age_labels,
            'config': self.config,
            'drop_cols': self.drop_cols
        }
        joblib.dump(to_save, path)
        logger.info(f"Препроцессор сохранён в {path}")
    
    def load(self, path="artifacts/preprocessor.pkl"):
        """Загружает препроцессор из файла"""
        loaded = joblib.load(path)
        self.scaler = loaded['scaler']
        self.age_bins = loaded['age_bins']
        self.age_labels = loaded['age_labels']
        self.config = loaded['config']
        self.drop_cols = loaded['drop_cols']
        logger.info(f"Препроцессор загружен из {path}")
        return self


def create_preprocessor_from_config(config_path="configs/config.yaml"):
    """Создаёт препроцессор из конфигурационного файла"""
    return DataPreprocessor(config_path)


# Для быстрого тестирования
if __name__ == "__main__":
    # Тестирование модуля
    print("Testing DataPreprocessor...")
    
    # Создаём тестовые данные
    test_df = pd.DataFrame({
        'CreditScore': [600],
        'Age': [35],
        'Tenure': [5],
        'Balance': [50000],
        'NumOfProducts': [2],
        'IsActiveMember': [1],
        'Gender': ['Male'],
        'Exited': [0]
    })
    
    preprocessor = DataPreprocessor()
    X, y, df_processed = preprocessor.preprocess(test_df, fit_scaler=True)
    
    print(f"Input shape: {test_df.shape}")
    print(f"Output shape: {X.shape}")
    print(f"Features: {preprocessor.get_feature_columns()}")
    print("Тест пройден!")