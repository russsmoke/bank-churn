import numpy as np


def predict(model, X, threshold=0.5):
    """
    Делает предсказание модели
    
    Parameters:
    ----------
    model : обученная модель
    X : np.ndarray
    threshold : float
    
    Returns:
    -------
    dict
    """
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= threshold).astype(int)

    return {
        "probabilities": probs,
        "labels": labels
    }