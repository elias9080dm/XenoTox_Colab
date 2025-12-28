# Cleaning function 
import numpy as np
def clean_numeric(X, clip_value=1e4):
    # Reemplazar NaN, inf, -inf
    X = np.nan_to_num(X, nan=0.0, posinf=clip_value, neginf=-clip_value)
    # Clip valores extremos
    X = np.clip(X, -clip_value, clip_value)
    return X