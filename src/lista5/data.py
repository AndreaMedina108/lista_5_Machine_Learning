from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# Encuentra la raíz del proyecto automáticamente desde la ubicación de data.py
RUTA_CSV = Path(__file__).resolve().parents[2] / "datos" / "Datos_Hitters.csv"

def cargar_hitters(ruta=RUTA_CSV):
    df = pd.read_csv(ruta)
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    return df


def particionar_datos(df, target="Salary", random_state=31):
    """
    Realiza la partición en dos etapas con random_state=31:
    1. 120 jugadores para desarrollo y 143 para prueba.
    2. Los 120 de desarrollo se dividen en 50 para entrenamiento y 70 para validación.
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Etapa 1: 120 desarrollo, 143 prueba
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, train_size=120, test_size=143, random_state=random_state
    )

    # Etapa 2: 50 entrenamiento, 70 validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, train_size=50, test_size=70, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test