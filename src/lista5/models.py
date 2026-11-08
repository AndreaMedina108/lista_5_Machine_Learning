import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from .metrics import riesgo_cuadratico


def ajustar_regresion(X_train, y_train, variables=None):
    """
    Ajusta un modelo de regresión lineal con intercepto sobre las variables indicadas.
    """
    if variables is not None:
        X_tr = X_train[list(variables)]
    else:
        X_tr = X_train

    modelo = LinearRegression(fit_intercept=True)
    modelo.fit(X_tr, y_train)
    return modelo


def evaluar_modelo(modelo, variables, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """
    Calcula el riesgo cuadrático en entrenamiento, validación y (opcionalmente) prueba.
    """
    vars_list = list(variables)
    resultados = {
        "riesgo_train": riesgo_cuadratico(y_train, modelo.predict(X_train[vars_list])),
        "riesgo_val": riesgo_cuadratico(y_val, modelo.predict(X_val[vars_list])),
    }
    if X_test is not None and y_test is not None:
        resultados["riesgo_test"] = riesgo_cuadratico(y_test, modelo.predict(X_test[vars_list]))
    return resultados


def correlaciones_entrenamiento(X_train, y_train):
    """
    Calcula la correlación de Pearson de cada variable con Salary usando solo
    entrenamiento y las ordena de mayor a menor por su valor absoluto.
    """
    filas = []
    for col in X_train.columns:
        r, p_val = stats.pearsonr(X_train[col], y_train)
        filas.append(
            {
                "Variable": col,
                "Correlacion": r,
                "Abs_Correlacion": abs(r),
                "p_valor_corr": p_val,
            }
        )
    df_corr = (
        pd.DataFrame(filas)
        .sort_values(by="Abs_Correlacion", ascending=False)
        .reset_index(drop=True)
    )
    return df_corr


def pruebas_t_regresion_completa(X_train, y_train):
    """
    Ajusta la regresión lineal completa con intercepto en entrenamiento y obtiene
    el estadístico t y el valor p de la prueba t para cada coeficiente predictor.
    """
    X_train_const = sm.add_constant(X_train)
    modelo_ols = sm.OLS(y_train, X_train_const).fit()

    df_t = pd.DataFrame(
        {
            "Variable": X_train.columns,
            "Coeficiente": modelo_ols.params.drop("const").values,
            "Error_Estandar": modelo_ols.bse.drop("const").values,
            "Estadistico_t": modelo_ols.tvalues.drop("const").values,
            "valor_p": modelo_ols.pvalues.drop("const").values,
        }
    ).sort_values(by="valor_p", ascending=True).reset_index(drop=True)

    return df_t


def seleccion_hacia_adelante(X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """
    Realiza selección hacia adelante (Forward Selection) comenzando con 1 variable
    y agregando en cada paso la variable que produzca el menor riesgo de entrenamiento,
    sin retirar las anteriores, hasta usar las 19 variables.
    """
    restantes = list(X_train.columns)
    seleccionadas = []
    historial = []

    for paso in range(1, len(X_train.columns) + 1):
        mejor_var = None
        menor_riesgo_train = float("inf")
        mejor_modelo = None

        for candidata in restantes:
            vars_prueba = seleccionadas + [candidata]
            modelo = ajustar_regresion(X_train, y_train, vars_prueba)
            r_train = riesgo_cuadratico(y_train, modelo.predict(X_train[vars_prueba]))

            if r_train < menor_riesgo_train:
                menor_riesgo_train = r_train
                mejor_var = candidata
                mejor_modelo = modelo

        seleccionadas.append(mejor_var)
        restantes.remove(mejor_var)

        r_val = riesgo_cuadratico(y_val, mejor_modelo.predict(X_val[seleccionadas]))

        fila = {
            "Num_Variables": paso,
            "Variable_Agregada": mejor_var,
            "Riesgo_Train": menor_riesgo_train,
            "Riesgo_Validacion": r_val,
            "Variables_Modelo": seleccionadas.copy(),
        }

        if X_test is not None and y_test is not None:
            fila["Riesgo_Prueba"] = riesgo_cuadratico(
                y_test, mejor_modelo.predict(X_test[seleccionadas])
            )

        historial.append(fila)

    return pd.DataFrame(historial)