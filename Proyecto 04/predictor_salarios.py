from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ARCHIVO_CSV = "ds_salaries.csv"
CARPETA_SALIDA = Path("resultados")

CARACTERISTICAS = [
    "job_title",
    "experience_level",
    "employment_type",
    "remote_ratio",
    "company_size",
]

OBJETIVO = "salary_in_usd"


def cargar_datos():
    datos = pd.read_csv(ARCHIVO_CSV)
    datos[OBJETIVO] = pd.to_numeric(datos[OBJETIVO], errors="coerce")
    return datos.dropna(subset=CARACTERISTICAS + [OBJETIVO])


def crear_modelo():
    columnas_categoricas = [
        "job_title",
        "experience_level",
        "employment_type",
        "company_size",
    ]

    preparador = ColumnTransformer(
        [
            (
                "categorias",
                OneHotEncoder(handle_unknown="ignore"),
                columnas_categoricas,
            ),
            ("numeros", "passthrough", ["remote_ratio"]),
        ]
    )

    return Pipeline(
        [
            ("preparador", preparador),
            (
                "modelo",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                ),
            ),
        ]
    )


def crear_grafico(valores_reales, predicciones):
    CARPETA_SALIDA.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(valores_reales, predicciones, alpha=0.7, color="royalblue")

    minimo = min(valores_reales.min(), predicciones.min())
    maximo = max(valores_reales.max(), predicciones.max())

    plt.plot([minimo, maximo], [minimo, maximo], color="red")
    plt.title("Salario real vs. salario predicho")
    plt.xlabel("Salario real en USD")
    plt.ylabel("Salario predicho en USD")
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA / "real_vs_predicho.png", dpi=150)
    plt.show()


def main():
    datos = cargar_datos()

    X = datos[CARACTERISTICAS]
    y = datos[OBJETIVO]

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    modelo = crear_modelo()
    modelo.fit(X_entrenamiento, y_entrenamiento)

    predicciones = modelo.predict(X_prueba)

    error_medio = mean_absolute_error(y_prueba, predicciones)
    puntuacion_r2 = r2_score(y_prueba, predicciones)

    print("\n--- EVALUACIÓN DEL MODELO ---")
    print(f"Error promedio: ${error_medio:,.2f}")
    print(f"Puntuación R²: {puntuacion_r2:.2f}")

    CARPETA_SALIDA.mkdir(exist_ok=True)

    comparacion = pd.DataFrame(
        {
            "salario_real": y_prueba,
            "salario_predicho": predicciones,
        }
    )

    comparacion.to_csv(
        CARPETA_SALIDA / "comparacion_predicciones.csv",
        index=False,
    )

    joblib.dump(modelo, CARPETA_SALIDA / "modelo_salarios.joblib")
    crear_grafico(y_prueba, predicciones)

    print("\nModelo guardado en la carpeta 'resultados'.")
    print("Es un proyecto educativo, no una herramienta para decidir sueldos reales.")


if __name__ == "__main__":
    main()