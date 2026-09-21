# Predictor de salarios con Machine Learning

Proyecto educativo de aprendizaje automático que estima salarios en ciencia de datos a partir del puesto, experiencia, tipo de empleo, modalidad remota y tamaño de empresa.

## Herramientas utilizadas

- Python
- pandas
- scikit-learn
- matplotlib
- joblib

## Proceso del modelo

1. Carga y limpieza de datos.
2. Preparación de variables categóricas y numéricas.
3. División entre datos de entrenamiento y prueba.
4. Entrenamiento de un modelo Random Forest.
5. Evaluación mediante error promedio y puntuación R².
6. Comparación entre salarios reales y predichos.

## Cómo ejecutar el proyecto

1. Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

2. Ejecutar el modelo:

```bash
cd "Proyecto 04" && python predictor_salarios.py
```

## Resultados

El programa genera una tabla de comparación y un gráfico llamado `real_vs_predicho.png` dentro de la carpeta `resultados`.

## Nota importante

Este proyecto tiene fines educativos. No debe utilizarse para tomar decisiones reales sobre salarios, contratación o compensación.
