[README (1).md](https://github.com/user-attachments/files/34611489/README.1.md)
# Predicción del precio de autos usados en la India

Proyecto final de **Análisis Predictivo** (Ing. Carlos Carrascal). Se predice el precio de venta de autos usados con modelos de regresión tradicionales y una red neuronal, y se comparan sus resultados.

- **Notebook:** [`tercerParcialAnalisis.ipynb`](./tercerParcialAnalisis.ipynb)
- **Video de sustentación:** https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1WQgk84A5Yu7TsWm2N6K5AZkStTA2JLny%2Fview%3Fusp%3Dsharing
- **Dataset:** https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=Car+details+v3.csv

## Problema

El mercado de autos usados en la India es grande y poco transparente, y muchos vendedores fijan el precio por intuición. Un modelo que estime el precio ayuda a vendedores (publicar un precio realista), compradores (juzgar si un anuncio es razonable) y a la plataforma (detectar precios raros).

- **Tipo de problema:** regresión
- **Variable objetivo:** `selling_price` (rupias)
- **Datos:** 8.128 registros y 13 columnas; 6.924 anuncios después de la limpieza

## Qué se hizo

| Bloque | Contenido |
| --- | --- |
| 1. Inspección | Carga, diccionario de datos, nulos, duplicados y la función reutilizable `informe_calidad` |
| 2. Limpieza e ingeniería | Se eliminaron 1.202 duplicados y 2 valores imposibles de `km_driven`; conversión de texto a número (`mileage`, `engine`, `max_power`, `torque`); imputación comparada (media, mediana, KNN); variables nuevas `age`, `km_per_year`, `power_per_cc` y `brand`; `ColumnTransformer`; demostración de fuga de datos |
| 3. Exploración | Distribución del precio, cajas por categoría, mapa de calor, pruebas t de Welch y Mann-Whitney |
| 4. Modelos tradicionales | Regresión lineal, árbol de decisión, Random Forest y Gradient Boosting; validación cruzada de 5 pliegues y `GridSearchCV` |
| 5. Red neuronal | Keras (64-32-1, ReLU, MSE, Adam, `EarlyStopping`) y dos experimentos de mejora |
| 6. Conclusiones | Informe ejecutivo, limitaciones y consideraciones éticas |

El preprocesamiento que aprende de los datos (imputar, escalar, codificar) va dentro de un `Pipeline` y se ajusta solo con el conjunto de entrenamiento, para evitar fuga de datos. El modelo se entrena con el logaritmo del precio (`TransformedTargetRegressor`) y se evalúa en rupias.

## Resultados

Partición 80 % / 20 % con semilla 42 (5.539 autos para entrenar y 1.385 para probar).

| Modelo | R² (prueba) | MAE | RMSE |
| --- | --- | --- | --- |
| **Gradient Boosting (optimizado)** | **0,926** | **74.700** | **153.346** |
| Red neuronal 128-64-32 | 0,903 | – | 175.704 |
| Red neuronal base | 0,880 | – | – |
| Red neuronal L2 + Dropout | 0,862 | – | – |

Por validación cruzada, el MAE fue 77.740 para Gradient Boosting, 79.778 para Random Forest y 93.736 para la regresión lineal. En la prueba, el MAPE de Gradient Boosting fue 16,4 %.

**Conclusión:** Gradient Boosting fue el mejor modelo, y ninguna red neuronal lo superó. En datos tabulares de unas siete mil filas, la red no justifica su complejidad. Las variables más importantes fueron la potencia y la antigüedad, seguidas de la marca.

## Limitaciones

- Datos de un solo sitio y hasta 2020.
- Pocos autos a gas (94) y pocos de lujo.
- El 3 % de los anuncios no tiene ficha técnica.
- El error promedio es de unas 74.700 rupias sobre un precio mediano de 400.000: sirve como referencia de precio de salida, no como tasación.

## Cómo ejecutarlo

1. Abre el notebook en Google Colab.
2. Descarga el dataset desde el enlace de arriba y súbelo con el botón de carga de la primera celda de datos.
3. Ejecuta las celdas en orden. La semilla está fija en `SEED = 42`.

**Librerías:** Pandas, NumPy, Matplotlib, Seaborn, scikit-learn y TensorFlow/Keras.

## Uso de IA

Se usó Claude como apoyo, como se declara al final del notebook.

## Autoría

Lizeth Juliana Luna Rincón

> Ejercicio académico: el modelo no debe usarse para tomar decisiones reales de precio sin validación adicional.
