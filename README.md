# Reconocimiento de Emociones con CNN Ligera

Este proyecto implementa una red neuronal convolucional ligera (Light CNN) para reconocer emociones faciales a partir de imágenes en escala de grises utilizando el conjunto de datos FER-2013.

## Descripción

- Las imágenes están representadas como arrays de 48x48 píxeles en escala de grises.
- Se convierten los datos de texto a arrays NumPy y se normalizan los valores de píxeles.
- Las etiquetas se codifican en formato one-hot para clasificación multiclase (7 emociones).
- El dataset se divide en conjuntos de entrenamiento, validación y prueba según el campo `Usage`.

## Arquitectura del Modelo

La red CNN está compuesta por:

- Tres capas convolucionales con activación ReLU, normalización por lotes y max pooling.
- Global Average Pooling seguida de una capa Dropout.
- Capa densa de salida con activación softmax para clasificar las 7 emociones.

## Entrenamiento

- Optimización con `Adam` y función de pérdida `categorical_crossentropy`.
- Entrenamiento por 50 épocas con batch size de 128.
- Uso de callbacks:
  - `EarlyStopping` para evitar sobreajuste.
  - `ModelCheckpoint` para guardar el mejor modelo como `best_fer_lightcnn.h5`.

## Requisitos

Instalación de dependencias necesarias:

```bash
pip install numpy pandas tensorflow scikit-learn
