# CreditVision 
CreditVision es una aplicación de **evaluación de riesgo crediticio** desarrollada con **Streamlit** que utiliza una **Red Neuronal Artificial (ANN)** combinada con **reducción de dimensionalidad mediante PCA** para clasificar el perfil financiero de un cliente en tres categorías de riesgo.

La aplicación permite ingresar variables financieras reales del cliente y obtener una **predicción del Credit Score en tiempo real**.

---

## Objetivo del proyecto

El objetivo del proyecto es construir un modelo de **Machine Learning supervisado** capaz de predecir la calidad del historial crediticio de un cliente a partir de variables financieras y comportamentales.

El sistema clasifica el perfil en tres categorías:

- **Bad** — Alto riesgo crediticio  
- **Standard** — Riesgo medio  
- **Good** — Bajo riesgo crediticio  

---

## Tecnologías utilizadas

El proyecto fue desarrollado utilizando:

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- PCA (Principal Component Analysis)

---

## Metodología del modelo

El flujo completo del modelo es el siguiente:

1. Limpieza de datos  
2. Codificación de variables categóricas con `LabelEncoder`  
3. Normalización de variables numéricas con `StandardScaler`  
4. Reducción de dimensionalidad mediante `PCA`  
5. Entrenamiento de una Red Neuronal Artificial (ANN)  
6. Predicción de la categoría de riesgo crediticio  

---

## Resultados del modelo

Evaluación en el conjunto de prueba:


## Reporte de clasificación

| Clase | Precision | Recall | F1-Score | Support |
|------|-----------|--------|---------|--------|
| Bad (0) | 0.85 | 0.67 | 0.75 | 625 |
| Standard (1) | 0.79 | 0.81 | 0.80 | 916 |
| Good (2) | 0.61 | 0.80 | 0.69 | 334 |

### Métricas globales

| Métrica | Valor |
|--------|------|
| Accuracy | 0.76 |
| Macro Avg Precision | 0.75 |
| Macro Avg Recall | 0.76 |
| Macro Avg F1-Score | 0.75 |
| Weighted Avg Precision | 0.78 |
| Weighted Avg Recall | 0.76 |
| Weighted Avg F1-Score | 0.76 |


El modelo muestra buen desempeño para identificar perfiles **Standard** y **Bad**, y logra detectar clientes **Good** con buen recall.

---

## Aplicación Streamlit

La aplicación permite ingresar información financiera del cliente como:

- edad
- ingresos
- cuentas bancarias
- tarjetas de crédito
- deuda pendiente
- utilización de crédito
- comportamiento de pago
- historial crediticio
- tipos de préstamo

A partir de estos datos, la aplicación genera:

- clasificación del riesgo
- nivel de confianza del modelo
- probabilidad para cada categoría

---

## Estructura del repositorio


credit-score-app/


app.py

requirements.txt


modelo_ann.keras

scaler_ann.pkl

pca_ann.pkl


notebooks/

└── entrenamiento_ann_credit_score.ipynb


Descripción:

| Archivo | Descripción |
|------|------|
| app.py | Aplicación Streamlit |
| modelo_ann.keras | Red neuronal entrenada |
| scaler_ann.pkl | Normalización de variables |
| pca_ann.pkl | Reducción de dimensionalidad |
| requirements.txt | Dependencias del proyecto |
| notebooks/ | Notebook de entrenamiento del modelo |

---

## Ejecutar el proyecto localmente

Clonar el repositorio:

```bash
git clone https://github.com/leydymf/credit-score-app.git
cd credit-score-app
```

Crear entorno virtual:

```bash
python -m venv venv
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

La aplicación se abrirá en:

```bash
http://localhost:8501
```

## Dataset

El dataset utilizado contiene información financiera de clientes como:

* ingresos

* deuda

* comportamiento de pago

* historial crediticio

* utilización de crédito

Fuente del dataset:

```bash
https://github.com/adiacla/bigdata/raw/master/riesgo.xlsx
```

## Autor

Proyecto desarrollado por:

Leydy Yohana Macareo Fuentes

Ingeniería de Sistemas
Ciencia de Datos

## Licencia

Este proyecto fue desarrollado con fines académicos.
