import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("data/pacientes.csv", parse_dates=["fecha_inicio", "fecha_ultima"])
df["tiempo_tratamiento"] = ((df["fecha_ultima"] - df["fecha_inicio"]).dt.days/30)
df = df[df["tiempo_tratamiento"] >= 0]

features = ["edad", "genero", "precio_sesion", "compromiso_bajo", "tipo_terapia", "terapeuta", "motivo", "tiempo_tratamiento"]
target = "abandono"

X = df[features].copy()
y = df[target].copy()


variables_cuantitativas = ["edad", "precio_sesion", "tiempo_tratamiento"]
variables_cualitativas = ["genero", "tipo_terapia", "motivo", "terapeuta"]
variables_binarias = ["compromiso_bajo"]

#Preprocesado

encoder_cuantitativo = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                       ("scaler", StandardScaler())])

encoder_cualitativo = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="desconocido")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

encoder_binario = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(
    transformers=[("cuant", encoder_cuantitativo, variables_cuantitativas),
                  ("cual", encoder_cualitativo, variables_cualitativas),
                  ("bin", encoder_binario, variables_binarias)]
                  )

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced"))
])

#Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

#Entrenar al modelo
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))

