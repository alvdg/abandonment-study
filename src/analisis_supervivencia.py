import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

df = pd.read_csv("data/pacientes.csv", parse_dates=["fecha_inicio", "fecha_ultima"])
df["tiempo_tratamiento"] = (df["fecha_ultima"] - df["fecha_inicio"]).dt.days
df = df[df["tiempo_tratamiento"] >= 0]

#Kaplan Meier
KMF = KaplanMeierFitter()
#duracion -> tiempo del tratamiento
#evento observado -> abandono
KMF.fit(durations = df["tiempo_tratamiento"], event_observed = df["abandono"])
print("Métricas clave:")
print(f"Probabilidad de supervivencia a 100 días: {KMF.survival_function_at_times(30).values[0]:.2%}")
print(f"Probabilidad de supervivencia a 300 días: {KMF.survival_function_at_times(90).values[0]:.2%}")

#Logrank test: comparar curvas de Kaplan Meier por grupos
#print(df["tipo_terapia"].unique())

plt.figure(figsize=(10,5))
KMF.plot_survival_function()
plt.title("Curva de supervivencia de Kaplan Meier: tiempo hasta el abandono")
plt.xlabel("Dias hasta el abandono")
plt.ylabel("Probabilidad de seguir")
plt.grid(True, alpha=0.3)
plt.show()
