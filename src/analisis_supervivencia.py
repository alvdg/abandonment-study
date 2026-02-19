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

#Curva de Kaplan Meier general

KMF.plot_survival_function()
plt.title("Curva de supervivencia de Kaplan Meier: tiempo hasta el abandono")
plt.xlabel("Dias hasta el abandono")
plt.ylabel("Probabilidad de seguir")
plt.grid(True, alpha=0.3)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#Curva de Kaplan Meier por tipo de terapia
ax1 = axes[0,0]
for terapia in df["tipo_terapia"].unique():
    mask = df["tipo_terapia"] == terapia
    KMF.fit(durations = df.loc[mask, "tiempo_tratamiento"], 
            event_observed = df.loc[mask, "abandono"],
            label = terapia)
    KMF.plot_survival_function(ax=ax1)
ax1.set_title('Supervivencia por Tipo de Terapia')
ax1.set_xlabel('Días')
ax1.set_ylabel('Probabilidad de permanencia')
ax1.grid(True, alpha=0.3)

#log-rank test por tipo de terapia:
terapias = df["tipo_terapia"].unique()
results = logrank_test(
    df.loc[df["tipo_terapia"] == terapias[1], "tiempo_tratamiento"],
    df.loc[df["tipo_terapia"] == terapias[2], "tiempo_tratamiento"],
    event_observed_A= df.loc[df['tipo_terapia']==terapias[1], "abandono"],
    event_observed_B= df.loc[df['tipo_terapia']==terapias[2], "abandono"]
)
print(f"p-valor test log-rank para igualdad de de curvas de Kaplan Meier: {results.p_value:.4f}")
if results.p_value < 0.05:
    print("Al ser el p-valor menor que 0.05 podemos decir que el tipo de terapia afecta significativamente al abandono")
else:
    print("Aceptamos la hipótesis nula: el tipo de terapia parece no influir en el abandono")
#Curva de Kaplan Meier por compromiso
ax2 = axes[0,1]
for compromiso in [0,1]:
    mask = df["compromiso_bajo"] == compromiso
    label = "Compromiso bajo" if compromiso == 0 else "Compromiso normal"
    KMF.fit(durations = df.loc[mask, "tiempo_tratamiento"],
            event_observed = df.loc[mask, "abandono"],
            label = label)
    KMF.plot_survival_function(ax=ax2)
ax2.set_title("Supervivencia por compromiso inicial")
ax2.set_xlabel("Días")
ax2.set_ylabel("Probabilidad de permanencia")
ax2.grid(True, alpha = 0.3)

#Curva de Kaplan Meier por precio
ax3 = axes[1,0]
df["rango_precio"] = pd.cut(df["precio_sesion"], bins = [0,50,75,100], labels=["Bajo", "Medio", "Alto"])
for precio in df["rango_precio"].unique():
    mask = df["rango_precio"] == precio
    KMF.fit(durations = df.loc[mask,"tiempo_tratamiento"],
            event_observed = df.loc[mask, "abandono"],
            label = f"Precio {precio}")
    KMF.plot_survival_function(ax=ax3)
ax3.set_title("Supervivencia por rango de precios")
ax3.set_xlabel("Días")
ax3.set_ylabel("Probabilidad de permanencia")
ax3.grid(True, alpha = 0.3)

ax4 = axes[1,1]
motivos_frecuentes = df["motivo"].value_counts().nlargest(3).index
for motivo in motivos_frecuentes:
    mask = df["motivo"] == motivo
    KMF.fit(durations = df.loc[mask, "tiempo_tratamiento"],
            event_observed = df.loc[mask, "abandono"],
            label = motivo
    )
    KMF.plot_survival_function(ax=ax4)
ax4.set_title("Supervivencia por tipo de consulta")
ax4.set_xlabel("Días")
ax4.set_ylabel("Probabilidad de permanencia")

#log-rank test por motivo de consulta:
terapias = df["tipo_terapia"].unique()
results = logrank_test(
    df.loc[df["tipo_terapia"] == terapias[0]]["tiempo_tratamiento"],
    df.loc[df["tipo_terapia"] == terapias[1]]["tiempo_tratamiento"],
    event_observed_A= df.loc[df['tipo_terapia']==terapias[0], "abandono"],
    event_observed_B= df.loc[df['tipo_terapia']==terapias[1], "abandono"]
)
print(f"p-valor test log-rank para igualdad de de curvas de Kaplan Meier: {results.p_value:.4f}")
if results.p_value < 0.05:
    print("Al ser el p-valor menor que 0.05 podemos decir que el motivo de consulta afecta significativamente al abandono")
else:
    print("Aceptamos la hipótesis nula: el motivo de consulta parece no influir en el abandono")


plt.show()


