import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("data/pacientes.csv", parse_dates=["fecha_inicio", "fecha_ultima"])

#Limpio datos con dias negativos
df["tiempo_tratamiento"] = (df["fecha_ultima"] - df["fecha_inicio"]).dt.days
df = df[df["tiempo_tratamiento"] >= 0]

df["estado"] = "activo"
df.loc[df["abandono"] == 1, "estado"] = "abandono"
df.loc[df["alta"] ==1, "estado"] = "alta"

print(f"Procentaje de abandonos: {df["abandono"].mean():.1%}")
print(f"Procentaje de alta: {df["alta"].mean():.1%}")
print(f"Porcentaje de activos: {1-df["abandono"].mean()-df["alta"].mean():.1%}")
print(df.info())

fig, axes = plt.subplots(2,1, figsize = (10,10))

sns.histplot(data=df, x="edad", bins=20, kde=True, ax=axes[0])
axes[0].set_title("Distribución de Edades", fontsize=12)
axes[0].set_xlabel("Edad")
axes[0].set_ylabel("Frecuencia")

sns.boxplot(x="estado", y="n_sesiones", data=df, 
            order=["abandono", "alta", "activo"], ax=axes[1])
axes[1].set_title('Número de Sesiones por Estado', fontsize=12)
axes[1].set_xlabel('Estado')
axes[1].set_ylabel('Número de Sesiones')

plt.show()



#print(df.head())
#print(df.describe())