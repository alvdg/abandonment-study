import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generar_datos_pacientes(n_pacientes=1000):
    """
    Genera datos simulados de pacientes considerando:
    - Abandono (dejaron de venir sin completar)
    - Alta (completaron tratamiento con éxito)
    - Activos (siguen en tratamiento)
    """
    fecha_base = datetime(2025, 7, 1)
    
    # Probabilidades base
    prob_abandono = {
        'individual': 0.20,
        'pareja': 0.40,
        'familiar': 0.30
    }
    
    prob_alta = {
        'individual': 0.50,
        'pareja': 0.30,
        'familiar': 0.35
    }
    
    # Terapeutas con diferentes tasas de retención y éxito
    terapeutas = {
        'Dra. García': {'retencion': 0.90, 'exito': 0.85},
        'Dr. Martínez': {'retencion': 0.88, 'exito': 0.82},
        'Dra. López': {'retencion': 0.75, 'exito': 0.80},
        'Dr. Sánchez': {'retencion': 0.60, 'exito': 0.70},
        'Dra. Torres': {'retencion': 0.80, 'exito': 0.78}
    }
    
    datos = []
    
    for i in range(n_pacientes):
        # Variables básicas
        tipo = np.random.choice(['individual', 'pareja', 'familiar'], 
                               p=[0.6, 0.25, 0.15])
        terapeuta = np.random.choice(list(terapeutas.keys()))
        ter_data = terapeutas[terapeuta]
        
        motivo = np.random.choice(['ansiedad', 'depresión', 'estrés', 'pareja', 'duelo'],
                                 p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Precio según tipo
        if tipo == 'pareja':
            precio = np.random.choice([60, 65, 70, 75])
        elif tipo == 'familiar':
            precio = np.random.choice([70, 75, 80])
        else:
            precio = np.random.choice([45, 50, 55, 60])
        
        # Ajustar probabilidades por terapeuta
        p_abandono = prob_abandono[tipo] * (1 - ter_data['retencion'] + 0.5)
        p_alta = prob_alta[tipo] * ter_data['exito']
        p_activo = 1 - (p_abandono + p_alta)
        
        # Normalizar para que sumen 1
        total = p_abandono + p_alta + p_activo
        p_abandono /= total
        p_alta /= total
        p_activo /= total
        
        # Determinar estado del paciente
        estado = np.random.choice(['abandono', 'alta', 'activo'], 
                                 p=[p_abandono, p_alta, p_activo])
        
        # Fechas: inicio entre 1 mes y 18 meses antes
        dias_inicio = np.random.randint(30, 540)
        fecha_inicio = fecha_base - timedelta(days=dias_inicio)
        
        # Configurar según estado
        if estado == 'abandono':
            n_sesiones = np.random.randint(1, 8)
            dias_ultima = np.random.randint(60, 180)
            compromiso_bajo = np.random.choice([0, 1], p=[0.2, 0.8])
            abandono = 1
            alta = 0
            
        elif estado == 'alta':
            n_sesiones = np.random.randint(12, 30)
            # Alta reciente (últimos 30 días) o hace tiempo
            dias_ultima = np.random.choice([
                np.random.randint(1, 30),    # alta reciente
                np.random.randint(60, 120)    # alta antigua
            ], p=[0.6, 0.4])
            compromiso_bajo = np.random.choice([0, 1], p=[0.95, 0.05])
            abandono = 0
            alta = 1
            
        else:  # activo
            n_sesiones = np.random.randint(3, 15)  # Pueden llevar pocas o muchas
            dias_ultima = np.random.randint(1, 30)  # Sesión reciente
            compromiso_bajo = np.random.choice([0, 1], p=[0.85, 0.15])
            abandono = 0
            alta = 0
        
        fecha_ultima = fecha_base - timedelta(days=int(dias_ultima))
        
        datos.append({
            'paciente_id': f"P{i+1:04d}",
            'edad': np.random.randint(18, 70),
            'genero': np.random.choice(['M', 'F']),
            'tipo_terapia': tipo,
            'precio_sesion': precio,
            'terapeuta': terapeuta,
            'motivo': motivo,
            'fecha_inicio': fecha_inicio,
            'n_sesiones': n_sesiones,
            'fecha_ultima': fecha_ultima,
            'dias_desde_ultima': dias_ultima,
            'compromiso_bajo': compromiso_bajo,
            'abandono': abandono,
            'alta': alta
        })
    
    return pd.DataFrame(datos)

# Generar y probar
df = generar_datos_pacientes(1000)
df.to_csv('data/pacientes.csv', index=False)

print("=== DISTRIBUCIÓN DE ESTADOS ===")
print(f"Abandonos: {df['abandono'].mean():.1%}")
print(f"Altas:     {df['alta'].mean():.1%}")
print(f"Activos:   {(1 - df['abandono'].mean() - df['alta'].mean()):.1%}")

print("\n=== POR TIPO DE TERAPIA ===")
print(df.groupby('tipo_terapia')[['abandono', 'alta']].mean().round(3))

print("\n=== POR TERAPEUTA ===")
print(df.groupby('terapeuta')[['abandono', 'alta']].mean().round(3))

print("\n=== SESIONES POR ESTADO ===")
print(df.groupby('abandono')['n_sesiones'].describe())
print("\n=== SESIONES POR ALTA ===")
print(df.groupby('alta')['n_sesiones'].describe())