# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 07 — TABLAS PARA POWER BI (LOS TRES EQUIPOS EN UN SOLO LUGAR)
=============================================================================
Entrada:  salidas de los pasos 03b, 04, 05a, 05b, 05c y 06 de cada equipo
Salida:   carpeta powerbi/ con un CSV por tema (separador ';' y coma decimal)
"""
import os
import sys

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EQUIPOS = os.environ.get('EQUIPOS', 'CR009,CR010,CR011').split(',')
W = 60
PERCENTIL_UMBRAL = 95
VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
LLAVES = ['tramo_id', 'ventana_id']
SALIDA = f'{RUTA_BASE}/powerbi'
os.makedirs(SALIDA, exist_ok=True)


def carpeta(eq):
    if eq == 'CR010' and not os.path.isdir(f'{RUTA_BASE}/CR010'):
        return RUTA_BASE
    return f'{RUTA_BASE}/{eq}'


def leer(ruta):
    """Lee un CSV del pipeline, con ';' y coma decimal o con ',' y punto."""
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding='utf-8-sig') as f:
        cab = f.readline()
    if cab.count(';') > cab.count(','):
        return pd.read_csv(ruta, sep=';', decimal=',', encoding='utf-8-sig', low_memory=False)
    return pd.read_csv(ruta, encoding='utf-8-sig', low_memory=False)


def con_clave(df):
    """Columna única equipo-tramo-ventana para relacionar las tablas en Power BI."""
    df.insert(1, 'clave', df.equipo + '-' + df.tramo_id.astype(str) + '-' + df.ventana_id.astype(str))
    return df


def guardar(df, nombre):
    df.to_csv(f'{SALIDA}/{nombre}', sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'  {nombre:<32} {len(df):>9,} filas')


print('=' * 72)
print('PASO 07 — TABLAS PARA POWER BI')
print('=' * 72)

ventanas, por_variable, umbrales, nivel_n, curvas = [], [], [], [], []
resumen, agrup, perfil, proto, anticip, episodios = [], [], [], [], [], []

for eq in EQUIPOS:
    r = carpeta(eq)
    err = leer(f'{r}/{eq}_error_reconstruccion_W{W}.csv')
    if err is None:
        print(f'{eq}: falta {eq}_error_reconstruccion_W{W}.csv (paso 04); se salta.')
        continue
    print(f'{eq}: {r}')

    # --- ventanas de validación y prueba con su error --------------------
    sin_alarma = (err.n_reglas_activas == 0) if 'n_reglas_activas' in err.columns else err.es_normal
    base_umbral = err.loc[(err.conjunto == 'val') & sin_alarma, 'error_reconstruccion']
    umbral_ae = float(np.percentile(base_umbral, PERCENTIL_UMBRAL))
    v = err.copy()
    v.insert(0, 'equipo', eq)
    v = con_clave(v)
    v['estado_reglas'] = np.where(sin_alarma, 'normal', 'anomala')
    v['umbral_ae'] = umbral_ae
    parada = ('mayor separación' if 'err_AE_normales_sep' in err.columns
              and np.allclose(err.error_reconstruccion, err.err_AE_normales_sep) else 'menor pérdida')
    v['punto_parada'] = parada
    v['marcada_ae'] = (v.error_reconstruccion > umbral_ae).astype(int)

    n05a = leer(f'{r}/05a_nivel_ventanas.csv')
    n05b = leer(f'{r}/05b_nivel_revisado.csv')
    if n05a is not None:
        c = [x for x in ['timestamp_inicio', 'N_empinada', 'banda_N_empinada_natural', 'criticidad_min']
             if x in n05a.columns and x not in v.columns]
        v = v.merge(n05a[LLAVES + c], on=LLAVES, how='left', suffixes=('', '_05a'))
    if n05b is not None:
        c = [x for x in ['N_presencia', 'nivel_presencia', 'nivel_combinado', 'subsistemas', 'alarmas']
             if x in n05b.columns]
        v = v.merge(n05b[LLAVES + c], on=LLAVES, how='left')
    ventanas.append(v)

    # --- error por variable, en formato largo ----------------------------
    u_var = leer(f'{r}/04_umbral_por_variable.csv')
    col_ev = [f'evar_{x}' for x in VARIABLES if f'evar_{x}' in v.columns]
    if col_ev:
        larga = v[['equipo', 'clave', 'conjunto', 'estado_reglas'] + col_ev].melt(
            id_vars=['equipo', 'clave', 'conjunto', 'estado_reglas'],
            var_name='variable', value_name='error_variable')
        larga['variable'] = larga.variable.str.replace('evar_', '', regex=False)
        if u_var is not None:
            larga = larga.merge(u_var.rename(columns={'umbral_P95': 'umbral_variable'}),
                                on='variable', how='left')
            larga['sobre_umbral'] = (larga.error_variable > larga.umbral_variable).astype(int)
            larga['veces_umbral'] = larga.error_variable / larga.umbral_variable
        por_variable.append(larga)
    u = pd.DataFrame({'equipo': eq, 'variable': ['TOTAL'], 'umbral_P95': [umbral_ae]})
    if u_var is not None:
        u = pd.concat([u, u_var.assign(equipo=eq)[['equipo', 'variable', 'umbral_P95']]])
    umbrales.append(u)

    # --- resumen normal contra anómala por conjunto ---------------------
    for conj in ['val', 'test']:
        s = v[v.conjunto == conj]
        y = (s.estado_reglas == 'anomala').astype(int)
        en = s.loc[y == 0, 'error_reconstruccion'].median()
        ea = s.loc[y == 1, 'error_reconstruccion'].median()
        resumen.append({'equipo': eq, 'conjunto': conj, 'punto_parada': parada,
                        'ventanas_normales': int((y == 0).sum()), 'ventanas_anomalas': int(y.sum()),
                        'error_mediano_normal': en, 'error_mediano_anomala': ea,
                        'razon_anomala_normal': ea / en if en else np.nan,
                        'AUC': roc_auc_score(y, s.error_reconstruccion) if y.nunique() == 2 else np.nan,
                        'umbral_ae': umbral_ae,
                        'pct_marcadas_normales': 100 * s.loc[y == 0, 'marcada_ae'].mean(),
                        'pct_marcadas_anomalas': 100 * s.loc[y == 1, 'marcada_ae'].mean()})

    # --- nivel N de todas las ventanas (entrenamiento incluido) ----------
    if n05a is not None:
        n = n05a.copy()
        if n05b is not None:
            c = [x for x in ['N_presencia', 'nivel_presencia', 'nivel_combinado', 'subsistemas', 'alarmas']
                 if x in n05b.columns]
            n = n.merge(n05b[LLAVES + c], on=LLAVES, how='left')
        n.insert(0, 'equipo', eq)
        nivel_n.append(con_clave(n))

    # --- resto de las tablas, con la columna equipo ----------------------
    for lista, archivo in [(curvas, '03b_comparacion_epocas.csv'), (agrup, '05c_agrupamiento.csv'),
                           (perfil, '05c_perfil_grupos.csv'), (proto, '05c_prototipos.csv'),
                           (anticip, '06_resumen.csv'), (episodios, '06_episodios.csv')]:
        t = leer(f'{r}/{archivo}')
        if t is not None:
            t.insert(0, 'equipo', eq)
            lista.append(t)

print(f'\nGuardando en {SALIDA}:')
for lista, nombre in [(ventanas, 'pbi_ventanas.csv'), (por_variable, 'pbi_error_por_variable.csv'),
                      (umbrales, 'pbi_umbrales.csv'), (resumen, 'pbi_resumen_deteccion.csv'),
                      (nivel_n, 'pbi_nivel_N.csv'), (curvas, 'pbi_entrenamiento.csv'),
                      (agrup, 'pbi_agrupamiento.csv'), (perfil, 'pbi_perfil_grupos.csv'),
                      (proto, 'pbi_prototipos.csv'), (anticip, 'pbi_anticipacion.csv'),
                      (episodios, 'pbi_episodios.csv')]:
    if lista:
        df = pd.DataFrame(lista) if isinstance(lista[0], dict) else pd.concat(lista, ignore_index=True)
        guardar(df, nombre)

if resumen:
    print('\nResumen de la detección (error mediano por conjunto):')
    pd.set_option('display.width', 200)
    print(pd.DataFrame(resumen).round(4).to_string(index=False))
print('\nLISTO.')
