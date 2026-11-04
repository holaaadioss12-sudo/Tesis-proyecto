# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04b — ¿CÓMO AUMENTAR LA SEPARACIÓN DEL ERROR ENTRE NORMALES Y ANÓMALAS?
=============================================================================
Entrada:  {EQUIPO}_error_reconstruccion_W60.csv del paso 04 (no reentrena nada)
Salida:   04b_separacion.csv en la carpeta base

Compara, con el mismo modelo ya entrenado:
  1. Punto de parada: menor pérdida (oficial) contra mayor separación.
  2. Forma de resumir el error de la ventana: promedio de las 14 variables
     (oficial) contra el error de cada variable dividido por su error típico
     en operación normal (promedio, 3 mayores o máximo).
La decisión se toma en validación; prueba solo se informa.
"""
import os

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
VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
TOLERANCIA_AUC = 0.01          # pérdida de AUC de validación que se acepta
BASE = ('menor pérdida', 'promedio')


def carpeta(eq):
    if eq == 'CR010' and not os.path.isdir(f'{RUTA_BASE}/CR010'):
        return RUTA_BASE
    return f'{RUTA_BASE}/{eq}'


def puntajes(d):
    """Puntaje de cada ventana según punto de parada y forma de resumir."""
    p = {}
    if 'err_AE_normales' in d.columns:
        p[('menor pérdida', 'promedio')] = d['err_AE_normales'].to_numpy()
    if 'err_AE_normales_sep' in d.columns:
        p[('mayor separación', 'promedio')] = d['err_AE_normales_sep'].to_numpy()
    ev_cols = [f'evar_{v}' for v in VARIABLES if f'evar_{v}' in d.columns]
    if len(ev_cols) == len(VARIABLES):
        ev = d[ev_cols].to_numpy()
        ref = np.median(ev[(d.conjunto == 'val').to_numpy() & d.normal.to_numpy()], axis=0)
        z = ev / np.maximum(ref, 1e-12)              # veces su error típico normal
        zs = -np.sort(-z, axis=1)
        p[('menor pérdida', 'estandarizado: promedio')] = z.mean(axis=1)
        p[('menor pérdida', 'estandarizado: 3 mayores')] = zs[:, :3].mean(axis=1)
        p[('menor pérdida', 'estandarizado: máximo')] = zs[:, 0]
    return p


def medir(s, y, umbral):
    en, ea = np.median(s[~y]), np.median(s[y])
    return {'error_mediano_normal': en, 'error_mediano_anomala': ea,
            'razon': ea / en if en > 0 else np.nan,
            'AUC': roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan,
            'pct_anomalas_sobre_P95': 100 * float((s[y] > umbral).mean()),
            'pct_normales_sobre_P95': 100 * float((s[~y] > umbral).mean())}


print('=' * 78)
print('PASO 04b — SEPARACIÓN DEL ERROR: PUNTO DE PARADA Y FORMA DE RESUMIR')
print('=' * 78)
print(f'Regla, fijada antes de mirar prueba: entre las opciones cuyo AUC de validación\n'
      f'no baja más de {TOLERANCIA_AUC} respecto de la oficial, se elige la de mayor\n'
      f'razón en validación.')

filas, elegidas = [], []
for eq in EQUIPOS:
    ruta = f'{carpeta(eq)}/{eq}_error_reconstruccion_W{W}.csv'
    if not os.path.exists(ruta):
        print(f'\n{eq}: falta {ruta} (paso 04); se salta.')
        continue
    d = pd.read_csv(ruta, low_memory=False)
    d['normal'] = (d.n_reglas_activas == 0) if 'n_reglas_activas' in d.columns else d.es_normal.astype(bool)
    es_val = (d.conjunto == 'val').to_numpy()
    y = ~d.normal.to_numpy()

    for (parada, resumen), s in puntajes(d).items():
        umbral = np.percentile(s[es_val & ~y], 95)
        for conj in ['val', 'test']:
            sel = (d.conjunto == conj).to_numpy()
            filas.append({'equipo': eq, 'parada': parada, 'resumen': resumen, 'conjunto': conj,
                          **medir(s[sel], y[sel], umbral)})

    t = pd.DataFrame([f for f in filas if f['equipo'] == eq])
    v = t[t.conjunto == 'val'].set_index(['parada', 'resumen'])
    p = t[t.conjunto == 'test'].set_index(['parada', 'resumen'])
    tabla = pd.DataFrame({'razón val': v.razon, 'AUC val': v.AUC, 'captura val %': v.pct_anomalas_sobre_P95,
                          'razón test': p.razon, 'AUC test': p.AUC, 'captura test %': p.pct_anomalas_sobre_P95,
                          'falsas test %': p.pct_normales_sobre_P95})
    admisibles = tabla[tabla['AUC val'] >= tabla.loc[BASE, 'AUC val'] - TOLERANCIA_AUC]
    elegida = admisibles['razón val'].idxmax()
    tabla['elegida'] = ['<-' if i == elegida else '' for i in tabla.index]
    print(f'\n--- {eq} ' + '-' * 70)
    print(tabla.round(3).to_string())
    b, e = tabla.loc[BASE], tabla.loc[elegida]
    print(f'Elegida en validación: {elegida[0]} + {elegida[1]}')
    print(f'  razón val  {b["razón val"]:.2f}x -> {e["razón val"]:.2f}x | AUC val  {b["AUC val"]:.3f} -> {e["AUC val"]:.3f}')
    print(f'  razón test {b["razón test"]:.2f}x -> {e["razón test"]:.2f}x | AUC test {b["AUC test"]:.3f} -> {e["AUC test"]:.3f}  (solo se informa)')
    elegidas.append({'equipo': eq, 'parada': elegida[0], 'resumen': elegida[1]})

if filas:
    out = pd.DataFrame(filas)
    out = out.merge(pd.DataFrame(elegidas).assign(elegida=1), on=['equipo', 'parada', 'resumen'], how='left')
    out['elegida'] = out.elegida.fillna(0).astype(int)
    out.to_csv(f'{RUTA_BASE}/04b_separacion.csv', sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'\nGuardado: {RUTA_BASE}/04b_separacion.csv')
print('\nrazón: error mediano anómalas / normales. captura: % de anómalas sobre el umbral P95')
print('de las normales de validación. falsas: % de normales de prueba sobre ese umbral.')
print('La mayor separación se eligió mirando validación: su razón de validación es optimista;')
print('prueba dice si se sostiene.')
print('\nLISTO.')
