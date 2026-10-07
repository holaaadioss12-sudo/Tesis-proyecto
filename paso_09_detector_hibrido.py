# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 09 — DETECTOR HÍBRIDO: AUTOENCODER + NIVEL (z_medias)
=============================================================================
Entradas: {EQUIPO}_ventanas_W60.csv, {EQUIPO}_features_W60.csv,
          {EQUIPO}_error_reconstruccion_W60.csv, {EQUIPO}_corte_split.json
          (las mismas del paso 06). No entrena nada: corre en CPU.
Salida:   09_hibrido.csv (una fila por detector) y 09_por_regla.csv

-----------------------------------------------------------------------------
POR QUÉ ESTE PASO (medido antes de decidir)
-----------------------------------------------------------------------------
El desglose de la AUC de detección en prueba por regla dominante mostró que
el autoencoder separa muy bien las anomalías de forma o de relación entre
sensores (D, O, A17, A13, C: AUC 0,85-0,99), y mal tres reglas que son un
cruce marginal de NIVEL de una sola variable: A y A4 (corriente CM > 100) y
E (presión PEL < 150). En CR009 la regla E incluso tiene MENOS error que lo
normal (AUC 0,32). Sin esas tres, la AUC del autoencoder es 0,91 / 0,93 /
0,91 en CR009 / CR010 / CR011: el modelo es consistente entre equipos; lo
que cambia es cuánto pesan esas reglas de nivel (57-62 % de las anomalías).

Un cruce marginal de nivel es una desviación chica en el espacio de los
sensores, y el autoencoder la reconstruye bien: no es su trabajo. Es
justamente lo que capta z_medias. La propuesta es combinar los dos puntajes,
cada uno en la escala de "qué tan raro es frente a lo normal":

    p_AE(x) = fracción de las ventanas QUIETAS de VALIDACIÓN con error <= x
    p_z(x)  = lo mismo con z_medias
    H_max   = max(p_AE, p_z)              sin parámetros: no supervisado
    H_prom  = a * p_AE + (1 - a) * p_z    a elegido en VALIDACIÓN

H_max es el que se defiende: no usa etiquetas para nada. H_prom se reporta
al lado para mostrar cuánto se gana eligiendo el peso con las reglas de
validación. Prueba se mira una sola vez, al final.

Se probaron y DESCARTARON (medido con los archivos del paso 05):
  - suavizar el error en el tiempo (EWMA dentro del tramo): baja la AUC en
    los tres equipos (CR010 0,836 -> 0,79 con a = 0,7), porque las alarmas
    duran poco y el suavizado las diluye.
=============================================================================
"""
import os
import sys
import json

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

if RUTA_BASE not in sys.path:
    sys.path.append(RUTA_BASE)

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and _os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'     # CR010 movido a su subcarpeta
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 42
W = 60
N_BOOT = 1000
PERCENTIL_UMBRAL = 95
MIN_POR_REGLA = 30
AE = 'err_AE_normales'          # checkpoint de pérdida mínima (04e, 13.6)

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
LLAVES = ['tramo_id', 'ventana_id']

ventanas = pd.read_csv(f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv')
err = pd.read_csv(f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv')
medias = [f'{v}_media' for v in VARIABLES]
features = pd.read_csv(f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv',
                       usecols=LLAVES + medias)

cols_err = [c for c in ('err_AE_normales', 'err_AE_normales_sep')
            if c in err.columns]
d = (ventanas[ventanas.tramo_id.isin(set(err.tramo_id))]
     [LLAVES + ['n_reglas_activas']]
     .merge(err[LLAVES + ['conjunto', 'etiqueta_ventana'] + cols_err],
            on=LLAVES, how='left', validate='one_to_one')
     .merge(features, on=LLAVES, how='left', validate='one_to_one')
     .sort_values(LLAVES).reset_index(drop=True))
assert d[AE].notna().all(), 'ventanas de val/test sin error de reconstrucción'
d['alarma'] = d.n_reglas_activas > 0
d['quieta'] = ~d.alarma

print('=' * 72)
print(f'PASO 09 — DETECTOR HÍBRIDO ({EQUIPO})')
print('=' * 72)

# --- los puntajes, todos referidos a las quietas de validación -------------
ref = d[(d.conjunto == 'val') & d.quieta]
_mu, _sd = ref[medias].mean(), ref[medias].std().replace(0, 1.0)
d['z_medias'] = ((d[medias] - _mu) / _sd).abs().max(axis=1)
ref = d[(d.conjunto == 'val') & d.quieta]


def percentil_normal(col):
    """Fracción de las quietas de validación con puntaje <= x."""
    base = np.sort(ref[col].to_numpy())
    return np.searchsorted(base, d[col].to_numpy(), side='right') / len(base)


d['p_AE'] = percentil_normal(AE)
d['p_z'] = percentil_normal('z_medias')
d['H_max'] = np.maximum(d.p_AE, d.p_z)

va = d[d.conjunto == 'val']
grilla = []
for a in np.round(np.arange(0, 1.01, 0.1), 1):
    s = a * va.p_AE + (1 - a) * va.p_z
    grilla.append((a, roc_auc_score(va.alarma, s)))
A_OPT = max(grilla, key=lambda t: t[1])[0]
print('Peso del AE en H_prom elegido en VALIDACIÓN (AUC de detección):')
print('  ' + ' | '.join(f'{a:.1f}: {v:.3f}' for a, v in grilla))
print(f'  -> a = {A_OPT}')
d['H_prom'] = A_OPT * d.p_AE + (1 - A_OPT) * d.p_z

DETECTORES = {'AE_normales': AE, 'z_medias': 'z_medias',
              'H_max': 'H_max', 'H_prom': 'H_prom'}
if 'err_AE_normales_sep' in d:
    DETECTORES = {'AE_normales_sep': 'err_AE_normales_sep', **DETECTORES}

# --- lo que viene después de cada ventana (igual que el paso 06) -----------
g = d.groupby('tramo_id')
fut = g.alarma.shift(-1)
d['futura_1'] = np.where(fut.isna(), np.nan, fut.fillna(False).astype(float))

te = d[d.conjunto == 'test'].copy()
rng = np.random.default_rng(SEED)


def bootstrap_tramos(df, estadistico, n=N_BOOT):
    grupos = df.groupby('tramo_id').indices
    claves = np.array(list(grupos))
    vals = []
    for _ in range(n):
        sel = np.concatenate([grupos[c] for c in rng.choice(claves, len(claves))])
        v = estadistico(df.iloc[sel])
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def auc(y, s):
    return roc_auc_score(y, s) if pd.Series(y).nunique() == 2 else np.nan


q1 = te[te.quieta & te.futura_1.notna()].copy()
q1['y'] = q1.futura_1.astype(bool)
ref_va = d[(d.conjunto == 'val') & d.quieta]

filas = []
print(f'\nPrueba: {len(te):,} ventanas, {int(te.alarma.sum()):,} con alarma | '
      f'quietas con siguiente observable: {len(q1):,}\n')
print(f'{"detector":<16}{"AUC detección":>24}{"AUC anticipación":>26}'
      f'{"RR 10 min":>24}')
for nom, col in DETECTORES.items():
    a_det = auc(te.alarma, te[col])
    lo_d, hi_d = bootstrap_tramos(te, lambda x, c=col: auc(x.alarma, x[c]))
    a_ant = auc(q1.y, q1[col])
    lo_a, hi_a = bootstrap_tramos(q1, lambda x, c=col: auc(x.y, x[c]))
    u = float(np.percentile(ref_va[col], PERCENTIL_UMBRAL))
    m = q1[col] > u

    def rr(x, c=col, u=u):
        mm = x[c] > u
        p1 = x.y[mm].mean() if mm.any() else np.nan
        p0 = x.y[~mm].mean() if (~mm).any() else np.nan
        return p1 / p0 if p0 and np.isfinite(p0) and p0 > 0 else np.nan
    r = rr(q1)
    lo_r, hi_r = bootstrap_tramos(q1, rr)
    filas.append({'equipo': EQUIPO, 'detector': nom,
                  'auc_deteccion': round(a_det, 3),
                  'auc_det_ic_inf': round(lo_d, 3),
                  'auc_det_ic_sup': round(hi_d, 3),
                  'auc_anticipacion': round(a_ant, 3),
                  'auc_ant_ic_inf': round(lo_a, 3),
                  'auc_ant_ic_sup': round(hi_a, 3),
                  'rr_10min': round(r, 3), 'rr_ic_inf': round(lo_r, 3),
                  'rr_ic_sup': round(hi_r, 3),
                  'pct_quietas_marcadas_test': round(100 * m.mean(), 1)})
    print(f'{nom:<16}{a_det:>8.3f} [{lo_d:.3f}, {hi_d:.3f}]'
          f'{a_ant:>10.3f} [{lo_a:.3f}, {hi_a:.3f}]'
          f'{r:>8.2f} [{lo_r:.2f}, {hi_r:.2f}]')

# --- diferencias pareadas (mismos tramos remuestreados para ambos) ---------
print('\nDiferencias pareadas de AUC de detección en prueba:')
for a_, b_ in (('H_max', 'z_medias'), ('H_max', 'AE_normales'),
               ('H_prom', 'z_medias'), ('AE_normales', 'z_medias')):
    ca, cb = DETECTORES[a_], DETECTORES[b_]
    f = lambda x, ca=ca, cb=cb: auc(x.alarma, x[ca]) - auc(x.alarma, x[cb])
    dif = f(te)
    lo, hi = bootstrap_tramos(te, f)
    veredicto = ('mejor' if lo > 0 else 'peor' if hi < 0 else 'empate')
    print(f'  {a_:<12} - {b_:<12} = {dif:+.3f}  [{lo:+.3f}, {hi:+.3f}]  '
          f'-> {veredicto}')
    filas.append({'equipo': EQUIPO, 'detector': f'{a_} - {b_}',
                  'auc_deteccion': round(dif, 3), 'auc_det_ic_inf': round(lo, 3),
                  'auc_det_ic_sup': round(hi, 3)})

# --- desglose por regla dominante ------------------------------------------
print(f'\nAUC de detección en prueba por regla dominante (>= {MIN_POR_REGLA} '
      f'ventanas), contra las quietas de prueba:')
nor = te[te.quieta]
filas_r = []
for regla, s in te[te.alarma].groupby('etiqueta_ventana'):
    if len(s) < MIN_POR_REGLA:
        continue
    fila = {'equipo': EQUIPO, 'regla': regla, 'ventanas': len(s)}
    y = np.r_[np.zeros(len(nor)), np.ones(len(s))]
    for nom in ('AE_normales', 'z_medias', 'H_max'):
        c = DETECTORES[nom]
        fila[f'auc_{nom}'] = round(roc_auc_score(y, np.r_[nor[c], s[c]]), 3)
    filas_r.append(fila)
por_regla = pd.DataFrame(filas_r).sort_values('ventanas', ascending=False)
print(por_regla.to_string(index=False))

pd.DataFrame(filas).to_csv(f'{RUTA_EQ}/09_hibrido.csv', sep=';', decimal=',',
                           index=False, encoding='utf-8-sig')
por_regla.to_csv(f'{RUTA_EQ}/09_por_regla.csv', sep=';', decimal=',',
                 index=False, encoding='utf-8-sig')
print(f'\nGuardado: {RUTA_EQ}/09_hibrido.csv y 09_por_regla.csv')
