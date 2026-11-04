# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 05b — NIVEL DE ANOMALÍA REVISADO: REPETICIÓN Y COMBINACIÓN DE ALARMAS
=============================================================================
Entrada:  {EQUIPO}_activos_por_ventana_W60.csv, 05a_nivel_ventanas.csv, error del paso 04
Salida:   05b_nivel_revisado.csv, 05b_comparacion.csv, 05b_cruces.csv, 05b_variables_por_nivel.csv
"""
import os
import sys

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

EQUIPO = os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')
if RUTA_BASE not in sys.path:
    sys.path.append(RUTA_BASE)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

import criticidad


# =============================================================================
# 0. CONSTANTES
# =============================================================================
SEED = 42
W = 60
LLAVES = ['tramo_id', 'ventana_id']
PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
if PUNTO_PARADA not in ('perdida', 'separacion'):
    raise SystemExit(f"PUNTO_PARADA tiene que ser 'perdida' o 'separacion', no {PUNTO_PARADA!r}.")
AE = 'err_AE_normales_sep' if PUNTO_PARADA == 'separacion' else 'err_AE_normales'
NIVELES = ['0_normal', '1_leve', '2_media', '3_grave']
PESO = {c: 3 ** (3 - c) for c in (1, 2, 3)}            # 9 / 3 / 1, como el N actual

# Nivel combinado: criticidad máxima presente x número de subsistemas afectados
#                       1 subsistema   2 subsistemas   3 o más
MATRIZ = {1: ['2_media', '3_grave', '3_grave'],
          2: ['1_leve', '2_media', '3_grave'],
          3: ['1_leve', '1_leve', '2_media']}

ARCHIVO_ACTIVOS = f'{RUTA_EQ}/{EQUIPO}_activos_por_ventana_W{W}.csv'
ARCHIVO_NIVEL = f'{RUTA_EQ}/05a_nivel_ventanas.csv'
ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
for _f in [ARCHIVO_ACTIVOS, ARCHIVO_NIVEL]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste el paso 02 y el 05a?')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


# =============================================================================
# 1. DATOS
# =============================================================================
print('=' * 72)
print(f'PASO 05b — NIVEL REVISADO ({EQUIPO})')
print('=' * 72)
act = pd.read_csv(ARCHIVO_ACTIVOS)
nivel = pd.read_csv(ARCHIVO_NIVEL)
d = nivel[LLAVES + ['timestamp_inicio', 'conjunto_fecha', 'N_empinada', 'banda_N_empinada_natural']].merge(
    act, on=LLAVES, how='left', validate='one_to_one')
IDS = [r for r in act.columns if r in criticidad.CRITICIDAD]
C = d[IDS].fillna(0).astype(int)
A = C > 0
print(f'Ventanas: {len(d):,} | reglas: {len(IDS)}')


# =============================================================================
# 2. TRES ESCALAS SOBRE LAS MISMAS VENTANAS
# =============================================================================
# a) actual: N = suma de peso x episodios activos (paso 05a)
d['nivel_actual'] = d['banda_N_empinada_natural']

# b) presencia: cada regla cuenta una vez; una familia anidada, una vez y con su nivel más estricto
fam = {k: [r for r in v if r in IDS] for k, v in criticidad.familias_activas().items()}
en_familia = {r for v in fam.values() for r in v}
N_pres = np.zeros(len(d))
for miembros in fam.values():
    peso_fam = np.zeros(len(d))
    for r in miembros:                       # ordenados de menos a más estricto
        peso_fam = np.where(A[r], PESO[criticidad.CRITICIDAD[r]], peso_fam)
    N_pres += peso_fam
for r in IDS:
    if r not in en_familia:
        N_pres += A[r].to_numpy() * PESO[criticidad.CRITICIDAD[r]]
d['N_presencia'] = N_pres


def cortes_naturales(x):
    z = np.log1p(x).reshape(-1, 1)
    c = np.sort(KMeans(n_clusters=3, n_init=20, random_state=SEED).fit(z).cluster_centers_.ravel())
    return list(np.expm1((c[:-1] + c[1:]) / 2))


_tr = (d.conjunto_fecha == 'train').to_numpy() & (N_pres > 0)
CORTES_PRES = cortes_naturales(N_pres[_tr])
d['nivel_presencia'] = np.where(N_pres == 0, '0_normal',
                                np.array(NIVELES[1:])[np.digitize(N_pres, CORTES_PRES, right=True)])
print(f'Cortes de N por presencia (K-means sobre log(1 + N), entrenamiento): '
      f'{CORTES_PRES[0]:.2f} / {CORTES_PRES[1]:.2f}')

# c) combinada: criticidad máxima x subsistemas, con las reglas compuestas de la tabla experta
comp_act = {}
for nombre, info in criticidad.REGLAS_COMPUESTAS.items():
    if all(m in IDS for m in info['miembros']):
        comp_act[nombre] = A[info['miembros']].all(axis=1).to_numpy()
crit = np.full(len(d), 4)
for r in IDS:
    crit = np.where(A[r], np.minimum(crit, criticidad.CRITICIDAD[r]), crit)
for nombre, m in comp_act.items():
    crit = np.where(m, np.minimum(crit, criticidad.REGLAS_COMPUESTAS[nombre]['criticidad']), crit)
subs = [set() for _ in range(len(d))]
for r in IDS:
    for i in np.where(A[r])[0]:
        subs[i].add(criticidad.COMPONENTE[r])
d['crit_max'] = crit
d['n_subsistemas'] = [len(s) for s in subs]
d['subsistemas'] = [', '.join(sorted(s)) for s in subs]
d['compuestas'] = [', '.join(n for n, m in comp_act.items() if m[i]) for i in range(len(d))]
d['nivel_combinado'] = [
    '0_normal' if c == 4 else MATRIZ[c][min(n, 3) - 1] for c, n in zip(crit, d.n_subsistemas)]
print('Compuestas de la tabla experta presentes:',
      {n: int(m.sum()) for n, m in comp_act.items()} or 'ninguna evaluable')


# =============================================================================
# 3. ¿CUÁNTO PESA LA REPETICIÓN EN EL N ACTUAL?
# =============================================================================
print('\n' + '=' * 72)
print('3. LAS GRAVES DEL N ACTUAL: ¿POR CRITICIDAD O POR REPETICIÓN?')
print('=' * 72)
g = d.nivel_actual == '3_grave'
sin_c1 = g & (d.crit_max > 1)
print(f'Ventanas graves: {int(g.sum()):,}; sin ninguna alarma de criticidad 1: '
      f'{int(sin_c1.sum()):,} ({100 * sin_c1.sum() / max(g.sum(), 1):.0f} %)')
_rep = C[sin_c1].sum().sort_values(ascending=False).head(3)
print('Reglas que más se repiten en esas ventanas:', {k: int(v) for k, v in _rep.items()})
for etq, col in [('por presencia', 'nivel_presencia'), ('combinado', 'nivel_combinado')]:
    print(f'  Esas mismas ventanas, nivel {etq}: '
          + ', '.join(f'{k[2:]} {100 * v:.0f} %' for k, v in
                      d.loc[sin_c1, col].value_counts(normalize=True).sort_index().items()))


# =============================================================================
# 4. COMPARACIÓN DE LAS TRES ESCALAS
# =============================================================================
print('\n' + '=' * 72)
print('4. DISTRIBUCIÓN, CRUCES Y COHERENCIA CON EL ERROR DEL AUTOENCODER')
print('=' * 72)
ESCALAS = {'actual': 'nivel_actual', 'presencia': 'nivel_presencia', 'combinado': 'nivel_combinado'}
hay_error = os.path.exists(ARCHIVO_ERROR)
if hay_error:
    _e = pd.read_csv(ARCHIVO_ERROR, usecols=lambda c: c in LLAVES + ['conjunto', AE, 'error_reconstruccion']
                     or c.startswith('evar_'))
    if 'error_reconstruccion' in _e.columns and not np.allclose(_e[AE], _e.error_reconstruccion):
        raise SystemExit(f'El paso 04 se corrió con otro punto de parada: córrelo con PUNTO_PARADA={PUNTO_PARADA!r}.')
    _e = _e.drop(columns='error_reconstruccion', errors='ignore')
    d = d.merge(_e, on=LLAVES, how='left')
    print(f'Error del autoencoder: {ARCHIVO_ERROR}')
else:
    print('Sin el archivo del paso 04: se omite la coherencia con el error.')


def auc(puntaje, positivo):
    positivo = np.asarray(positivo, bool)
    n1, n0 = positivo.sum(), (~positivo).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(puntaje).rank().to_numpy()
    return (r[positivo].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


filas = []
for esc, col in ESCALAS.items():
    for cj in ['train', 'val', 'test']:
        s = d[d.conjunto_fecha == cj][col].value_counts(normalize=True)
        f = {'escala': esc, 'conjunto': cj}
        f.update({f'pct_{n[2:]}': round(100 * s.get(n, 0), 1) for n in NIVELES})
        if hay_error and cj == 'val':
            v = d[(d.conjunto == 'val') & d[AE].notna()]
            med = v.groupby(col)[AE].median()
            for n in NIVELES[1:]:
                f[f'veces_{n[2:]}'] = round(med.get(n, np.nan) / med.get('0_normal', np.nan), 2)
            for a_, b_ in [('1_leve', '2_media'), ('2_media', '3_grave')]:
                m = v[col].isin([a_, b_])
                f[f'auc_{a_[2:]}_{b_[2:]}'] = round(auc(v.loc[m, AE], v.loc[m, col] == b_), 3)
            pos = v[v[col] != '0_normal']
            f['spearman_error'] = round(spearmanr(pos[col].map(NIVELES.index), pos[AE]).correlation, 3)
        filas.append(f)
comp = pd.DataFrame(filas)
pd.set_option('display.width', 220)
print(comp.to_string(index=False))
print('\nveces_*: error mediano del nivel / el de las ventanas sin alarma (validación).')
print('auc_*: ¿el error separa niveles vecinos? 0,5 = no los distingue.')
print('Es coherencia con lo que ve el modelo, no la verdad: ninguna escala tiene etiqueta real.')

cruces = pd.concat({esc: pd.crosstab(d.nivel_actual, d[col]) for esc, col in ESCALAS.items()
                    if esc != 'actual'}, names=['escala'])
print('\nCruce con el nivel actual (filas: actual; columnas: escala revisada):')
print(cruces.to_string())


# =============================================================================
# 4b. ¿QUÉ VARIABLES EXPLICAN CADA NIVEL? (error por variable del paso 04)
# =============================================================================
EVAR = [c for c in d.columns if c.startswith('evar_')]
vpn = pd.DataFrame()
if EVAR:
    print('\n' + '=' * 72)
    print('4b. PARTICIPACIÓN DE CADA VARIABLE EN EL ERROR, POR NIVEL (validación y prueba)')
    print('=' * 72)
    _ev = d[EVAR].to_numpy()
    _ok = ~np.isnan(_ev).any(axis=1)
    _aporte = np.where(_ok[:, None], _ev / np.where(_ok, _ev.sum(axis=1), 1)[:, None], np.nan)
    _nombres = [c[len('evar_'):] for c in EVAR]
    filas_v = []
    for esc, col in ESCALAS.items():
        for n in NIVELES:
            s = _ok & (d[col] == n).to_numpy()
            if s.sum() == 0:
                continue
            media = _aporte[s].mean(axis=0)
            top = np.argsort(-media)[:3]
            fila = {'escala': esc, 'nivel': n[2:], 'ventanas': int(s.sum())}
            fila.update({f'var_{i + 1}': f'{_nombres[j]} {100 * media[j]:.0f} %' for i, j in enumerate(top)})
            fila.update({f'pct_{v}': round(100 * m, 1) for v, m in zip(_nombres, media)})
            filas_v.append(fila)
    vpn = pd.DataFrame(filas_v)
    print(vpn[['escala', 'nivel', 'ventanas', 'var_1', 'var_2', 'var_3']].to_string(index=False))
    print('Porcentaje: parte del error de la ventana que aporta esa variable, en promedio.')
else:
    print('\nEl archivo del paso 04 no trae el error por variable (evar_*): se omite la sección 4b.')


# =============================================================================
# 5. EJEMPLOS CON VENTANAS REALES
# =============================================================================
print('\n' + '=' * 72)
print('5. EJEMPLOS')
print('=' * 72)
d['alarmas'] = [' + '.join(f'{r}x{n}' for r, n in fila.items() if n > 0)
                for fila in C.to_dict('records')]
ejemplos = []
for etq, m in [
        ('una alarma de presión que se repite (criticidad 2)', (d.alarmas.str.fullmatch(r'Dx\d+')) & (C['D'] >= 10)),
        ('una sola alarma crítica', d.alarmas.str.fullmatch(r'A5x1') | d.alarmas.str.fullmatch(r'Dx1 \+ A5x1')),
        ('presión alta con corriente alta (dos subsistemas)', (A.get('A5', False)) & (A.get('A', False))),
        ('tres o más subsistemas', d.n_subsistemas >= 3)]:
    s = d[m]
    if len(s):
        r = s.iloc[0]
        ejemplos.append({'caso': etq, 'ventanas_asi': int(len(s)), 'alarmas': r.alarmas,
                         'N_actual': int(r.N_empinada), 'nivel_actual': r.nivel_actual[2:],
                         'N_presencia': int(r.N_presencia), 'nivel_presencia': r.nivel_presencia[2:],
                         'subsistemas': r.subsistemas, 'nivel_combinado': r.nivel_combinado[2:]})
print(pd.DataFrame(ejemplos).to_string(index=False))


# =============================================================================
# 6. GUARDAR
# =============================================================================
d[LLAVES + ['timestamp_inicio', 'conjunto_fecha', 'alarmas', 'N_empinada', 'nivel_actual',
            'N_presencia', 'nivel_presencia', 'crit_max', 'n_subsistemas', 'subsistemas',
            'compuestas', 'nivel_combinado']].to_csv(
    salida('05b_nivel_revisado.csv'), sep=';', decimal=',', index=False, encoding='utf-8-sig')
comp.to_csv(salida('05b_comparacion.csv'), sep=';', decimal=',', index=False, encoding='utf-8-sig')
cruces.to_csv(salida('05b_cruces.csv'), sep=';', encoding='utf-8-sig')
if len(vpn):
    vpn.to_csv(salida('05b_variables_por_nivel.csv'), sep=';', decimal=',', index=False, encoding='utf-8-sig')
print(f'\nGuardado: 05b_nivel_revisado.csv, 05b_comparacion.csv, 05b_cruces.csv y '
      f'05b_variables_por_nivel.csv en {RUTA_EQ}')
