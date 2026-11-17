# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04h — LÍNEA BASE DE INGENIERÍA DE PROCESOS: PCA (SPE Y T² DE HOTELLING)
=============================================================================
El monitoreo multivariable clásico de la industria de procesos, sobre las mismas
ventanas que el autoencoder. El PCA se ajusta por instante (14 variables) con los
registros de las MISMAS ventanas normales de entrenamiento del 03b y con su
normalizador: es la versión lineal y sin tiempo del autoencoder temporal, cuyo
latente también tiene k valores por instante.
- SPE (o Q): error de reconstrucción del PCA; el de la ventana es el promedio de sus
  60 instantes. Su reparto por variable es el gráfico de contribuciones.
- T² de Hotelling: distancia dentro del plano de las componentes, promedio de la ventana.
- k con la regla del 03a, solo en validación: mayor AUC del SPE; si empata (IC 95 %
  por bloques de 6 h con el 0), mayor ventaja de localización; si también empata, el
  menor k. Se informan además el k del autoencoder y el k con 90 % de varianza.
- Umbral P95 de las ventanas sin alarma de validación, como en el paso 04.
Entrada:  {EQUIPO}_limpio.csv y {EQUIPO}_corte_split.json (01), {EQUIPO}_ventanas_W60.csv y
          {EQUIPO}_features_W60.csv (02), normalizador_W60.npz y autoencoder_W60_config.json
          (03b), {EQUIPO}_error_reconstruccion_W60.csv (04) y 05a_nivel_ventanas.csv (opcional)
Salida:   04h_pca_barrido.csv, 04h_pca_cargas.csv, 04h_pca_deteccion.csv,
          04h_pca_comparaciones.csv, 04h_pca_localizacion.csv, 04h_pca_delta.csv,
          04h_pca_ventanas.csv, 04h_pca.png
Todo se decide en validación; prueba solo se informa.
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

EQUIPO = os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')
if RUTA_BASE not in sys.path:
    sys.path.append(RUTA_BASE)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from reglas_engine import evaluar_reglas, POR_ID, IDS as IDS_REGLAS


# =============================================================================
# 0. CONSTANTES
# =============================================================================
W = 60
STRIDE_TRAIN = 12               # como el 03b: una ventana cada 2 min
PERCENTIL = 95                  # umbral del paso 04
VAR_CLASICA = 0.90              # k de la práctica clásica: 90 % de la varianza
N_BOOT = 1000
BLOQUE = 36                     # bloques de 6 h para los intervalos (como el 03a y el 04c)
SEED = 42
LLAVES = ['tramo_id', 'ventana_id']
BANDA = 'banda_N_empinada_natural'
GRAVE = '3_grave'

PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
if PUNTO_PARADA not in ('perdida', 'separacion'):
    raise SystemExit(f"PUNTO_PARADA tiene que ser 'perdida' o 'separacion', no {PUNTO_PARADA!r}.")
AE = 'err_AE_normales_sep' if PUNTO_PARADA == 'separacion' else 'err_AE_normales'

ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
ARCHIVO_NIVEL = f'{RUTA_EQ}/05a_nivel_ventanas.csv'
RUTA_NORMALIZADOR = f'{RUTA_EQ}/normalizador_W{W}.npz'
RUTA_CONFIG = f'{RUTA_EQ}/autoencoder_W{W}_config.json'
for _f in [ARCHIVO_LIMPIO, ARCHIVO_CORTE, ARCHIVO_VENTANAS, ARCHIVO_ERROR, RUTA_NORMALIZADOR]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste los pasos 01, 02, 03b y 04?')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


def guardar(df, nombre):
    df.to_csv(salida(nombre), sep=';', decimal=',', index=False, encoding='utf-8-sig')


def variables_de(regla):
    r = POR_ID[regla]
    return [r[c] for c in ('var', 'var2', 'a', 'b') if c in r] + list(r.get('vars', []))


def auc(y, s):
    """AUC por rangos (empates promediados); y = 1 en las ventanas con alarma."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# =============================================================================
# 1. LOS MISMOS TRAMOS, PARTICIÓN Y NORMALIZADOR QUE EL AUTOENCODER
# =============================================================================
print('=' * 72)
print(f'PASO 04h — LÍNEA BASE PCA: SPE Y T² ({EQUIPO})')
print('=' * 72)


def cargar_limpio(ruta):
    cabecera = pd.read_csv(ruta, nrows=0)
    col_tiempo = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col_tiempo], low_memory=False)
    return df.set_index(col_tiempo).sort_index().rename_axis('timestamp')


def segmentar_tramos_continuos(df, freq='10s', max_gap=1):
    idx = df.index.to_series()
    gaps = idx.diff() > pd.Timedelta(freq) * max_gap
    df = df.copy()
    df['tramo_id'] = gaps.cumsum()
    return [g.drop(columns='tramo_id') for _, g in df.groupby('tramo_id')]


df = cargar_limpio(ARCHIVO_LIMPIO)
_norm = np.load(RUTA_NORMALIZADOR, allow_pickle=True)
VARIABLES = list(_norm['variables'])
NV = len(VARIABLES)
_faltan = [v for v in VARIABLES if v not in df.columns]
if _faltan:
    raise SystemExit(f'El limpio no trae {_faltan}: no es el mismo que usó el 03b.')
MEDIA_N = pd.Series(_norm['media'], index=VARIABLES)
DESV_N = pd.Series(_norm['std'], index=VARIABLES)


def normalizar(t):
    return ((t[VARIABLES] - MEDIA_N) / DESV_N).to_numpy(dtype=np.float64)


tramos = [t for t in segmentar_tramos_continuos(df) if len(t) >= W]   # el índice es el tramo_id del 04
_corte = json.load(open(ARCHIVO_CORTE))
T_TRAIN, T_VAL = pd.Timestamp(_corte['T_train']), pd.Timestamp(_corte['T_val'])
_ini = np.array([t.index[0] for t in tramos])
ID_TR = np.flatnonzero(_ini <= T_TRAIN)
ID_EV = np.flatnonzero(_ini > T_TRAIN)
print(f'Tramos útiles: {len(tramos):,} | entrenamiento {len(ID_TR):,} | validación y prueba {len(ID_EV):,}')
print(f'Corte: T_train = {T_TRAIN} | T_val = {T_VAL} | normalizador del 03b ({NV} variables)')


# =============================================================================
# 2. AJUSTE DEL PCA CON LAS VENTANAS NORMALES DE ENTRENAMIENTO
# =============================================================================
print('\n' + '=' * 72)
print('2. AJUSTE DEL PCA (mismas ventanas normales de entrenamiento que el 03b)')
print('=' * 72)
S1, S2, N_REG = np.zeros(NV), np.zeros((NV, NV)), 0.0
N_VEN = N_NORM = 0
for c, i in enumerate(ID_TR):
    t = tramos[i]
    res = evaluar_reglas(t, verbose=False)
    cols = [x for x in res.columns if x not in ('n_reglas_activadas', 'estado')]
    acum = np.concatenate(([0], np.cumsum(res[cols].to_numpy().any(axis=1).astype(np.int64))))
    ini = np.arange(0, len(t) - W + 1, STRIDE_TRAIN)
    ok = (acum[ini + W] - acum[ini]) == 0                 # ninguna regla activa en los 60 registros
    N_VEN += len(ini)
    N_NORM += int(ok.sum())
    if ok.any():
        cob = np.zeros(len(t) + 1)                        # cuántas ventanas normales cubren cada registro
        np.add.at(cob, ini[ok], 1)
        np.add.at(cob, ini[ok] + W, -1)
        cob = np.cumsum(cob)[:-1]
        x = normalizar(t)
        S1 += cob @ x
        S2 += (x * cob[:, None]).T @ x
        N_REG += cob.sum()
    if (c + 1) % 500 == 0:
        print(f'  {c + 1:,} / {len(ID_TR):,} tramos')
if N_NORM == 0:
    raise SystemExit('No hay ventanas normales de entrenamiento.')
print(f'Ventanas de entrenamiento: {N_VEN:,} | normales {N_NORM:,} ({100 * N_NORM / N_VEN:.1f} %)'
      '   <- deben coincidir con las del 03b')

MU = S1 / N_REG                                           # centro: la operación normal
COV = S2 / N_REG - np.outer(MU, MU)
LAM, P = np.linalg.eigh(COV)
_o = np.argsort(LAM)[::-1]
LAM, P = np.clip(LAM[_o], 1e-12, None), P[:, _o]
P = P * np.where(P[np.abs(P).argmax(axis=0), np.arange(NV)] < 0, -1, 1)   # signo: la carga mayor positiva
VAR_ACUM = np.cumsum(LAM) / LAM.sum()
K_MAX = NV - 1
K_90 = int(min(np.searchsorted(VAR_ACUM, VAR_CLASICA) + 1, K_MAX))
K_AE = json.load(open(RUTA_CONFIG)).get('latent_dim') if os.path.exists(RUTA_CONFIG) else None
K_AE = int(min(max(K_AE, 1), K_MAX)) if K_AE else None

cargas = pd.DataFrame(P[:, :K_MAX], index=VARIABLES, columns=[f'PC{a + 1}' for a in range(K_MAX)])
_vt = pd.DataFrame({'variable': ['varianza_explicada_pct', 'varianza_acumulada_pct'],
                    **{f'PC{a + 1}': [100 * LAM[a] / LAM.sum(), 100 * VAR_ACUM[a]] for a in range(K_MAX)}})
guardar(pd.concat([cargas.reset_index().rename(columns={'index': 'variable'}), _vt], ignore_index=True)
        .round(4), '04h_pca_cargas.csv')
print('Varianza explicada (%): ' + ' | '.join(f'PC{a + 1} {100 * LAM[a] / LAM.sum():.1f}' for a in range(5)))
print(f'Componentes para {100 * VAR_CLASICA:.0f} % de la varianza: {K_90}'
      + (f' | k del autoencoder: {K_AE}' if K_AE else ''))
for a in range(min(3, K_MAX)):
    _c = cargas[f'PC{a + 1}']
    _top = _c.abs().sort_values(ascending=False).index[:4]
    print(f'  PC{a + 1}: ' + ', '.join(f'{v} {_c[v]:+.2f}' for v in _top))
print('Las cargas dicen qué variables se mueven juntas en la operación normal (04h_pca_cargas.csv).')


# =============================================================================
# 3. LAS VENTANAS DEL PASO 04 Y SUS ETIQUETAS
# =============================================================================
print('\n' + '=' * 72)
print('3. VENTANAS DE VALIDACIÓN Y PRUEBA (las mismas del paso 04)')
print('=' * 72)
_X, _ids = [], []
for i in ID_EV:
    x = normalizar(tramos[i])
    for v, s in enumerate(range(0, len(x) - W + 1, W)):  # sin traslape, como el 04
        _X.append(x[s:s + W])
        _ids.append((i, v))
_X = np.stack(_X)

m = pd.read_csv(ARCHIVO_ERROR)
if AE not in m.columns or not np.allclose(m[AE], m.error_reconstruccion):
    raise SystemExit(f'El paso 04 se corrió con otro punto de parada: córrelo con PUNTO_PARADA={PUNTO_PARADA!r}.')
_k = pd.DataFrame(_ids, columns=LLAVES)
_k['fila'] = np.arange(len(_ids))
_n0 = len(m)
m = m.merge(_k, on=LLAVES, how='inner', validate='one_to_one')
if len(m) != _n0:
    raise SystemExit(f'Solo {len(m):,} de {_n0:,} ventanas del paso 04 calzan: el limpio o el corte cambiaron. '
                     'Corre de nuevo el 04.')
X = _X[m.fila.to_numpy()]
del _X

EVAR = [f'evar_{v}' for v in VARIABLES]
RC = [r for r in IDS_REGLAS if r in pd.read_csv(ARCHIVO_VENTANAS, nrows=0).columns]
_ven = pd.read_csv(ARCHIVO_VENTANAS, usecols=LLAVES + RC).drop_duplicates(LLAVES)
m = m.drop(columns=[c for c in RC if c in m.columns]).merge(_ven, on=LLAVES, how='left', validate='one_to_one')
HAY_NIVEL = os.path.exists(ARCHIVO_NIVEL)
if HAY_NIVEL:
    m = m.drop(columns=[BANDA], errors='ignore').merge(
        pd.read_csv(ARCHIVO_NIVEL, usecols=LLAVES + [BANDA]).drop_duplicates(LLAVES), on=LLAVES,
        how='left', validate='one_to_one')
MEDIAS = [f'{v}_media' for v in VARIABLES]
HAY_Z = os.path.exists(ARCHIVO_FEATURES)
if HAY_Z:
    _f = pd.read_csv(ARCHIVO_FEATURES, usecols=lambda c: c in LLAVES + MEDIAS).drop_duplicates(LLAVES)
    HAY_Z = all(c in _f.columns for c in MEDIAS)
    if HAY_Z:
        m = m.drop(columns=[c for c in MEDIAS if c in m.columns]).merge(_f, on=LLAVES, how='left',
                                                                         validate='one_to_one')

ACT = m[RC].fillna(0).to_numpy() > 0
anomala = (m.n_reglas_activas > 0).to_numpy() if 'n_reglas_activas' in m.columns else ACT.any(axis=1)
normal = ~anomala
conj = m.conjunto.to_numpy()
va, te = conj == 'val', conj == 'test'
if not (va & normal).any() or not (va & anomala).any():
    raise SystemExit('Validación necesita ventanas con y sin alarma.')
grave = (m[BANDA] == GRAVE).to_numpy() if HAY_NIVEL else np.zeros(len(m), dtype=bool)
EV_AE = m[EVAR].to_numpy(dtype=float)
ERR_AE = m.error_reconstruccion.to_numpy(dtype=float)
R2V = np.array([[v in variables_de(r) for v in VARIABLES] for r in RC], dtype=int)
VARS_OK = (ACT.astype(int) @ R2V) > 0                     # variables que vigilan las reglas que sonaron
print(f'Ventanas: validación {va.sum():,} ({(va & anomala).sum():,} con alarma) | '
      f'prueba {te.sum():,} ({(te & anomala).sum():,} con alarma)')

# remuestreos por bloques de 6 h, los mismos para todas las comparaciones (pareadas)
_rng = np.random.default_rng(SEED)
_bloque = m.tramo_id.to_numpy().astype(np.int64) * 100000 + m.ventana_id.to_numpy() // BLOQUE


def remuestreos(sel):
    idx = np.flatnonzero(sel)
    grupos = pd.Series(np.arange(len(idx))).groupby(_bloque[idx]).indices
    claves = np.array(list(grupos))
    return [idx[np.concatenate([grupos[c] for c in _rng.choice(claves, len(claves))])] for _ in range(N_BOOT)]


MUESTRAS = {'val': remuestreos(va), 'test': remuestreos(te)}
SEL = {'val': va, 'test': te}


def ic(fun, cj):
    """Valor con todas las ventanas del conjunto e IC 95 % por bloques de 6 h."""
    v = [fun(s) for s in MUESTRAS[cj]]
    v = [x for x in v if np.isfinite(x)]
    lo, hi = np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)
    return fun(np.flatnonzero(SEL[cj])), lo, hi


def localizar(E):
    """Criterio del 04c: la variable que más se pasa de su umbral P95. Ventaja por ventana = acierto − azar."""
    ref = np.percentile(E[va & normal], PERCENTIL, axis=0)
    top = np.argmax(E - ref, axis=1)
    dist0 = np.bincount(top[va & normal], minlength=NV) / int((va & normal).sum())
    acierto = 100.0 * VARS_OK[np.arange(len(E)), top]
    azar = 100.0 * (VARS_OK * dist0).sum(axis=1)
    return top, acierto, azar


# =============================================================================
# 4. BARRIDO DE k Y ELECCIÓN EN VALIDACIÓN
# =============================================================================
print('\n' + '=' * 72)
print('4. BARRIDO DE k (número de componentes): se elige en validación con la regla del 03a')
print('=' * 72)
XC = X.reshape(-1, NV) - MU                               # (ventanas · 60, 14), centrado en lo normal
TS = XC @ P                                               # puntajes de las componentes


def pca_k(k):
    """Contribución de cada variable al SPE (promedio de la ventana) y T² de la ventana."""
    R = XC - TS[:, :k] @ P[:, :k].T
    C = (R ** 2).reshape(len(X), W, NV).mean(axis=1)
    t2 = ((TS[:, :k] ** 2) / LAM[:k]).sum(axis=1).reshape(len(X), W).mean(axis=1)
    return C, t2


filas_k, CACHE = [], {}
for k in range(1, K_MAX + 1):
    C, t2 = pca_k(k)
    spe = C.sum(axis=1)
    top, acierto, azar = localizar(C)
    CACHE[k] = (C, spe, t2, acierto - azar)
    filas_k.append({'k': k, 'varianza_acumulada_pct': 100 * VAR_ACUM[k - 1],
                    'auc_spe_val': auc(anomala[va], spe[va]), 'auc_spe_test': auc(anomala[te], spe[te]),
                    'auc_t2_val': auc(anomala[va], t2[va]), 'auc_t2_test': auc(anomala[te], t2[te]),
                    'ventaja_val': float(np.mean((acierto - azar)[va & anomala])),
                    'ventaja_test': float(np.mean((acierto - azar)[te & anomala]))})
barrido = pd.DataFrame(filas_k)

# regla del 03a: 1) mayor AUC; 2) entre empatados, mayor ventaja; 3) entre empatados, el menor k
_y = anomala
k1 = int(barrido.loc[barrido.auc_spe_val.idxmax(), 'k'])
_a = va & anomala
ic_auc = {k1: (0.0, 0.0)}
for k in barrido.k:
    if k != k1:
        _, lo, hi = ic(lambda s, a=CACHE[k][1], b=CACHE[k1][1]: auc(_y[s], a[s]) - auc(_y[s], b[s]), 'val')
        ic_auc[k] = (lo, hi)
barrido['dif_auc'] = barrido.auc_spe_val - float(barrido.auc_spe_val[barrido.k == k1].iloc[0])
barrido['ic_auc_inf'] = [ic_auc[k][0] if k != k1 else np.nan for k in barrido.k]
barrido['ic_auc_sup'] = [ic_auc[k][1] if k != k1 else np.nan for k in barrido.k]
barrido['empate_auc'] = [k == k1 or ic_auc[k][0] <= 0 <= ic_auc[k][1] for k in barrido.k]
_t1 = barrido[barrido.empate_auc]
k2 = int(_t1.loc[_t1.ventaja_val.idxmax(), 'k'])
ic_ven = {}
for k in _t1.k:
    if k != k2:
        _, lo, hi = ic(lambda s, a=CACHE[k][3], b=CACHE[k2][3]: float(np.mean((a - b)[s[_a[s]]]))
                       if _a[s].any() else np.nan, 'val')
        ic_ven[k] = (lo, hi)
barrido['dif_ventaja'] = barrido.ventaja_val - float(barrido.ventaja_val[barrido.k == k2].iloc[0])
barrido['ic_ven_inf'] = [ic_ven[k][0] if k in ic_ven else np.nan for k in barrido.k]
barrido['ic_ven_sup'] = [ic_ven[k][1] if k in ic_ven else np.nan for k in barrido.k]
barrido['empate_ventaja'] = [k == k2 or (k in ic_ven and ic_ven[k][0] <= 0 <= ic_ven[k][1]) for k in barrido.k]
K_EL = int(barrido[barrido.empate_ventaja].k.min())
barrido['elegido'] = barrido.k == K_EL
guardar(barrido.round(4), '04h_pca_barrido.csv')
print(barrido[['k', 'varianza_acumulada_pct', 'auc_spe_val', 'auc_t2_val', 'ventaja_val', 'empate_auc',
               'empate_ventaja', 'elegido']].round(3).to_string(index=False))
print(f'\nk elegido para el PCA: {K_EL} ({100 * VAR_ACUM[K_EL - 1]:.1f} % de la varianza)')


# =============================================================================
# 5. DETECCIÓN: AUTOENCODER, PCA Y z DE LAS MEDIAS CON EL MISMO UMBRAL
# =============================================================================
print('\n' + '=' * 72)
print(f'5. DETECCIÓN CON EL UMBRAL P{PERCENTIL} (ventanas sin alarma de validación)')
print('=' * 72)
C_EL, SPE_EL, T2_EL, VEN_EL = CACHE[K_EL]
PUNTAJES = {'autoencoder (MSE)': ERR_AE,
            f'PCA SPE, k = {K_EL} (elegido)': SPE_EL,
            f'PCA T², k = {K_EL} (elegido)': T2_EL}
_vistos = {K_EL}
for k, nom in [(K_AE, 'el del autoencoder'), (K_90, f'{100 * VAR_CLASICA:.0f} % de varianza')]:
    if k and k not in _vistos:
        PUNTAJES[f'PCA SPE, k = {k} ({nom})'] = CACHE[k][1]
        _vistos.add(k)
if HAY_Z:
    _med = m[MEDIAS].to_numpy(dtype=float)
    _mu = np.nanmean(_med[va & normal], axis=0)
    _sd = np.nanstd(_med[va & normal], axis=0, ddof=1)
    PUNTAJES['z de las medias (sin modelo)'] = np.nan_to_num(
        np.abs((_med - _mu) / np.where(_sd > 0, _sd, 1.0)), nan=0.0).max(axis=1)

filas_d, MARCA = [], {}
for nom, s in PUNTAJES.items():
    u = float(np.percentile(s[va & normal], PERCENTIL))
    marca = s > u
    MARCA[nom] = marca
    for cj, c in [('val', va), ('test', te)]:
        vp, fp = int((marca & c & anomala).sum()), int((marca & c & normal).sum())
        fn = int((~marca & c & anomala).sum())
        filas_d.append({'metodo': nom, 'conjunto': cj, 'ventanas': int(c.sum()), 'auc': auc(anomala[c], s[c]),
                        'especificidad': 1 - fp / max(int((c & normal).sum()), 1),
                        'sensibilidad': vp / max(int((c & anomala).sum()), 1),
                        'precision': vp / max(vp + fp, 1), 'f1': 2 * vp / max(2 * vp + fp + fn, 1),
                        'graves_detectadas': (float(marca[c & grave].mean()) if (c & grave).any() else np.nan),
                        'umbral': u})
deteccion = pd.DataFrame(filas_d)
guardar(deteccion.round(4), '04h_pca_deteccion.csv')
pd.set_option('display.width', 220)
print(deteccion.drop(columns=['umbral']).round(3).to_string(index=False))

NOM_SPE, NOM_T2 = f'PCA SPE, k = {K_EL} (elegido)', f'PCA T², k = {K_EL} (elegido)'
filas_c = []
pares = [('autoencoder (MSE)', NOM_SPE), ('autoencoder (MSE)', NOM_T2)]
if HAY_Z:
    pares += [(NOM_SPE, 'z de las medias (sin modelo)')]
for a_, b_ in pares:
    for cj in ('val', 'test'):
        dif, lo, hi = ic(lambda s, a=PUNTAJES[a_], b=PUNTAJES[b_]: auc(_y[s], a[s]) - auc(_y[s], b[s]), cj)
        filas_c.append({'medida': 'AUC de detección', 'comparacion': f'{a_} − {b_}', 'conjunto': cj,
                        'grupo': 'todas las ventanas', 'diferencia': dif, 'ic_inf': lo, 'ic_sup': hi})


# =============================================================================
# 6. LOCALIZACIÓN: CONTRIBUCIONES AL SPE CONTRA EL ERROR POR VARIABLE DEL AUTOENCODER
# =============================================================================
print('\n' + '=' * 72)
print('6. LOCALIZACIÓN: ¿LA VARIABLE QUE MÁS SE PASA ES LA DE LA REGLA QUE SONÓ?')
print('=' * 72)
TOP_AE, AC_AE, AZ_AE = localizar(EV_AE)
TOP_PCA, AC_PCA, AZ_PCA = localizar(C_EL)
GRUPOS = {'todas las anómalas': anomala, 'marcadas por el autoencoder': anomala & MARCA['autoencoder (MSE)'],
          'marcadas por el PCA': anomala & MARCA[NOM_SPE]}
filas_l = []
for cj, c in [('val', va), ('test', te)]:
    for gn, g in GRUPOS.items():
        s = c & g
        if s.sum() < 10:
            continue
        for met, ac, az in [('autoencoder', AC_AE, AZ_AE), (f'PCA, k = {K_EL}', AC_PCA, AZ_PCA)]:
            filas_l.append({'metodo': met, 'conjunto': cj, 'grupo': gn, 'ventanas': int(s.sum()),
                            'acierta_pct': float(ac[s].mean()), 'azar_pct': float(az[s].mean()),
                            'ventaja': float((ac - az)[s].mean())})
        if gn != 'marcadas por el PCA':                     # mismas ventanas para los dos: diferencia pareada
            dv = (AC_AE - AZ_AE) - (AC_PCA - AZ_PCA)
            dif, lo, hi = ic(lambda x, gg=g: float(np.mean(dv[x[gg[x]]])) if gg[x].any() else np.nan, cj)
            filas_c.append({'medida': 'ventaja de localización (puntos)',
                            'comparacion': f'autoencoder − PCA, k = {K_EL}', 'conjunto': cj, 'grupo': gn,
                            'diferencia': dif, 'ic_inf': lo, 'ic_sup': hi})
loc = pd.DataFrame(filas_l)
guardar(loc.round(2), '04h_pca_localizacion.csv')
print(loc.round(1).to_string(index=False))
comp = pd.DataFrame(filas_c)
guardar(comp.round(4), '04h_pca_comparaciones.csv')
print('\nComparaciones (diferencia e IC 95 % por bloques de 6 h; si el IC incluye el 0, empatan):')
print(comp.round(3).to_string(index=False))


# =============================================================================
# 7. DELTA POR VARIABLE: ¿EXPLICAN LAS MISMAS VARIABLES?
# =============================================================================
filas_v = []
for met, E in [('autoencoder', EV_AE), (f'PCA, k = {K_EL}', C_EL)]:
    for cj, c in [('val', va), ('test', te)]:
        if not (c & normal).any() or not (c & anomala).any():
            continue
        d = E[c & anomala].mean(axis=0) - E[c & normal].mean(axis=0)
        for j, v in enumerate(VARIABLES):
            filas_v.append({'metodo': met, 'conjunto': cj, 'variable': v, 'delta': d[j],
                            'pct_delta': 100 * d[j] / d.sum() if d.sum() > 0 else np.nan})
delta = pd.DataFrame(filas_v)
guardar(delta.round(6), '04h_pca_delta.csv')
print('\nVariables con más % del Δ (anómalas − normales):')
for (met, cj), t in delta.groupby(['metodo', 'conjunto'], sort=False):
    t = t.sort_values('pct_delta', ascending=False).head(3)
    print(f'  {met:<14} {cj:<5} ' + ' · '.join(f'{r.variable} {r.pct_delta:.0f} %' for r in t.itertuples()))

out = m[LLAVES + ['conjunto']].copy()
out['anomala'] = anomala.astype(int)
out['spe_pca'], out['t2_pca'] = SPE_EL, T2_EL
out['marcada_pca'] = MARCA[NOM_SPE].astype(int)
out['variable_pca'] = np.where(MARCA[NOM_SPE], np.array(VARIABLES)[TOP_PCA], '')
out['error_ae'] = ERR_AE
out['marcada_ae'] = MARCA['autoencoder (MSE)'].astype(int)
out['variable_ae'] = np.where(MARCA['autoencoder (MSE)'], np.array(VARIABLES)[TOP_AE], '')
guardar(out, '04h_pca_ventanas.csv')


# =============================================================================
# 8. FIGURA Y LECTURA
# =============================================================================
def etiqueta_corta(n):
    """Rótulo de dos líneas para el eje del gráfico."""
    if n.startswith('autoencoder'):
        return 'autoencoder'
    if n.startswith('z de las medias'):
        return 'z de las\nmedias'
    met, resto = n.split(', ', 1)
    nota = resto.split('(')[1].rstrip(')') if '(' in resto else ''
    nota = {'el del autoencoder': 'del autoencoder', f'{100 * VAR_CLASICA:.0f} % de varianza':
            f'{100 * VAR_CLASICA:.0f} % varianza'}.get(nota, nota)
    return f"{met}\n{resto.split(' (')[0]}" + (f'\n({nota})' if nota else '')


fig, ejes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={'width_ratios': [1.1, 1.6, 0.9]})
ax = ejes[0]
ax.plot(barrido.k, barrido.auc_spe_val, marker='o', color='#2a78d6', lw=2, label='PCA, SPE')
ax.plot(barrido.k, barrido.auc_t2_val, marker='s', color='#eb6834', lw=1.5, label='PCA, T²')
ax.axhline(auc(anomala[va], ERR_AE[va]), color='#1baf7a', lw=2, ls='--', label='autoencoder')
ax.scatter([K_EL], barrido.auc_spe_val[barrido.k == K_EL], s=220, marker='*', color='k', zorder=5,
           label='k elegido')
ax.set_xlabel('número de componentes (k)')
ax.set_ylabel('AUC de validación')
ax2 = ax.twinx()
ax2.plot(barrido.k, barrido.varianza_acumulada_pct, color='#9b9a96', lw=1, ls=':')
ax2.axhline(100 * VAR_CLASICA, color='#9b9a96', lw=0.8, ls=':')
ax2.set_ylabel('varianza acumulada (%)', color='#6b6a66')
ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc='center left')
ax.set_title('Elección de k (validación)', loc='left', fontsize=11)
ax = ejes[1]
_noms = list(PUNTAJES)
_x = np.arange(len(_noms))
for o, (cj, col) in enumerate([('val', '#2a78d6'), ('test', '#eda100')]):
    _v = [deteccion[(deteccion.metodo == n) & (deteccion.conjunto == cj)].auc.iloc[0] for n in _noms]
    ax.bar(_x + (o - 0.5) * 0.38, _v, 0.38, color=col, label='validación' if cj == 'val' else 'prueba')
    for xi, vi in zip(_x + (o - 0.5) * 0.38, _v):
        ax.text(xi, vi + 0.01, f'{vi:.2f}', ha='center', fontsize=7)
ax.axhline(0.5, color='k', lw=0.8, ls=':')
ax.set_xticks(_x)
ax.set_xticklabels([etiqueta_corta(n) for n in _noms], fontsize=7.5)
ax.set_ylim(0.3, 1.05)
ax.set_ylabel('AUC')
ax.legend(fontsize=8, frameon=False)
ax.set_title('Detección: AUC', loc='left', fontsize=11)
ax = ejes[2]
_l = loc[loc.grupo == 'todas las anómalas']
for o, met in enumerate(['autoencoder', f'PCA, k = {K_EL}']):
    _v = [_l[(_l.metodo == met) & (_l.conjunto == cj)].ventaja.iloc[0] if len(_l[(_l.metodo == met)
          & (_l.conjunto == cj)]) else np.nan for cj in ('val', 'test')]
    ax.bar(np.arange(2) + (o - 0.5) * 0.38, _v, 0.38, color=['#1baf7a', '#2a78d6'][o], label=met)
    for xi, vi in zip(np.arange(2) + (o - 0.5) * 0.38, _v):
        if np.isfinite(vi):
            ax.text(xi, vi + 0.5, f'{vi:.0f}', ha='center', fontsize=8)
ax.set_xticks(np.arange(2))
ax.set_xticklabels(['validación', 'prueba'])
ax.set_ylabel('ventaja sobre el azar (puntos)')
ax.legend(fontsize=8, frameon=False)
ax.set_title('Localización (todas las ventanas con alarma)', loc='left', fontsize=11)
for a in ejes:
    for s_ in ('top', 'right'):
        if a is not ejes[0] or s_ == 'top':
            a.spines[s_].set_visible(False)
fig.suptitle(f'{EQUIPO}: autoencoder temporal contra PCA (SPE y T²), el monitoreo clásico de procesos',
             x=0.01, ha='left', fontsize=12)
fig.tight_layout()
fig.savefig(salida('04h_pca.png'), dpi=150)
plt.close(fig)


def lectura(comparacion, grupo, cj):
    r = comp[(comp.comparacion == comparacion) & (comp.conjunto == cj) & (comp.grupo == grupo)]
    if r.empty:
        return 'sin datos'
    r = r.iloc[0]
    quien = ('el autoencoder es mejor' if r.ic_inf > 0 else 'el PCA es mejor' if r.ic_sup < 0 else 'empatan')
    return f'{quien} (diferencia {r.diferencia:+.3f}, IC {r.ic_inf:+.3f} a {r.ic_sup:+.3f})'


print('\n' + '=' * 72)
print('LECTURA (se decide en validación; prueba solo se informa)')
print('=' * 72)
_cd, _cl = f'autoencoder (MSE) − {NOM_SPE}', f'autoencoder − PCA, k = {K_EL}'
print(f'Detección (AUC), validación: {lectura(_cd, "todas las ventanas", "val")}')
print(f'Detección (AUC), prueba:     {lectura(_cd, "todas las ventanas", "test")}')
for _g in ('todas las anómalas', 'marcadas por el autoencoder'):
    print(f'Localización, {_g}, validación: {lectura(_cl, _g, "val")}')
    print(f'Localización, {_g}, prueba:     {lectura(_cl, _g, "test")}')
print('El PCA es lineal y no mira el tiempo; el autoencoder es no lineal y relaciona los 60 instantes. Si el PCA')
print('empata, lo no lineal y temporal no agrega en esa tarea; si el autoencoder gana, ese es su aporte.')
print(f'\nGuardado en {RUTA_EQ}: 04h_pca_barrido.csv, 04h_pca_cargas.csv, 04h_pca_deteccion.csv, '
      '04h_pca_comparaciones.csv, 04h_pca_localizacion.csv, 04h_pca_delta.csv, 04h_pca_ventanas.csv, 04h_pca.png')
