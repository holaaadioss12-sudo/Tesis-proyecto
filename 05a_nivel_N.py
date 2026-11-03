# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 05a — NIVEL DE ANOMALÍA N: CRITICIDAD Y FRECUENCIA, CORTES Y AGRUPAMIENTO
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
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score,
                             f1_score, balanced_accuracy_score)
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight

import criticidad

try:
    from xgboost import XGBClassifier
    HAY_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAY_XGB = False


# =============================================================================
# 0. CONSTANTES QUE DECIDEN
# =============================================================================
SEED = 42
W = 60
LLAVES = ['tramo_id', 'ventana_id']
AE = 'err_AE_normales'          # Checkpoint de pérdida mínima
PERCENTIL_UMBRAL = 95           # umbral del autoencoder, normales de validación

ESCALAS = {'lineal': {c: 4 - c for c in (1, 2, 3)},
           'empinada': {c: 3 ** (3 - c) for c in (1, 2, 3)}}
ESCALA_PRINCIPAL = 'empinada'
CORTE_PRINCIPAL = 'natural'
NIVELES = ['1_leve', '2_media', '3_grave']

K_RANGO = range(2, 7)           # "pueden ser más, quintiles serían cinco"
DBSCAN_CUANTILES_EPS = [0.10, 0.25, 0.50, 0.75]
PCA_VARIANZA = 0.80             # componentes hasta explicar el 80 % (máx. 10)
PCA_MAX = 10
MAX_SILUETA = 5000              # submuestra para la silueta (es O(n^2))
MIN_CLASE_XGB = 10

ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_ACTIVOS = f'{RUTA_EQ}/{EQUIPO}_activos_por_ventana_W{W}.csv'
ARCHIVO_EVENTOS = f'{RUTA_EQ}/{EQUIPO}_eventos_por_ventana_W{W}.csv'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
ARCHIVO_W = f'{RUTA_EQ}/{EQUIPO}_w_sensores_W{W}.csv'

NO_FEATURE = ['tramo_id', 'ventana_id', 'timestamp_inicio', 'timestamp_fin',
              'etiqueta_ventana', 'etiqueta_tfidf', 'banda_severidad',
              'total_alarmas_ventana', 'total_eventos_ventana',
              'n_reglas_activas', 'concentracion']


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


# =============================================================================
# 1. CARGA
# =============================================================================
print('=' * 72)
print(f'PASO 05a — NIVEL DE ANOMALÍA N (W = {W})')
print('=' * 72)

corte = json.load(open(ARCHIVO_CORTE))
T_TRAIN, T_VAL = pd.Timestamp(corte['T_train']), pd.Timestamp(corte['T_val'])
print(f'Corte único: T_train = {T_TRAIN} | T_val = {T_VAL}')

ventanas = pd.read_csv(ARCHIVO_VENTANAS, parse_dates=['timestamp_inicio',
                                                      'timestamp_fin'])
ids_reglas = [c for c in ventanas.columns if c in criticidad.CRITICIDAD]
print(f'Ventanas: {len(ventanas):,} | reglas: {len(ids_reglas)}   <- esperado: 31')

_fuente = next((f for f in [ARCHIVO_ACTIVOS, ARCHIVO_EVENTOS]
                if os.path.exists(f)), None)
if _fuente:
    _ev = pd.read_csv(_fuente)
    conteo = (ventanas[LLAVES]
              .merge(_ev, on=LLAVES, how='left', validate='one_to_one')
              .reindex(columns=LLAVES + ids_reglas).fillna(0))
    if _fuente == ARCHIVO_ACTIVOS:
        CONTEO = 'activos'
        print('n_j = episodios ACTIVOS en la ventana (paso 02).')
    else:
        CONTEO = 'iniciados'
        print('\n' + '!' * 72)
        print(f'No está {os.path.basename(ARCHIVO_ACTIVOS)}: n_j = eventos que')
        print('ARRANCAN en la ventana. Las ventanas del medio de una alarma larga')
        print('quedan con N = 0. Sirve para mirar, NO para la memoria: corre de')
        print('nuevo 02_reglas_ventanas.py.')
        print('!' * 72)
else:
    conteo = ventanas[LLAVES + ids_reglas].copy()
    CONTEO = 'muestras'
    print('\n' + '!' * 72)
    print(f'No está {os.path.basename(ARCHIVO_EVENTOS)}: el paso 02 se corrió')
    print('con una versión antigua. N se calcula con MUESTRAS de 10 s en alarma.')
    print('Una regla de estado como A4 suma 60 por ventana y domina N: estos')
    print('números sirven para mirar, NO para la memoria. Corre el paso 02.')
    print('!' * 72)
C = conteo[ids_reglas].astype(float).reset_index(drop=True)
ventanas = ventanas.reset_index(drop=True)

if 'total_alarmas_ventana' in ventanas.columns:
    ventanas['sin_alarma'] = ventanas.total_alarmas_ventana == 0
else:
    ventanas['sin_alarma'] = ventanas[ids_reglas].sum(axis=1) == 0

# banda_w y el índice W por sensor: en ventanas o en su propio archivo
COLS_W = [f'w_{v}' for v in ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
                             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']]
if 'banda_w' not in ventanas.columns and os.path.exists(ARCHIVO_W):
    _w = pd.read_csv(ARCHIVO_W)
    _cols = [c for c in LLAVES + ['banda_w', 'w_max', 'sensor_origen'] + COLS_W
             if c in _w.columns]
    ventanas = ventanas.merge(_w[_cols], on=LLAVES, how='left',
                              validate='one_to_one')
COLS_W = [c for c in COLS_W if c in ventanas.columns]
print(f'Índice W por sensor disponible: {"sí" if COLS_W else "no"}')


def conjunto_por_fecha(ts):
    ts = pd.to_datetime(ts)
    return np.where(ts <= T_TRAIN, 'train', np.where(ts <= T_VAL, 'val', 'test'))


ventanas['conjunto_fecha'] = conjunto_por_fecha(ventanas.timestamp_inicio)


# =============================================================================
# 2. EL NIVEL N DE CADA VENTANA
# =============================================================================
print('\n' + '=' * 72)
print('2. EL NIVEL N = suma de peso x activaciones')
print('=' * 72)

FAMILIAS = {k: [r for r in v if r in ids_reglas]
            for k, v in criticidad.familias_activas().items()}
_vistas = [r for m in FAMILIAS.values() for r in m]
assert len(_vistas) == len(set(_vistas)), 'Una regla aparece en dos familias.'


def nivel_N(C, pesos, colapsar=True):
    w = {r: pesos[criticidad.CRITICIDAD[r]] for r in ids_reglas}
    N = np.zeros(len(C))
    en_familia = set()
    if colapsar:
        for miembros in FAMILIAS.values():
            if not miembros:
                continue
            en_familia |= set(miembros)
            for i, r in enumerate(miembros):
                sig = C[miembros[i + 1]].to_numpy() if i + 1 < len(miembros) else 0
                N += w[r] * np.clip(C[r].to_numpy() - sig, 0, None)
    for r in ids_reglas:
        if r not in en_familia:
            N += w[r] * C[r].to_numpy()
    return N


for esc, pesos in ESCALAS.items():
    ventanas[f'N_{esc}'] = nivel_N(C, pesos)
ventanas['N_lineal_sin_colapso'] = nivel_N(C, ESCALAS['lineal'], colapsar=False)
_crit = np.full(len(C), 4)
for r in ids_reglas:
    _crit = np.where(C[r].to_numpy() > 0,
                     np.minimum(_crit, criticidad.CRITICIDAD[r]), _crit)
ventanas['criticidad_min'] = _crit

_infl = 100 * (ventanas.N_lineal_sin_colapso.sum() / max(ventanas.N_lineal.sum(), 1) - 1)
print(f'Conteo usado: {CONTEO}')
print(f'Inflación de N si no se colapsaran las familias: {_infl:.1f} %')
print(f'Ventanas con N = 0: {100 * (ventanas.N_lineal == 0).mean():.1f} %')
_n0_con_alarma = int(((ventanas.N_lineal == 0) & ~ventanas.sin_alarma).sum())
print(f'Ventanas con N = 0 pero alguna alarma activa: {_n0_con_alarma:,}   '
      f'<- con activos debe ser 0')
print(ventanas.loc[ventanas.N_lineal > 0, ['N_lineal', 'N_empinada']]
      .describe(percentiles=[.25, .5, .75, .9, .99]).round(1).to_string())

# Ejemplo con estos pesos
print('\nEjemplo: una regla de criticidad 2 activada 20 veces contra una de')
print('criticidad 1 activada una vez:')
for esc, p in ESCALAS.items():
    print(f'  escala {esc:<9} 20 x {p[2]} = {20 * p[2]:>3}   contra   1 x {p[1]} = {p[1]}')


# =============================================================================
# 3. LOS CORTES: SALEN DE ENTRENAMIENTO Y SE APLICAN FIJOS
# =============================================================================
print('\n' + '=' * 72)
print('3. CORTES DE LEVE / MEDIA / GRAVE (ventanas anómalas de entrenamiento)')
print('=' * 72)


def cortes_naturales(x):
    """K-means de 3 grupos en una dimensión, sobre log(1 + N)."""
    z = np.log1p(x).reshape(-1, 1)
    km = KMeans(n_clusters=3, n_init=20, random_state=SEED).fit(z)
    c = np.sort(km.cluster_centers_.ravel())
    return list(np.expm1((c[:-1] + c[1:]) / 2))


def bandas(N, cortes):
    b = np.full(len(N), '0_normal', dtype=object)
    pos = N > 0
    idx = np.digitize(N[pos], cortes, right=True)      # 0, 1, 2
    b[pos] = np.array(NIVELES, dtype=object)[idx]
    return b


filas_cortes = []
for esc in ESCALAS:
    N = ventanas[f'N_{esc}'].to_numpy()
    pos_tr = N[(ventanas.conjunto_fecha == 'train').to_numpy() & (N > 0)]
    if len(pos_tr) < 30:
        raise SystemExit(f'Sólo {len(pos_tr)} ventanas anómalas en entrenamiento.')
    metodos = {'natural': cortes_naturales(pos_tr),
               'terciles': list(np.percentile(pos_tr, [100 / 3, 200 / 3]))}
    for met, ct in metodos.items():
        col = f'banda_N_{esc}_{met}'
        ventanas[col] = bandas(N, ct)
        fila = {'escala': esc, 'metodo': met, 'conteo': CONTEO,
                'corte_leve_media': round(ct[0], 2),
                'corte_media_grave': round(ct[1], 2)}
        for cj in ['train', 'val', 'test']:
            vc = (ventanas.loc[ventanas.conjunto_fecha == cj, col]
                  .value_counts(normalize=True))
            for nv in ['0_normal'] + NIVELES:
                fila[f'{cj}_{nv}_pct'] = round(100 * vc.get(nv, 0), 1)
        filas_cortes.append(fila)
cortes = pd.DataFrame(filas_cortes)
print(cortes[['escala', 'metodo', 'corte_leve_media', 'corte_media_grave',
              'train_1_leve_pct', 'train_2_media_pct', 'train_3_grave_pct',
              'test_1_leve_pct', 'test_2_media_pct', 'test_3_grave_pct']]
      .to_string(index=False))
if (cortes.corte_leve_media == cortes.corte_media_grave).any():
    print('  >> Hay cortes repetidos: N toma pocos valores y dos bandas se juntan.')

# Dónde cae una sola alarma grave contra varias menores
print('\n¿Dónde cae una ventana con UNA alarma de criticidad 1 (p. ej. A5) y una')
print('con TRES alarmas de criticidad 3? (cortes naturales de cada escala)')
for esc, p in ESCALAS.items():
    _ct = cortes[(cortes.escala == esc) & (cortes.metodo == CORTE_PRINCIPAL)].iloc[0]
    _ct = [_ct.corte_leve_media, _ct.corte_media_grave]
    _b1 = bandas(np.array([float(p[1])]), _ct)[0]
    _b3 = bandas(np.array([3.0 * p[3]]), _ct)[0]
    print(f'  {esc:<9} una de crit. 1: N = {p[1]:<2} -> {_b1:<8} | '
          f'tres de crit. 3: N = {3 * p[3]:<2} -> {_b3}')

BANDA = f'banda_N_{ESCALA_PRINCIPAL}_{CORTE_PRINCIPAL}'
N_PRINC = f'N_{ESCALA_PRINCIPAL}'
print(f'\nLa banda que se defiende: {BANDA}')
if 'banda_severidad' in ventanas.columns:
    print('\nContra la banda por número de reglas (todas las ventanas):')
    print(pd.crosstab(ventanas.banda_severidad, ventanas[BANDA]).to_string())
print('\nContra la criticidad de la regla más grave (4 = ninguna):')
print(pd.crosstab(ventanas.criticidad_min, ventanas[BANDA]).to_string())


# =============================================================================
# 4. VALIDACIÓN EXTERNA: ¿EL ERROR DEL AUTOENCODER ORDENA LAS BANDAS?
# =============================================================================
print('\n' + '=' * 72)
print('4. EL ERROR DEL AUTOENCODER CONTRA LAS BANDAS (validación y prueba)')
print('=' * 72)

err = pd.read_csv(ARCHIVO_ERROR)
if AE not in err.columns:
    raise SystemExit(f'El archivo del paso 04 no trae {AE}.')
d = err[LLAVES + ['conjunto', AE]].merge(ventanas, on=LLAVES, how='inner',
                                         validate='one_to_one')
d = d.rename(columns={AE: 'error_ae'})
assert len(d) == len(err), 'El cruce con el paso 04 perdió ventanas.'

DATOS_ERR = {'validación': d.conjunto == 'val', 'prueba': d.conjunto == 'test',
             'validación + prueba': d.conjunto.isin(['val', 'test'])}
filas_err = []
for col in [c for c in ventanas.columns if c.startswith('banda_N_')] + \
        [c for c in ['banda_severidad', 'banda_w'] if c in d.columns]:
    for datos, m in DATOS_ERR.items():
        g = d[m].groupby(col).error_ae.median()
        if '0_normal' not in g.index:
            continue
        veces = (g / g['0_normal']).round(1)
        orden = [x for x in ['0_normal'] + NIVELES if x in veces.index]
        mono = bool(np.all(np.diff(veces[orden].to_numpy()) > 0))
        cuenta = d[m][col].value_counts()
        fila = {'escala': col, 'datos': datos, 'monotona': mono}
        fila.update({f'veces_{k}': veces.get(k, np.nan) for k in orden})
        fila.update({f'n_{k}': int(cuenta.get(k, 0)) for k in orden})
        filas_err.append(fila)
bandas_err = pd.DataFrame(filas_err)
_cn = [c for c in bandas_err if c.startswith('n_')]
bandas_err[_cn] = bandas_err[_cn].astype('Int64')
print('Error mediano de cada banda, en veces el de la banda normal:')
for datos in DATOS_ERR:
    print(f'\n  {datos}:')
    print(bandas_err[bandas_err.datos == datos]
          .drop(columns=['datos'] + [c for c in bandas_err if c.startswith('n_')])
          .to_string(index=False))
print(f'\n¿Qué escala ordena el error en VALIDACIÓN? (corte {CORTE_PRINCIPAL})')
for esc in ESCALAS:
    _f = bandas_err[(bandas_err.escala == f'banda_N_{esc}_{CORTE_PRINCIPAL}')
                    & (bandas_err.datos == 'validación')]
    if len(_f):
        print(f'  {esc:<9} {"sí" if bool(_f.monotona.iloc[0]) else "NO"}'
              f'{"   <- principal" if esc == ESCALA_PRINCIPAL else ""}')

print('\nSpearman con el error del autoencoder:')
for col in dict.fromkeys([N_PRINC, 'N_lineal', 'N_empinada', 'n_reglas_activas',
                          'total_eventos_ventana', 'w_max']):
    if col in d.columns:
        rho = spearmanr(d[col], d.error_ae).correlation
        print(f'  {col:<24} {rho:+.3f}')


# =============================================================================
# 5. ANÓMALAS PARA EL AUTOENCODER CON N = 0: LA LISTA PARA REVISAR
# =============================================================================
print('\n' + '=' * 72)
print('5. EL AUTOENCODER DICE ANÓMALA Y N = 0')
print('=' * 72)

d_va, d_te = d[d.conjunto == 'val'], d[d.conjunto == 'test']
u = float(np.percentile(d_va.loc[d_va.sin_alarma, 'error_ae'], PERCENTIL_UMBRAL))
d['marcada'] = d.error_ae > u
print(f'Umbral del autoencoder (P{PERCENTIL_UMBRAL} de las ventanas sin alarma de '
      f'validación): {u:.5f}   <- debe ser el mismo del paso 04')

sin_alarma = d[d.marcada & d.sin_alarma].copy()
for cj in ['val', 'test']:
    m = d[(d.conjunto == cj) & d.marcada]
    n0 = int(m.sin_alarma.sum())
    n_cj = int(((d.conjunto == cj) & d.sin_alarma).sum())
    print(f'  {cj:<5} marcadas {len(m):>6,} | sin ninguna alarma: {n0:>5,} '
          f'({100 * n0 / max(len(m), 1):.1f} % de las marcadas; '
          f'{100 * n0 / max(n_cj, 1):.1f} % de las ventanas sin alarma)')
sin_alarma['mes'] = pd.to_datetime(sin_alarma.timestamp_inicio).dt.to_period('M')
print('\nPor mes:')
print(sin_alarma.groupby(['conjunto', 'mes']).size().to_string())
if 'sensor_origen' in sin_alarma.columns:
    print('\nSensor que más se aleja de lo normal (índice W):')
    print(sin_alarma.sensor_origen.value_counts().head(8).to_string())
cols_lista = [c for c in LLAVES + ['conjunto', 'timestamp_inicio', 'error_ae',
                                   'sensor_origen', 'w_max', 'banda_w']
              if c in sin_alarma.columns]
sin_alarma.sort_values('error_ae', ascending=False)[cols_lista].to_csv(
    salida('05a_anomalas_sin_alarma.csv'), index=False)


# =============================================================================
# 6. NO SUPERVISADO: K-MEANS Y DBSCAN SOBRE LAS QUE MARCA EL AUTOENCODER
# =============================================================================
print('\n' + '=' * 72)
print('6. AGRUPAMIENTO NO SUPERVISADO DE LAS VENTANAS MARCADAS')
print('=' * 72)

features = pd.read_csv(ARCHIVO_FEATURES)
FEATURES = [c for c in features.columns if c not in NO_FEATURE]
SIN_EXTREMOS = [f for f in FEATURES if not f.endswith(('_max', '_min'))]
assert len(SIN_EXTREMOS) == 42, f'{len(SIN_EXTREMOS)} sin extremos, esperaba 42'
assert not set(SIN_EXTREMOS) & set(ids_reglas), 'Reglas entre las features.'

dm = d[d.marcada].merge(features[LLAVES + SIN_EXTREMOS], on=LLAVES,
                        how='inner', validate='one_to_one').copy()
m_va, m_te = dm[dm.conjunto == 'val'], dm[dm.conjunto == 'test']
print(f'Marcadas: validación {len(m_va):,} | prueba {len(m_te):,}')
REPRES = {'SIN_EXTREMOS': SIN_EXTREMOS}
if COLS_W:
    REPRES['W_SENSORES'] = COLS_W


def silueta(X, lab):
    ok = lab >= 0
    if len(set(lab[ok])) < 2:
        return np.nan
    rng = np.random.RandomState(SEED)
    idx = np.where(ok)[0]
    if len(idx) > MAX_SILUETA:
        idx = rng.choice(idx, MAX_SILUETA, replace=False)
    return float(silhouette_score(X[idx], lab[idx]))


def comparar(lab, banda):
    """ARI y NMI contra las bandas de N, sólo donde N > 0."""
    pos = banda != '0_normal'
    if pos.sum() < 2:
        return np.nan, np.nan
    return (float(adjusted_rand_score(banda[pos], lab[pos])),
            float(normalized_mutual_info_score(banda[pos], lab[pos])))


def asignar_dbscan(modelo, X_ajuste, X_nuevo, eps):
    nucleos = modelo.core_sample_indices_
    if len(nucleos) == 0:
        return np.full(len(X_nuevo), -1)
    nn = NearestNeighbors(n_neighbors=1).fit(X_ajuste[nucleos])
    dist, j = nn.kneighbors(X_nuevo)
    lab = modelo.labels_[nucleos][j.ravel()]
    return np.where(dist.ravel() <= eps, lab, -1)


filas_ag = []
tablas_k3 = {}
for rep, cols in REPRES.items():
    sc = StandardScaler().fit(m_va[cols].fillna(0))
    _zva = sc.transform(m_va[cols].fillna(0))
    pca = PCA(random_state=SEED).fit(_zva)
    n_comp = int(min(PCA_MAX, np.searchsorted(
        np.cumsum(pca.explained_variance_ratio_), PCA_VARIANZA) + 1))
    pca = PCA(n_components=n_comp, random_state=SEED).fit(_zva)
    Xva = pca.transform(_zva)
    Xte = pca.transform(sc.transform(m_te[cols].fillna(0)))
    print(f'{rep}: {len(cols)} variables -> {n_comp} componentes '
          f'({100 * pca.explained_variance_ratio_.sum():.0f} % de la varianza)')
    b_va, b_te = m_va[BANDA].to_numpy(), m_te[BANDA].to_numpy()

    for k in K_RANGO:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xva)
        l_va, l_te = km.labels_, km.predict(Xte)
        ari_va, nmi_va = comparar(l_va, b_va)
        ari_te, nmi_te = comparar(l_te, b_te)
        filas_ag.append({
            'representacion': rep, 'componentes': n_comp,
            'metodo': f'K-means k={k}', 'grupos': k,
            'ruido_pct_test': 0.0, 'silueta_val': round(silueta(Xva, l_va), 3),
            'davies_bouldin_val': round(float(davies_bouldin_score(Xva, l_va)), 3),
            'ARI_val': round(ari_va, 3), 'ARI_test': round(ari_te, 3),
            'NMI_test': round(nmi_te, 3)})
        if k == 3:
            _t = m_te.copy()
            _t['grupo'] = l_te
            tablas_k3[rep] = _t

    ms = 2 * Xva.shape[1]
    kd = NearestNeighbors(n_neighbors=ms).fit(Xva).kneighbors(Xva)[0][:, -1]
    for q in DBSCAN_CUANTILES_EPS:
        eps = float(np.quantile(kd, q))
        db = DBSCAN(eps=eps, min_samples=ms).fit(Xva)
        l_va = db.labels_
        l_te = asignar_dbscan(db, Xva, Xte, eps)
        g = len(set(l_va) - {-1})
        ari_va, _ = comparar(l_va, b_va)
        ari_te, nmi_te = comparar(l_te, b_te)
        filas_ag.append({
            'representacion': rep, 'componentes': n_comp,
            'metodo': f'DBSCAN eps=q{int(q * 100)}',
            'grupos': g, 'ruido_pct_test': round(100 * (l_te == -1).mean(), 1),
            'silueta_val': round(silueta(Xva, l_va), 3),
            'davies_bouldin_val': np.nan,
            'ARI_val': round(ari_va, 3), 'ARI_test': round(ari_te, 3),
            'NMI_test': round(nmi_te, 3)})

agrup = pd.DataFrame(filas_ag)
print(agrup.to_string(index=False))

print('\nCómo leerlo: la silueta mide si los grupos están bien separados entre')
print('sí (de -1 a 1, más es mejor). ARI mide si coinciden con leve / media /')
print('grave de N (0 = azar, 1 = los mismos grupos). DBSCAN deja ruido: lo que')
print('no cae en ningún grupo denso.')
# El mejor se elige en VALIDACIÓN (ARI_val); prueba sólo se reporta.
_mejor = agrup.sort_values('ARI_val', ascending=False).iloc[0]
_sil = agrup.dropna(subset=['silueta_val']).sort_values('silueta_val',
                                                         ascending=False).iloc[0]
print(f'\nEl que más coincide con las bandas de N (elegido en validación): '
      f'{_mejor.metodo} sobre {_mejor.representacion} '
      f'(ARI val {_mejor.ARI_val} | prueba {_mejor.ARI_test})')
print(f'Los grupos mejor separados (silueta): {_sil.metodo} sobre '
      f'{_sil.representacion} (silueta {_sil.silueta_val})')
_k3 = agrup[agrup.metodo == 'K-means k=3']
print('K-means con 3 grupos (leve / media / grave): ' + ' | '.join(
    f'{r.representacion} ARI {r.ARI_test}' for r in _k3.itertuples()))
for rep, _t in tablas_k3.items():
    orden = (_t.groupby('grupo')[N_PRINC].median().sort_values().index)
    print(f'\nK-means k=3 sobre {rep}, prueba. Grupos ordenados por su N mediano:')
    print(pd.crosstab(_t.grupo, _t[BANDA]).reindex(orden).to_string())


# =============================================================================
# 7. XGBOOST, POR TANTEAR: ¿SE RECONOCE LA BANDA DE N DESDE LA SEÑAL?
# =============================================================================
print('\n' + '=' * 72)
print(f'7. {"XGBOOST" if HAY_XGB else "RANDOM FOREST (no hay xgboost)"}: '
      f'BANDA DE N DESDE LA SEÑAL')
print('=' * 72)

tr = m_va[m_va[BANDA] != '0_normal']
te = m_te[m_te[BANDA] != '0_normal']
filas_xgb = []
_n_tr = tr[BANDA].value_counts()
if (_n_tr.reindex(NIVELES).fillna(0) < MIN_CLASE_XGB).any():
    print(f'Alguna banda tiene menos de {MIN_CLASE_XGB} ventanas para entrenar:')
    print(_n_tr.to_string())
    print('Se salta el tanteo supervisado.')
else:
    cod = {nv: i for i, nv in enumerate(NIVELES)}
    y_tr, y_te = tr[BANDA].map(cod).to_numpy(), te[BANDA].map(cod).to_numpy()
    pw = compute_sample_weight('balanced', y_tr)
    if HAY_XGB:
        mod = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, eval_metric='mlogloss')
    else:
        mod = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     random_state=SEED, n_jobs=-1)
    mod.fit(tr[SIN_EXTREMOS], y_tr, sample_weight=pw)
    pred = mod.predict(te[SIN_EXTREMOS])
    filas_xgb.append({'modelo': 'XGBoost' if HAY_XGB else 'RandomForest',
                      'f1_macro': f1_score(y_te, pred, average='macro'),
                      'bal_acc': balanced_accuracy_score(y_te, pred)})
    for est in ['most_frequent', 'stratified']:
        dc = DummyClassifier(strategy=est, random_state=SEED).fit(tr[SIN_EXTREMOS], y_tr)
        p = dc.predict(te[SIN_EXTREMOS])
        filas_xgb.append({'modelo': f'base_{est}',
                          'f1_macro': f1_score(y_te, p, average='macro'),
                          'bal_acc': balanced_accuracy_score(y_te, p)})
    xgb_t = pd.DataFrame(filas_xgb).round(3)
    print(f'Entrenamiento (val): {len(tr):,} | prueba: {len(te):,}')
    print(xgb_t.to_string(index=False))


# =============================================================================
# 8. GUARDAR Y RESUMEN
# =============================================================================
cols_out = [c for c in LLAVES + ['timestamp_inicio', 'conjunto_fecha',
                                 'criticidad_min', 'N_lineal', 'N_empinada']
            + [c for c in ventanas.columns if c.startswith('banda_N_')]
            if c in ventanas.columns]
ventanas[cols_out].to_csv(salida('05a_nivel_ventanas.csv'), index=False)
cortes.to_csv(salida('05a_cortes.csv'), index=False)
bandas_err.to_csv(salida('05a_bandas_vs_error.csv'), index=False)
agrup.to_csv(salida('05a_agrupamiento.csv'), index=False)
if filas_xgb:
    pd.DataFrame(filas_xgb).round(4).to_csv(salida('05a_xgb.csv'), index=False)

print('\n' + '=' * 72)
print(f'RESUMEN — {EQUIPO} | n_j = {CONTEO} | banda {BANDA}')
print('=' * 72)
_c = cortes[(cortes.escala == ESCALA_PRINCIPAL) & (cortes.metodo == CORTE_PRINCIPAL)].iloc[0]
print(f'Cortes de N: leve hasta {_c.corte_leve_media}, media hasta '
      f'{_c.corte_media_grave}, grave sobre eso.')
for _datos in ['validación', 'prueba']:
    _e = bandas_err[(bandas_err.escala == BANDA) & (bandas_err.datos == _datos)]
    if len(_e):
        _v = ' / '.join(f'{_e[f"veces_{k}"].iloc[0]:.1f}'
                        for k in ['0_normal'] + NIVELES if f'veces_{k}' in _e)
        print(f'¿El error del autoencoder ordena las bandas de N en {_datos}? '
              f'{"sí" if bool(_e.monotona.iloc[0]) else "NO"}  ({_v} veces el normal)')
print(f'Anómalas para el autoencoder sin ninguna alarma: {len(sin_alarma):,} '
      f'(lista en 05a_anomalas_sin_alarma.csv)')
print(f'Agrupamiento que más coincide con las bandas (elegido en validación): '
      f'{_mejor.metodo} / {_mejor.representacion}, ARI en prueba {_mejor.ARI_test}')
print('Guardado: 05a_nivel_ventanas, 05a_cortes, 05a_bandas_vs_error, '
      '05a_anomalas_sin_alarma, 05a_agrupamiento, 05a_xgb')
