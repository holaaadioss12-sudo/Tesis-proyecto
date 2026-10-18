# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 10b — ¿EL AUTOENCODER SEPARA LA GRAVEDAD EN SU REPRESENTACIÓN INTERNA?
=============================================================================
El paso 10 mostró que K-means y DBSCAN no encuentran leve / media / grave
sobre las variables de la ventana (ARI en prueba <= 0,035). Probado aparte, lo
mismo pasó con GMM, HDBSCAN, HMM, K-means sembrado, DEC e IDEC sobre el índice
W y el error del autoencoder. Queda una prueba con motivo propio: agrupar el
espacio latente del autoencoder Transformer ya entrenado, los 64 números con
que el encoder resume cada ventana (media sobre los 60 pasos). Es la
representación que aprendió cómo se ve la operación normal, y no hay que
entrenar nada nuevo. Si tampoco ahí aparecen las tres bandas, la conclusión
del paso 10 queda firme: la gravedad no forma grupos naturales en la señal.

Qué hace:
  1. Rearma las ventanas de validación y prueba EXACTAMENTE como el paso 05
     (mismo corte, mismo normalizador, sin traslape) y las pasa por el
     checkpoint de mínima pérdida (PARCHE P8).
  2. Verifica que el error de reconstrucción recalculado sea el mismo del
     paso 05. Si no coincide, se detiene: las ventanas no serían las mismas.
  3. Guarda el latente de cada ventana (10b_latente.npz).
  4. Agrupa el latente de las ventanas que el autoencoder marca, con la misma
     receta del paso 10: se ajusta en validación y se mide en prueba.
        K-means k = 2..6, GMM k = 3 y k por BIC, HDBSCAN, y K-means sembrado
        con TODAS las etiquetas de validación (diagnóstico: si ni partiendo
        de los centros reales recupera las bandas, las bandas se solapan).
     Se informa la coincidencia con las bandas de N (ARI, NMI) y con la regla
     dominante de la ventana (NMI): ¿los grupos son de gravedad o de tipo?
  5. Sonda supervisada: regresión logística y XGBoost sobre el latente. Dice
     si la gravedad está en el latente aunque no forme grupos.
  6. Un mapa del latente (2 componentes principales) coloreado por banda y por
     regla dominante.

Entradas: {EQUIPO}_limpio.csv, {EQUIPO}_corte_split.json   (paso 01)
          normalizador_W60.npz, autoencoder_W60_normales.pt (paso 04)
          {EQUIPO}_error_reconstruccion_W60.csv            (paso 05)
          10_nivel_ventanas.csv                            (paso 10)
Salidas:  10b_latente.npz, 10b_agrupamiento_latente.csv,
          10b_sonda_latente.csv, 10b_latente_mapa.png
Corre en un par de minutos (CPU o GPU); lo más lento es leer el limpio.
=============================================================================
"""
import os
import json
import math

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

EQUIPO = os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'     # CR010 movido a su subcarpeta
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
try:
    from sklearn.cluster import HDBSCAN      # scikit-learn >= 1.3
except ImportError:
    HDBSCAN = None
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, f1_score, balanced_accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    HAY_XGB = True
except ImportError:
    HAY_XGB = False


# =============================================================================
# 0. CONSTANTES
# =============================================================================
SEED = 42
W = 60
LLAVES = ['tramo_id', 'ventana_id']
AE = 'err_AE_normales'          # checkpoint de mínima pérdida (PARCHE P8)
PERCENTIL_UMBRAL = 95
BANDA = 'banda_N_lineal_natural'
NIVELES = ['1_leve', '2_media', '3_grave']
K_RANGO = range(2, 7)
PCA_VARIANZA, PCA_MAX = 0.80, 10
MAX_SILUETA = 5000
TOL_ERROR = 1e-3                # diferencia relativa máxima con el paso 05
MIN_CLASE = 10                  # ventanas mínimas por banda para sembrar y sondear

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
ARCHIVO_NIVEL = f'{RUTA_EQ}/10_nivel_ventanas.csv'
_carpetas = [RUTA_EQ, f'{RUTA_EQ}/CAMBIOS'] + (
    [RUTA_BASE, f'{RUTA_BASE}/CAMBIOS'] if EQUIPO == 'CR010' else [])


def buscar(nombre):
    for c in _carpetas:
        if os.path.exists(f'{c}/{nombre}'):
            return f'{c}/{nombre}'
    raise SystemExit(f'No encontré {nombre} en: {_carpetas}')


RUTA_MODELO = buscar(f'autoencoder_W{W}_normales.pt')
RUTA_NORMALIZADOR = buscar(f'normalizador_W{W}.npz')
for _f in [ARCHIVO_LIMPIO, ARCHIVO_CORTE, ARCHIVO_ERROR, ARCHIVO_NIVEL]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste los pasos 01 a 05 y el 10?')
print(f'Modelo: {RUTA_MODELO}')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 1. LAS MISMAS VENTANAS DEL PASO 05
# =============================================================================
# Copia literal de lo que hace el paso 05: tramos continuos de 10 s, se
# descartan los de menos de W muestras y se numeran DESPUÉS de descartar,
# corte por la fecha de inicio del tramo, normalizador del paso 04 y ventanas
# sin traslape.
print('=' * 72)
print(f'PASO 10b — EL ESPACIO LATENTE DEL AUTOENCODER (W = {W})')
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
VARIABLES = [v for v in VARIABLES if v in df.columns]
tramos = [t for t in segmentar_tramos_continuos(df) if len(t) >= W]
tramos_con_id = list(enumerate(tramos))

corte = json.load(open(ARCHIVO_CORTE))
T_TRAIN, T_VAL = pd.Timestamp(corte['T_train']), pd.Timestamp(corte['T_val'])
_ini = np.array([t.index[0] for _, t in tramos_con_id])
tramos_val = [t for t, i in zip(tramos_con_id, _ini) if T_TRAIN < i <= T_VAL]
tramos_test = [t for t, i in zip(tramos_con_id, _ini) if i > T_VAL]

norm = np.load(RUTA_NORMALIZADOR, allow_pickle=True)
_vars = list(norm['variables'])
if _vars != VARIABLES:
    raise SystemExit(f'Las variables no coinciden con el normalizador: {_vars}')
media = pd.Series(norm['media'], index=_vars)
desv = pd.Series(norm['std'], index=_vars)


def ventanas_con_id(tramos):
    X, ids = [], []
    for tid, t in tramos:
        arr = ((t[VARIABLES] - media) / desv).to_numpy(dtype=np.float32)
        for vid, inicio in enumerate(range(0, len(arr) - W + 1, W)):
            X.append(arr[inicio:inicio + W])
            ids.append((tid, vid))
    return np.stack(X), ids


X_val, ids_val = ventanas_con_id(tramos_val)
X_test, ids_test = ventanas_con_id(tramos_test)
X = np.concatenate([X_val, X_test])
ids = pd.DataFrame(ids_val + ids_test, columns=LLAVES)
ids['conjunto'] = ['val'] * len(X_val) + ['test'] * len(X_test)
print(f'Ventanas: validación {len(X_val):,} | prueba {len(X_test):,}')


# =============================================================================
# 2. EL MODELO: ERROR (PARA VERIFICAR) Y LATENTE
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AutoencoderTransformer(nn.Module):
    def __init__(self, n_vars, W, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1):
        super().__init__()
        self.W = W
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        capa_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(capa_encoder, num_layers=num_layers)
        capa_decoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(capa_decoder, num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def encode(self, x):
        return self.encoder(self.pos_enc(self.embed(x))).mean(dim=1)

    def decode(self, latente):
        h = latente.unsqueeze(1).repeat(1, self.W, 1)
        return self.salida(self.decoder(self.pos_enc(h)))

    def forward(self, x):
        return self.decode(self.encode(x))


modelo = AutoencoderTransformer(n_vars=len(VARIABLES), W=W).to(device)
modelo.load_state_dict(torch.load(RUTA_MODELO, map_location=device))
modelo.eval()
Z, errores = [], []
with torch.no_grad():
    for i in range(0, len(X), 256):
        b = torch.from_numpy(X[i:i + 256]).to(device)
        z = modelo.encode(b)
        errores.append(((modelo.decode(z) - b) ** 2).mean(dim=(1, 2)).cpu().numpy())
        Z.append(z.cpu().numpy())
Z = np.concatenate(Z).astype(np.float64)
ids['error_recalculado'] = np.concatenate(errores)

err = pd.read_csv(ARCHIVO_ERROR)
_cols = [c for c in LLAVES + ['conjunto', AE, 'es_normal', 'etiqueta_tfidf']
         if c in err.columns]
d = ids.merge(err[_cols], on=LLAVES + ['conjunto'], how='left',
              validate='one_to_one')
if d[AE].isna().any():
    raise SystemExit(f'{int(d[AE].isna().sum())} ventanas no están en el paso 05: '
                     'las ventanas no calzan.')
_dif = (np.abs(d.error_recalculado - d[AE]) / d[AE].clip(lower=1e-9)).max()
print(f'Error recalculado contra el del paso 05: diferencia relativa máxima '
      f'{_dif:.2e}   <- debe ser menor que {TOL_ERROR:g}')
if _dif > TOL_ERROR:
    raise SystemExit('El error no coincide con el del paso 05: no son las mismas '
                     'ventanas o no es el mismo modelo. No sigo.')
np.savez_compressed(salida('10b_latente.npz'), z=Z.astype(np.float32),
                    tramo_id=d.tramo_id.to_numpy(), ventana_id=d.ventana_id.to_numpy(),
                    conjunto=d.conjunto.to_numpy())
print(f'Latente guardado: {Z.shape[0]:,} ventanas x {Z.shape[1]} dimensiones')


# =============================================================================
# 3. BANDAS DE N Y VENTANAS MARCADAS (igual que el paso 10)
# =============================================================================
nivel = pd.read_csv(ARCHIVO_NIVEL)
d = d.merge(nivel[LLAVES + ['N_lineal', BANDA]], on=LLAVES, how='left',
            validate='one_to_one')
if d[BANDA].isna().any():
    raise SystemExit('Faltan ventanas en 10_nivel_ventanas.csv: corre el paso 10.')
d['sin_alarma'] = d.es_normal if 'es_normal' in d.columns else d.N_lineal == 0
d['error_ae'] = d[AE]
u = float(np.percentile(d.loc[(d.conjunto == 'val') & d.sin_alarma, 'error_ae'],
                        PERCENTIL_UMBRAL))
d['marcada'] = d.error_ae > u
print(f'Umbral del autoencoder (P{PERCENTIL_UMBRAL} sin alarma, validación): {u:.5f}')

mv = (d.conjunto == 'val').to_numpy() & d.marcada.to_numpy()
mt = (d.conjunto == 'test').to_numpy() & d.marcada.to_numpy()
b_va, b_te = d[BANDA].to_numpy()[mv], d[BANDA].to_numpy()[mt]
pos_va, pos_te = b_va != '0_normal', b_te != '0_normal'
tipo = d.etiqueta_tfidf.to_numpy() if 'etiqueta_tfidf' in d.columns else None
tipo_te = tipo[mt] if tipo is not None else None
print(f'Marcadas: validación {mv.sum():,} (N > 0: {pos_va.sum():,}) | '
      f'prueba {mt.sum():,} (N > 0: {pos_te.sum():,})')


# =============================================================================
# 4. AGRUPAMIENTO DEL LATENTE
# =============================================================================
print('\n' + '=' * 72)
print('4. AGRUPAMIENTO DEL ESPACIO LATENTE (ventanas marcadas)')
print('=' * 72)

sc = StandardScaler().fit(Z[mv])
_zv, _zt = sc.transform(Z[mv]), sc.transform(Z[mt])
pca = PCA(random_state=SEED).fit(_zv)
n_comp = int(min(PCA_MAX, np.searchsorted(
    np.cumsum(pca.explained_variance_ratio_), PCA_VARIANZA) + 1))
pca = PCA(n_components=n_comp, random_state=SEED).fit(_zv)
Xv, Xt = pca.transform(_zv), pca.transform(_zt)
print(f'Latente: 64 dimensiones -> {n_comp} componentes '
      f'({100 * pca.explained_variance_ratio_.sum():.0f} % de la varianza)')


def silueta(Xs, lab):
    ok = lab >= 0
    if len(set(lab[ok])) < 2:
        return np.nan
    idx = np.where(ok)[0]
    if len(idx) > MAX_SILUETA:
        idx = np.random.RandomState(SEED).choice(idx, MAX_SILUETA, replace=False)
    return float(silhouette_score(Xs[idx], lab[idx]))


filas = []


def anota(metodo, grupos, l_va, l_te, Xs=Xv):
    fila = {'metodo': metodo, 'grupos': grupos,
            'silueta_val': round(silueta(Xs, l_va), 3),
            'ARI_val': round(adjusted_rand_score(b_va[pos_va], l_va[pos_va]), 3),
            'ARI_test': round(adjusted_rand_score(b_te[pos_te], l_te[pos_te]), 3),
            'NMI_banda_test': round(normalized_mutual_info_score(b_te[pos_te], l_te[pos_te]), 3)}
    if tipo_te is not None:
        fila['NMI_tipo_test'] = round(normalized_mutual_info_score(
            tipo_te[pos_te], l_te[pos_te]), 3)
    filas.append(fila)


for k in K_RANGO:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xv)
    anota(f'K-means k={k}', k, km.labels_, km.predict(Xt))


def gmm(k, n_init):
    return GaussianMixture(k, covariance_type='full', n_init=n_init,
                           reg_covar=1e-4, random_state=SEED).fit(Xv)


g = gmm(3, 5)
anota('GMM k=3', 3, g.predict(Xv), g.predict(Xt))
_bic = {}
for k in range(2, 9):
    try:
        _bic[k] = gmm(k, 3).bic(Xv)
    except ValueError:              # covarianza degenerada: ese k no se usa
        pass
_kb = min(_bic, key=_bic.get)
g = gmm(_kb, 5)
anota(f'GMM k={_kb} (BIC)', _kb, g.predict(Xv), g.predict(Xt))
if HDBSCAN is not None:
    h = HDBSCAN(min_cluster_size=int(min(30, max(5, len(Xv) // 10)))).fit(Xv)
    _gh = len(set(h.labels_) - {-1})
    _lt = (KNeighborsClassifier(1).fit(Xv, h.labels_).predict(Xt) if _gh
           else np.full(len(Xt), -1))
    anota(f'HDBSCAN ({_gh} grupos, {100 * (h.labels_ == -1).mean():.0f} % ruido)',
          _gh, h.labels_, _lt)
# Diagnóstico: partir de los centros REALES de cada banda (todas las etiquetas
# de validación). Si ni así recupera las bandas, éstas se solapan en el latente.
_n_banda = pd.Series(b_va[pos_va]).value_counts().reindex(NIVELES).fillna(0)
HAY_BANDAS = bool((_n_banda >= MIN_CLASE).all())
if HAY_BANDAS:
    _cent = np.vstack([Xv[pos_va][b_va[pos_va] == nv].mean(0) for nv in NIVELES])
    km = KMeans(3, init=_cent, n_init=1, random_state=SEED).fit(Xv[pos_va])
    anota('K-means sembrado (100 % etiquetas, diagnóstico)', 3,
          km.predict(Xv), km.predict(Xt))
else:
    print(f'Alguna banda tiene menos de {MIN_CLASE} ventanas marcadas en validación: '
          f'{_n_banda.to_dict()}. Se saltan el sembrado y la sonda.')

agrup = pd.DataFrame(filas)
print(agrup.to_string(index=False))
print('\nARI: 0 = azar, 1 = los mismos grupos que leve / media / grave.')
print('NMI_tipo > NMI_banda: los grupos se parecen más a la regla dominante')
print('(el tipo de anomalía) que a la gravedad.')


# =============================================================================
# 5. SONDA SUPERVISADA: ¿LA GRAVEDAD ESTÁ EN EL LATENTE?
# =============================================================================
print('\n' + '=' * 72)
print('5. SONDA SUPERVISADA SOBRE EL LATENTE (entrena en validación)')
print('=' * 72)
cod = {nv: i for i, nv in enumerate(NIVELES)}
Ztr = sc.transform(Z[mv][pos_va]); ytr = pd.Series(b_va[pos_va]).map(cod).to_numpy()
Zte = sc.transform(Z[mt][pos_te]); yte = pd.Series(b_te[pos_te]).map(cod).to_numpy()
pw = compute_sample_weight('balanced', ytr)
modelos = {'Regresión logística': LogisticRegression(max_iter=2000, C=1.0)}
if HAY_XGB:
    modelos['XGBoost'] = XGBClassifier(n_estimators=300, max_depth=4,
                                       learning_rate=0.1, subsample=0.8,
                                       colsample_bytree=0.8, random_state=SEED,
                                       eval_metric='mlogloss')
filas_s = []
for nom, mod in (modelos.items() if HAY_BANDAS else []):
    mod.fit(Ztr, ytr, sample_weight=pw)
    p = mod.predict(Zte)
    filas_s.append({'modelo': f'{nom} (latente)', 'f1_macro': f1_score(yte, p, average='macro'),
                    'bal_acc': balanced_accuracy_score(yte, p)})
if HAY_BANDAS:
    p = DummyClassifier(strategy='stratified', random_state=SEED).fit(Ztr, ytr).predict(Zte)
    filas_s.append({'modelo': 'Base: al azar estratificado',
                    'f1_macro': f1_score(yte, p, average='macro'),
                    'bal_acc': balanced_accuracy_score(yte, p)})
if os.path.exists(salida('10_xgb.csv')):
    _x = pd.read_csv(salida('10_xgb.csv'))
    _x = _x[_x.modelo.isin(['XGBoost', 'RandomForest'])]
    for r in _x.itertuples():
        filas_s.append({'modelo': f'{r.modelo} (42 variables, paso 10)',
                        'f1_macro': r.f1_macro, 'bal_acc': r.bal_acc})
sonda = pd.DataFrame(filas_s).round(3)
print(sonda.to_string(index=False))


# =============================================================================
# 6. MAPA DEL LATENTE
# =============================================================================
# Dos componentes principales del latente (ajustadas en validación), ventanas
# de prueba marcadas con N > 0. Izquierda: banda de N, rampa ordinal de un solo
# tono. Derecha: regla dominante, tres más frecuentes + resto en gris, con
# forma distinta además del color.
P2 = PCA(n_components=2, random_state=SEED).fit(_zv)
xy = P2.transform(_zt)[pos_te]
TINTA, TINTA2, GRILLA, FONDO = '#0b0b0b', '#52514e', '#e1e0d9', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': TINTA2, 'xtick.color': TINTA2,
                     'ytick.color': TINTA2})
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True,
                       facecolor=FONDO)
for a in ax:
    a.set_facecolor(FONDO)
    a.grid(color=GRILLA, linewidth=0.6)
    a.set_axisbelow(True)
    a.set_xlabel('Componente principal 1 del latente')
ax[0].set_ylabel('Componente principal 2 del latente')

RAMPA = {'1_leve': '#86b6ef', '2_media': '#2a78d6', '3_grave': '#104281'}
for nv, nombre in zip(NIVELES, ['Leve', 'Media', 'Grave']):
    s = b_te[pos_te] == nv
    ax[0].scatter(xy[s, 0], xy[s, 1], s=10, c=RAMPA[nv], alpha=0.75,
                  linewidths=0, label=f'{nombre} ({s.sum():,})')
ax[0].set_title('Coloreado por banda de N', color=TINTA, loc='left')
ax[0].legend(frameon=False, loc='best', markerscale=2)

if tipo_te is not None:
    t = pd.Series(tipo_te[pos_te]).fillna('?').to_numpy()
    top = pd.Series(t).value_counts().index[:3].tolist()
    estilos = [('#2a78d6', 'o'), ('#eb6834', 's'), ('#1baf7a', '^')]
    resto = ~np.isin(t, top)
    ax[1].scatter(xy[resto, 0], xy[resto, 1], s=8, c='#898781', alpha=0.5,
                  linewidths=0, marker='D', label=f'Otras reglas ({resto.sum():,})')
    for r, (col, mk) in zip(top, estilos):
        s = t == r
        ax[1].scatter(xy[s, 0], xy[s, 1], s=12, c=col, alpha=0.75, linewidths=0,
                      marker=mk, label=f'Regla {r} ({s.sum():,})')
    ax[1].set_title('Coloreado por regla dominante', color=TINTA, loc='left')
    ax[1].legend(frameon=False, loc='best', markerscale=2)
fig.suptitle(f'{EQUIPO} · espacio latente del autoencoder, ventanas de prueba '
             f'marcadas con N > 0', color=TINTA, x=0.01, ha='left')
fig.tight_layout()
fig.savefig(salida('10b_latente_mapa.png'), dpi=160, facecolor=FONDO)
plt.close(fig)


# =============================================================================
# 7. GUARDAR Y RESUMEN
# =============================================================================
agrup.to_csv(salida('10b_agrupamiento_latente.csv'), index=False)
sonda.to_csv(salida('10b_sonda_latente.csv'), index=False)

print('\n' + '=' * 72)
print(f'RESUMEN — {EQUIPO}')
print('=' * 72)
_no_diag = agrup[~agrup.metodo.str.startswith('K-means sembrado')]
_m = _no_diag.sort_values('ARI_val', ascending=False).iloc[0]
print(f'Mejor agrupamiento del latente (elegido en validación): {_m.metodo}, '
      f'ARI en prueba {_m.ARI_test}')
print(f'Máximo ARI en prueba de todos los métodos sin etiquetas: '
      f'{_no_diag.ARI_test.max()}')
_s = agrup[agrup.metodo.str.startswith('K-means sembrado')]
if len(_s):
    print(f'K-means sembrado con todas las etiquetas: ARI en prueba '
          f'{_s.ARI_test.iloc[0]}')
print('Sonda supervisada: ' + ' | '.join(
    f'{r.modelo} F1 {r.f1_macro}' for r in sonda.itertuples()))
print('Guardado: 10b_latente.npz, 10b_agrupamiento_latente.csv, '
      '10b_sonda_latente.csv, 10b_latente_mapa.png')
