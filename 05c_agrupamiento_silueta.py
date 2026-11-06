# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 05c — AGRUPAMIENTO DEL LATENTE TEMPORAL CON ÍNDICE DE SILUETA
=============================================================================
Entrada:  el modelo del paso 03b (FUENTE=oficial) o de 03a_modelos/ (FUENTE=03a), 05a_nivel_ventanas.csv
Salida:   05c_agrupamiento.csv, 05c_perfil_grupos.csv, 05c_silueta.png, 05c_umap.png, 05c_prototipos.csv/png,
          05c_grupos_ventanas.csv y 05c_eleccion.json (para el paso 05d)
"""
import os
import sys
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
    RUTA_EQ = f'{RUTA_BASE}/CR010'
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')
if RUTA_BASE not in sys.path:
    sys.path.append(RUTA_BASE)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    HDBSCAN = None
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
try:
    import umap
except ImportError:
    umap = None
import warnings
warnings.filterwarnings('ignore', message='.*n_jobs value.*')

from reglas_engine import evaluar_reglas

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 0. CONSTANTES
# =============================================================================
SEED = 42
W = 60
STRIDE_EVAL = W
PERCENTIL_UMBRAL = 95
K_RANGO = range(2, 21)
CUANTILES_EPS = [0.10, 0.25, 0.50, 0.75]
PCA_VARIANZA, PCA_MAX = 0.80, 20
MAX_SILUETA = 5000
TOL_ERROR = 1e-3
MAX_RUIDO = 0.20                 # DBSCAN/HDBSCAN con más ruido no se eligen
UMAP_DIMS = (2, 5, 10)           # UMAP sobre el latente antes de agrupar (McConville et al., 2021)
UMAP_VECINOS = 20
BANDA = 'banda_N_empinada_natural'
NIVELES = ['1_leve', '2_media', '3_grave']
VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
FAMILIAS = {
    'lubricacion': ['D', 'A5', 'E'],
    'corriente':   ['A', 'A1', 'A4'],
    'atollo':      ['B', 'A3'],
    'filtro':      ['C'],
    'temperatura': ['F', 'A7', 'A6', 'A10', 'A9', 'A8', 'I', 'A11', 'A13', 'A12',
                    'K', 'A15', 'A14', 'A16', 'L', 'A21', 'A22'],
    'vibracion':   ['O', 'A20', 'A19', 'A18', 'A17'],
}

ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
ARCHIVO_04G = f'{RUTA_EQ}/03a_latente_temporal.csv'
ARCHIVO_NIVEL = f'{RUTA_EQ}/05a_nivel_ventanas.csv'
ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
# FUENTE = 'oficial': el modelo del paso 03b; '03a': un modelo del barrido del 03a
FUENTE = os.environ.get('FUENTE', 'oficial')
PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
if PUNTO_PARADA not in ('perdida', 'separacion'):
    raise SystemExit(f"PUNTO_PARADA tiene que ser 'perdida' o 'separacion', no {PUNTO_PARADA!r}.")
AE = 'err_AE_normales_sep' if PUNTO_PARADA == 'separacion' else 'err_AE_normales'
_SUF = '_sep' if PUNTO_PARADA == 'separacion' else ''
for _f in [ARCHIVO_LIMPIO, ARCHIVO_CORTE, ARCHIVO_NIVEL] + (
        [ARCHIVO_04G] if FUENTE == '03a' else [ARCHIVO_ERROR]):
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste los pasos 03b, 04 y 05a (o el 03a)?')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# 1. MODELO
# =============================================================================
print('=' * 72)
print(f'PASO 05c — AGRUPAMIENTO DEL LATENTE TEMPORAL ({EQUIPO}, fuente {FUENTE})')
print('=' * 72)
if FUENTE == '03a':
    r04 = pd.read_csv(ARCHIVO_04G, sep=';', decimal=',')
    _temp = r04[r04.red != 'transformer'].groupby(['red', 'latent_dim']).auc_val.mean()
    if _temp.empty:
        raise SystemExit('El 03a no tiene corridas de redes temporales.')
    RED = os.environ.get('MODELO_RED', _temp.idxmax()[0])
    K = int(os.environ.get('MODELO_K', _temp.idxmax()[1]))
    SEMILLA = int(os.environ.get('SEMILLA', r04[(r04.red == RED) & (r04.latent_dim == K)].semilla.min()))
    print(f'Modelo: {RED}, latent_dim = {K}, semilla {SEMILLA} '
          f'(por defecto, el de mayor AUC de validación en el 03a)')
    RUTA_MOD = salida(f'03a_modelos/{RED}_k{K}_s{SEMILLA}.pt')
    RUTA_REF = salida(f'03a_modelos/transformer_k64_s{SEMILLA}.pt')
    RUTA_ERR = salida(f'03a_errores/{RED}_k{K}_s{SEMILLA}.npz')
    for _f in [RUTA_MOD, RUTA_ERR]:
        if not os.path.exists(_f):
            raise SystemExit(f'Falta {_f}: corre el 03a con esa red, k y semilla.')
elif FUENTE == 'oficial':
    _cands = [salida(f'autoencoder_W{W}_normales{_SUF}.pt'), salida(f'CAMBIOS/autoencoder_W{W}_normales{_SUF}.pt')]
    RUTA_MOD = next((r for r in _cands if os.path.exists(r)), None)
    if RUTA_MOD is None:
        raise SystemExit(f'No encontré el modelo del paso 03b en: {_cands}')
    _cfg_ruta = os.path.join(os.path.dirname(RUTA_MOD), f'autoencoder_W{W}_config.json')
    CONFIG_AE = (json.load(open(_cfg_ruta)) if os.path.exists(_cfg_ruta)
                 else {'tipo': 'colapsado', 'latent_dim': 64})
    RED = 'transformer_t' if CONFIG_AE['tipo'] == 'temporal' else 'transformer'
    K = int(CONFIG_AE['latent_dim'])
    RUTA_REF = os.environ.get('MODELO_REFERENCIA', '')     # opcional: el modelo anterior (.pt)
    print(f"Modelo oficial: {CONFIG_AE['tipo']}, latent_dim = {K}, punto de parada: {PUNTO_PARADA} ({RUTA_MOD})")
else:
    raise SystemExit(f"FUENTE tiene que ser 'oficial' o '03a', no {FUENTE!r}.")


# =============================================================================
# 2. LAS MISMAS VENTANAS DEL 03a Y DEL 03b
# =============================================================================
def cargar_limpio(ruta):
    cabecera = pd.read_csv(ruta, nrows=0)
    col = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col], low_memory=False)
    return df.set_index(col).sort_index().rename_axis('timestamp')


df = cargar_limpio(ARCHIVO_LIMPIO)
VARIABLES = [v for v in VARIABLES if v in df.columns]
N_VARS = len(VARIABLES)
corte = json.load(open(ARCHIVO_CORTE))
T_TRAIN, T_VAL = pd.Timestamp(corte['T_train']), pd.Timestamp(corte['T_val'])

idx = df.index.to_series()
df['_tramo'] = (idx.diff() > pd.Timedelta('10s')).cumsum().values
tramos, reglas_tramo, IDS = [], [], None
for _, g in df.groupby('_tramo'):
    if len(g) >= W:
        t = g.drop(columns='_tramo')
        r = evaluar_reglas(t, verbose=False)
        if IDS is None:
            IDS = [c for c in r.columns if c not in ('n_reglas_activadas', 'estado')]
        tramos.append(t)
        reglas_tramo.append(r[IDS].to_numpy().astype(np.int16))

_ini = np.array([t.index[0] for t in tramos])
conjunto_tramo = np.where(_ini <= T_TRAIN, 'train', np.where(_ini <= T_VAL, 'val', 'test'))
concat = pd.concat([t for t, c in zip(tramos, conjunto_tramo) if c == 'train'])[VARIABLES]
media_n, std_n = concat.mean(), concat.std().replace(0, 1.0)
if FUENTE == 'oficial':                        # el normalizador que guardó el paso 03b
    _nz = np.load(os.path.join(os.path.dirname(RUTA_MOD), f'normalizador_W{W}.npz'), allow_pickle=True)
    media_n = pd.Series(_nz['media'], index=list(_nz['variables']))[VARIABLES]
    std_n = pd.Series(_nz['std'], index=list(_nz['variables']))[VARIABLES]


def construir(conjunto):
    X, cuenta, inicio = [], [], []
    for t, rg, c in zip(tramos, reglas_tramo, conjunto_tramo):
        if c != conjunto:
            continue
        arr = ((t[VARIABLES] - media_n) / std_n).to_numpy(dtype=np.float32)
        for i in range(0, len(arr) - W + 1, STRIDE_EVAL):
            X.append(arr[i:i + W])
            cuenta.append(rg[i:i + W].sum(axis=0))      # muestras en alarma por regla
            inicio.append(t.index[i])
    return np.stack(X), np.array(cuenta), pd.DatetimeIndex(inicio)


X_va, cu_va, ts_va = construir('val')
X_te, cu_te, ts_te = construir('test')
al_va, al_te = cu_va.sum(axis=1) > 0, cu_te.sum(axis=1) > 0
print(f'Ventanas: validación {len(X_va):,} | prueba {len(X_te):,}')


def conteo_familias(cuenta):
    return np.column_stack([cuenta[:, [IDS.index(r) for r in rs if r in IDS]].sum(axis=1)
                            for rs in FAMILIAS.values()])


def familia_dominante(cuenta):
    fam = conteo_familias(cuenta)
    nombres = np.array(list(FAMILIAS))
    return np.where(fam.sum(axis=1) > 0, nombres[fam.argmax(axis=1)], 'sin_alarma')


tipo_va, tipo_te = familia_dominante(cu_va), familia_dominante(cu_te)


# =============================================================================
# 3. MODELOS (mismas clases del 03a) Y LATENTE
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


def capa_transformer(d_model=64, nhead=4, dim_ff=128, dropout=0.1):
    return nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                      dropout=dropout, batch_first=True)


class AETransformer(nn.Module):
    def __init__(self, n_vars, W, k=None, d_model=64, num_layers=2):
        super().__init__()
        self.W = W
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        self.encoder = nn.TransformerEncoder(capa_transformer(d_model), num_layers=num_layers)
        self.decoder = nn.TransformerEncoder(capa_transformer(d_model), num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def encode(self, x):
        return self.encoder(self.pos_enc(self.embed(x))).mean(dim=1)

    def decode(self, z):
        h = z.unsqueeze(1).repeat(1, self.W, 1)
        return self.salida(self.decoder(self.pos_enc(h)))

    def forward(self, x):
        return self.decode(self.encode(x))


class AETransformerT(nn.Module):
    def __init__(self, n_vars, W, k, d_model=64, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        self.encoder = nn.TransformerEncoder(capa_transformer(d_model), num_layers=num_layers)
        self.a_latente = nn.Linear(d_model, k)
        self.desde_latente = nn.Linear(k, d_model)
        self.decoder = nn.TransformerEncoder(capa_transformer(d_model), num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def encode(self, x):
        return self.a_latente(self.encoder(self.pos_enc(self.embed(x))))

    def decode(self, z):
        return self.salida(self.decoder(self.pos_enc(self.desde_latente(z))))

    def forward(self, x):
        return self.decode(self.encode(x))


class AECNNT(nn.Module):
    def __init__(self, n_vars, W, k, canales=32, nucleo=5, dropout=0.1):
        super().__init__()
        p = nucleo // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(n_vars, canales, nucleo, padding=p), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(canales, k, nucleo, padding=p))
        self.decoder = nn.Sequential(
            nn.Conv1d(k, canales, nucleo, padding=p), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(canales, n_vars, nucleo, padding=p))

    def encode(self, x):
        return self.encoder(x.transpose(1, 2)).transpose(1, 2)

    def decode(self, z):
        return self.decoder(z.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        return self.decode(self.encode(x))


class AECNNLSTM(nn.Module):
    def __init__(self, n_vars, W, k, canales=32, oculto=64, nucleo=5, dropout=0.1):
        super().__init__()
        p = nucleo // 2
        self.conv_in = nn.Sequential(nn.Conv1d(n_vars, canales, nucleo, padding=p), nn.ReLU())
        self.lstm_enc = nn.LSTM(canales, oculto, batch_first=True)
        self.a_latente = nn.Linear(oculto, k)
        self.desde_latente = nn.Linear(k, oculto)
        self.lstm_dec = nn.LSTM(oculto, oculto, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(oculto, n_vars, nucleo, padding=p)

    def encode(self, x):
        h, _ = self.lstm_enc(self.conv_in(x.transpose(1, 2)).transpose(1, 2))
        return self.a_latente(self.drop(h))

    def decode(self, z):
        h, _ = self.lstm_dec(self.desde_latente(z))
        return self.conv_out(self.drop(h).transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        return self.decode(self.encode(x))


CLASES = {'transformer': AETransformer, 'transformer_t': AETransformerT,
          'cnn_t': AECNNT, 'cnn_lstm': AECNNLSTM}


def cargar(red, k, ruta):
    m = CLASES[red](N_VARS, W, k).to(device)
    m.load_state_dict(torch.load(ruta, map_location=device))
    return m.eval()


def latente_y_error(modelo, X, batch=256):
    Z, E = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.from_numpy(X[i:i + batch]).to(device)
            Z.append(modelo.encode(b).cpu().numpy())
            E.append(((modelo(b) - b) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(Z).astype(np.float64), np.concatenate(E)


modelo = cargar(RED, K, RUTA_MOD)
Z_va, ev_va = latente_y_error(modelo, X_va)
Z_te, ev_te = latente_y_error(modelo, X_te)
print(f'Latente temporal: {Z_va.shape[1:]} por ventana')

# verificación: mismas ventanas y mismo modelo que el paso 04 (o que el 03a)
if FUENTE == '03a':
    _g = np.load(RUTA_ERR)
    _ref_va, _ref_te, _yo_va, _yo_te = _g['ev_va'], _g['ev_te'], ev_va, ev_te
else:                     # el paso 04 numera distinto: se comparan los errores ordenados
    _e5 = pd.read_csv(ARCHIVO_ERROR, usecols=lambda c: c in ('conjunto', AE, 'error_reconstruccion'))
    if 'error_reconstruccion' in _e5.columns and not np.allclose(_e5[AE], _e5.error_reconstruccion):
        raise SystemExit(f'El paso 04 se corrió con otro punto de parada: córrelo con PUNTO_PARADA={PUNTO_PARADA!r}.')
    _ref_va = np.sort(_e5.loc[_e5.conjunto == 'val', AE].to_numpy())
    _ref_te = np.sort(_e5.loc[_e5.conjunto == 'test', AE].to_numpy())
    _yo_va, _yo_te = np.sort(ev_va.mean(axis=1)), np.sort(ev_te.mean(axis=1))
    if len(_ref_va) != len(_yo_va) or len(_ref_te) != len(_yo_te):
        raise SystemExit(f'Distinto número de ventanas que el paso 04: val {len(_yo_va)} contra '
                         f'{len(_ref_va)}, prueba {len(_yo_te)} contra {len(_ref_te)}.')
_dif = max(np.max(np.abs(_yo_va - _ref_va) / np.clip(_ref_va, 1e-9, None)),
           np.max(np.abs(_yo_te - _ref_te) / np.clip(_ref_te, 1e-9, None)))
print(f'Error recalculado contra el del {FUENTE if FUENTE == "03a" else "paso 04"}: '
      f'diferencia relativa máxima {_dif:.2e}')
if _dif > TOL_ERROR:
    raise SystemExit('No coincide: no son las mismas ventanas o el mismo modelo.')

e_va, e_te = ev_va.mean(axis=1), ev_te.mean(axis=1)
umbral = float(np.percentile(e_va[~al_va], PERCENTIL_UMBRAL))
mv, mt = e_va > umbral, e_te > umbral
print(f'Umbral P{PERCENTIL_UMBRAL} (ventanas sin alarma de validación): {umbral:.5f}')
print(f'Ventanas marcadas: validación {mv.sum():,} | prueba {mt.sum():,}')

if RUTA_REF and os.path.exists(RUTA_REF) and RED != 'transformer':
    Zr_va, _ = latente_y_error(cargar('transformer', 64, RUTA_REF), X_va)
    Zr_te, _ = latente_y_error(cargar('transformer', 64, RUTA_REF), X_te)
else:
    Zr_va = Zr_te = None
    print('Sin modelo de referencia (latente de 64): se omite esa comparación.')


# =============================================================================
# 4. NIVEL DE CADA VENTANA (paso 05a)
# =============================================================================
_nv = pd.read_csv(ARCHIVO_NIVEL, usecols=['timestamp_inicio', 'N_empinada', BANDA],
                  parse_dates=['timestamp_inicio']).drop_duplicates('timestamp_inicio')
_mapa_b = _nv.set_index('timestamp_inicio')[BANDA]
_mapa_n = _nv.set_index('timestamp_inicio')['N_empinada']
b_va, b_te = ts_va.map(_mapa_b).to_numpy(dtype=object), ts_te.map(_mapa_b).to_numpy(dtype=object)
n_va, n_te = ts_va.map(_mapa_n).to_numpy(dtype=float), ts_te.map(_mapa_n).to_numpy(dtype=float)
_cob = np.mean(pd.notna(np.concatenate([b_va, b_te])))
print(f'Ventanas con nivel del paso 05a: {_cob:.1%}')
if _cob < 0.9:
    print('OJO: menos del 90 % cruzó con 05a_nivel_ventanas.csv. Revisa que sea del mismo limpio.')


# =============================================================================
# 5. REPRESENTACIONES DEL LATENTE
# =============================================================================
def resumen_temporal(Z):
    t = np.arange(Z.shape[1]) - (Z.shape[1] - 1) / 2
    pendiente = np.einsum('t,ntk->nk', t, Z - Z.mean(axis=1, keepdims=True)) / (t ** 2).sum()
    return np.concatenate([Z.mean(1), Z.std(1), Z.min(1), Z.max(1), pendiente], axis=1)


def preparar(A_va, A_te):
    sc = StandardScaler().fit(A_va)
    a_va, a_te = sc.transform(A_va), sc.transform(A_te)
    p = PCA(random_state=SEED).fit(a_va)
    n = int(min(PCA_MAX, a_va.shape[1], np.searchsorted(np.cumsum(p.explained_variance_ratio_),
                                                         PCA_VARIANZA) + 1))
    p = PCA(n_components=n, random_state=SEED).fit(a_va)
    return p.transform(a_va), p.transform(a_te), n, a_va, a_te


if Z_va.ndim == 3:
    REPRES = {
        f'temporal_resumen (media, desv, mín, máx y pendiente de las {K} dimensiones)':
            (resumen_temporal(Z_va[mv]), resumen_temporal(Z_te[mt])),
        f'temporal_aplanado (60 x {K} valores)':
            (Z_va[mv].reshape(mv.sum(), -1), Z_te[mt].reshape(mt.sum(), -1)),
    }
else:                                          # modelo sin eje de tiempo (latente de 64)
    REPRES = {'actual_64 (latente sin tiempo)': (Z_va[mv], Z_te[mt])}
if Zr_va is not None:
    REPRES['actual_64 (latente sin tiempo, referencia)'] = (Zr_va[mv], Zr_te[mt])


# =============================================================================
# 6. AGRUPAMIENTO: K-MEANS (k = 2 a 20), DBSCAN, HDBSCAN Y GMM
# =============================================================================
print('\n' + '=' * 72)
print('6. AGRUPAMIENTO DE LAS VENTANAS MARCADAS')
print('=' * 72)
bv, bt = b_va[mv], b_te[mt]
pos_v = np.isin(bv, NIVELES)
pos_t = np.isin(bt, NIVELES)
tv, tt = tipo_va[mv], tipo_te[mt]


def silueta(X, lab):
    ok = lab >= 0
    if len(set(lab[ok])) < 2:
        return np.nan
    i = np.where(ok)[0]
    if len(i) > MAX_SILUETA:
        i = np.random.RandomState(SEED).choice(i, MAX_SILUETA, replace=False)
    return float(silhouette_score(X[i], lab[i]))


def ari(b, lab, m):
    return adjusted_rand_score(b[m], lab[m]) if m.sum() > 1 else np.nan


filas, etiquetas = [], {}


def anota(rep, metodo, k, l_va, l_te, Xv, Xt, espacio='PCA', Lv=None, Lt=None):
    s_v, s_t = silueta(Xv, l_va), silueta(Xt, l_te)
    fila = {'representacion': rep, 'espacio': espacio, 'metodo': metodo, 'grupos': k,
            'ruido_pct': round(100 * float((l_va < 0).mean()), 1),
            'silueta_val': round(s_v, 3),
            'silueta_test': round(s_t, 3),
            'silueta_val_latente': round(s_v if Lv is None else silueta(Lv, l_va), 3),
            'silueta_test_latente': round(s_t if Lt is None else silueta(Lt, l_te), 3),
            'ARI_nivel_val': round(ari(bv, l_va, pos_v), 3),
            'ARI_nivel_test': round(ari(bt, l_te, pos_t), 3),
            'NMI_nivel_test': round(normalized_mutual_info_score(bt[pos_t], l_te[pos_t]), 3),
            'NMI_tipo_test': round(normalized_mutual_info_score(tt[pos_t], l_te[pos_t]), 3)}
    filas.append(fila)
    etiquetas[(rep, metodo)] = (l_va, l_te)


def agrupar(rep, Xv, Xt, espacio='PCA', Lv=None, Lt=None, con_dbscan=True):
    """K-means (k = 2 a 20), DBSCAN, HDBSCAN y GMM sobre un espacio."""
    extra = dict(espacio=espacio, Lv=Lv, Lt=Lt)
    for k in K_RANGO:
        if k >= len(Xv):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xv)
        anota(rep, f'K-means k={k}', k, km.labels_, km.predict(Xt), Xv, Xt, **extra)
    min_s = 2 * Xv.shape[1]                        # Sander et al. (1998)
    dk = (np.sort(NearestNeighbors(n_neighbors=min_s).fit(Xv).kneighbors(Xv)[0][:, -1])
          if con_dbscan and len(Xv) > min_s else None)
    for q in (CUANTILES_EPS if dk is not None else []):
        eps = float(np.quantile(dk, q))
        db = DBSCAN(eps=eps, min_samples=min_s).fit(Xv)
        g = len(set(db.labels_) - {-1})
        l_te = (KNeighborsClassifier(1).fit(Xv, db.labels_).predict(Xt) if g else np.full(len(Xt), -1))
        anota(rep, f'DBSCAN eps=P{int(100 * q)}', g, db.labels_, l_te, Xv, Xt, **extra)
    if HDBSCAN is not None:
        h = HDBSCAN(min_cluster_size=int(min(30, max(5, len(Xv) // 10))), copy=True).fit(Xv)
        g = len(set(h.labels_) - {-1})
        l_te = (KNeighborsClassifier(1).fit(Xv, h.labels_).predict(Xt) if g else np.full(len(Xt), -1))
        anota(rep, 'HDBSCAN', g, h.labels_, l_te, Xv, Xt, **extra)
    bic = {}
    for k in range(2, 11):
        try:
            bic[k] = GaussianMixture(k, covariance_type='full', reg_covar=1e-4, n_init=3,
                                     random_state=SEED).fit(Xv).bic(Xv)
        except ValueError:
            pass
    if bic:
        kb = min(bic, key=bic.get)
        gm = GaussianMixture(kb, covariance_type='full', reg_covar=1e-4, n_init=5,
                             random_state=SEED).fit(Xv)
        anota(rep, f'GMM k={kb} (BIC)', kb, gm.predict(Xv), gm.predict(Xt), Xv, Xt, **extra)


ESPACIOS = {}
for rep, (A_va, A_te) in REPRES.items():
    Xv, Xt, n, Sv, St = preparar(A_va, A_te)
    ESPACIOS[rep] = (Xv, Xt, Sv, St)
    print(f'\n{rep}: {A_va.shape[1]} valores -> {n} componentes principales')
    agrupar(rep, Xv, Xt)


# =============================================================================
# 6b. UMAP SOBRE EL LATENTE Y LUEGO AGRUPAMIENTO
# =============================================================================
# UMAP conserva los vecinos cercanos del latente en 2, 5 o 10 dimensiones y sobre
# eso se agrupa (McConville et al., 2021). La silueta en el espacio UMAP sale
# optimista; por eso cada partición se mide también en el latente (PCA).
UMAP2D = {}
if umap is None:
    print('\nUMAP no está instalado: se omite la sección 6b (en Colab: !pip install umap-learn).')
else:
    print('\n' + '=' * 72)
    print('6b. UMAP SOBRE EL LATENTE Y LUEGO AGRUPAMIENTO')
    print('=' * 72)
    for rep in [r for r in REPRES if r.startswith('temporal')]:
        Xv, Xt, Sv, St = ESPACIOS[rep]
        if len(Sv) <= UMAP_VECINOS + 1:
            continue
        for d in UMAP_DIMS:
            red_u = umap.UMAP(n_components=d, n_neighbors=UMAP_VECINOS, min_dist=0.0,
                              random_state=SEED).fit(Sv)
            Uv, Ut = red_u.embedding_, red_u.transform(St)
            if d == 2:
                UMAP2D[rep] = Uv
            print(f'{rep} + UMAP {d}D')
            agrupar(f'{rep} + UMAP {d}D', Uv, Ut, f'UMAP {d}D', Xv, Xt, con_dbscan=False)

agrup = pd.DataFrame(filas)
agrup.to_csv(salida('05c_agrupamiento.csv'), sep=';', decimal=',', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 240)
pd.set_option('display.max_colwidth', 40)


# =============================================================================
# 7. ELECCIÓN POR SILUETA (en validación) Y LECTURA
# =============================================================================
print('\n' + '=' * 72)
print('7. MEJOR AGRUPAMIENTO DE CADA REPRESENTACIÓN, ELEGIDO POR SILUETA')
print('=' * 72)
_ok = agrup[(agrup.grupos >= 2) & (agrup.ruido_pct <= 100 * MAX_RUIDO)].dropna(subset=['silueta_val'])
mejores = _ok.loc[_ok.groupby('representacion').silueta_val.idxmax()]
print(mejores.to_string(index=False))
print(f'\nSolo se eligen particiones que dejan hasta {MAX_RUIDO:.0%} de las ventanas como ruido.')
print('Silueta: sobre 0,50 hay grupos razonables; 0,25 o menos, no hay grupos claros.')
print('silueta_val: medida donde se agrupó (PCA o UMAP). silueta_val_latente: la misma partición')
print('medida en el latente (PCA); es la que permite comparar PCA con UMAP.')
print('ARI de nivel: 0 = azar, 1 = los mismos grupos que leve / media / grave.')
print('NMI de tipo mayor que NMI de nivel: los grupos se parecen más al tipo de alarma.')

fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
for rep in agrup.representacion.unique():
    s = agrup[(agrup.representacion == rep) & agrup.metodo.str.startswith('K-means')]
    etq = rep.split(' (')[0] + (rep[rep.index(' + UMAP'):] if ' + UMAP' in rep else '')
    ejes[0].plot(s.grupos, s.silueta_val, 'o-', ms=3, label=etq)
    ejes[1].plot(s.grupos, s.ARI_nivel_val, 'o-', ms=3, label=etq)
ejes[0].axhline(0.25, color='k', ls=':', lw=1)
ejes[0].set(xlabel='Número de grupos (K-means)', ylabel='Silueta, validación',
            title='¿Hay grupos? (sobre la línea punteada, algo de estructura)')
ejes[1].set(xlabel='Número de grupos (K-means)', ylabel='ARI contra leve / media / grave',
            title='¿Son los niveles?')
for a in ejes:
    a.set_xticks(list(K_RANGO))
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
fig.suptitle(f'{EQUIPO}: latente de {RED} (60 x {K}) y latente actual (64)')
fig.tight_layout()
fig.savefig(salida('05c_silueta.png'), dpi=150)
plt.close(fig)


# =============================================================================
# 8. PERFIL DE LOS GRUPOS DEL MEJOR AGRUPAMIENTO TEMPORAL
# =============================================================================
print('\n' + '=' * 72)
print('8. ¿QUÉ ES CADA GRUPO? (mejor agrupamiento del latente temporal, validación)')
print('=' * 72)
_temporales = mejores[mejores.representacion.str.startswith('temporal')]
if _temporales.empty:
    print('Ningún agrupamiento temporal tiene al menos 2 grupos con poco ruido.')
    for _f in ('05c_grupos_ventanas.csv', '05c_eleccion.json'):     # que el 05d no lea una corrida vieja
        if os.path.exists(salida(_f)):
            os.remove(salida(_f))
else:
    elegido = _temporales.loc[_temporales.silueta_val_latente.idxmax()]
    l_va, l_te = etiquetas[(elegido.representacion, elegido.metodo)]
    _base = elegido.representacion.split(' + UMAP')[0]
    print(f'{elegido.representacion} · {elegido.metodo} · silueta en el latente '
          f'{elegido.silueta_val_latente} (elegido por la silueta en el latente)')
    ev_m = ev_va[mv]
    aporte = ev_m / ev_m.sum(axis=1, keepdims=True)
    perfil = []
    for gr in sorted(set(l_va)):
        s = l_va == gr
        niveles = pd.Series(bv[s]).value_counts(normalize=True)
        tipos = pd.Series(tv[s]).value_counts(normalize=True)
        top_v = np.argsort(-aporte[s].mean(axis=0))[:3]
        perfil.append({
            'grupo': 'ruido' if gr < 0 else int(gr), 'ventanas': int(s.sum()),
            'N_mediana': float(np.nanmedian(n_va[mv][s])),
            'pct_N0': round(100 * niveles.get('0_normal', 0), 1),
            'pct_leve': round(100 * niveles.get('1_leve', 0), 1),
            'pct_media': round(100 * niveles.get('2_media', 0), 1),
            'pct_grave': round(100 * niveles.get('3_grave', 0), 1),
            'tipo_principal': f'{tipos.index[0]} ({100 * tipos.iloc[0]:.0f} %)',
            'tipo_segundo': f'{tipos.index[1]} ({100 * tipos.iloc[1]:.0f} %)' if len(tipos) > 1 else '',
            'variables_con_mas_error': ', '.join(f'{VARIABLES[j]} {100 * aporte[s][:, j].mean():.0f} %'
                                                 for j in top_v),
            'error_mediano': round(float(np.median(e_va[mv][s])), 5)})
    perfil = pd.DataFrame(perfil)
    print(perfil.to_string(index=False))
    perfil.to_csv(salida('05c_perfil_grupos.csv'), sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print('\nSi un grupo concentra un tipo de alarma y unas variables, describe un modo de')
    print('comportamiento aunque no coincida con leve / media / grave.')

    # grupo de cada ventana marcada y medoide (la más cercana al centro del grupo en el latente), para el 05d
    _Xb = ESPACIOS[_base][0]
    _medoide = np.zeros(len(l_va), dtype=int)
    for gr in sorted(g for g in set(l_va) if g >= 0):
        _i = np.where(l_va == gr)[0]
        _medoide[_i[np.argmin(((_Xb[_i] - _Xb[_i].mean(axis=0)) ** 2).sum(axis=1))]] = 1
    _partes = []
    for _conj, _m, _lab, _ts, _b, _n, _tipo, _ev, _cu, _e, _med in [
            ('val', mv, l_va, ts_va, b_va, n_va, tipo_va, ev_va, cu_va, e_va, _medoide),
            ('test', mt, l_te, ts_te, b_te, n_te, tipo_te, ev_te, cu_te, e_te, np.zeros(mt.sum(), dtype=int))]:
        _d = {'conjunto': _conj, 'timestamp_inicio': _ts[_m], 'grupo': _lab, 'nivel': _b[_m], 'N': _n[_m],
              'tipo_dominante': _tipo[_m], 'error_reconstruccion': _e[_m], 'medoide': _med}
        _d.update({f'evar_{v}': _ev[_m][:, j] for j, v in enumerate(VARIABLES)})
        _d.update({f'fam_{f}': c for f, c in zip(FAMILIAS, conteo_familias(_cu[_m]).T)})
        _partes.append(pd.DataFrame(_d))
    pd.concat(_partes).to_csv(salida('05c_grupos_ventanas.csv'), sep=';', decimal=',', index=False,
                              encoding='utf-8-sig')
    with open(salida('05c_eleccion.json'), 'w', encoding='utf-8') as _f:
        json.dump({'equipo': EQUIPO, 'punto_parada': PUNTO_PARADA, 'fuente': FUENTE, 'red': RED,
                   'latent_dim': K, 'representacion': str(elegido.representacion),
                   'metodo': str(elegido.metodo), 'espacio': str(elegido.espacio),
                   'grupos': int(elegido.grupos), 'ruido_pct': float(elegido.ruido_pct),
                   'silueta_val_latente': float(elegido.silueta_val_latente),
                   'silueta_test_latente': float(elegido.silueta_test_latente),
                   'umbral': umbral, 'banda': BANDA}, _f, ensure_ascii=False, indent=1)
    print('Grupo de cada ventana marcada: 05c_grupos_ventanas.csv (lo usa el paso 05d)')

    # mapa UMAP 2D de las ventanas marcadas: por grupo y por nivel
    if _base in UMAP2D:
        U = UMAP2D[_base]
        fig, ejes = plt.subplots(1, 2, figsize=(13, 5.5))
        for gr in sorted(set(l_va)):
            s = l_va == gr
            ejes[0].scatter(U[s, 0], U[s, 1], s=6, alpha=0.6,
                            c='lightgray' if gr < 0 else None, label='ruido' if gr < 0 else f'grupo {gr}')
        for nv, col in zip(['0_normal'] + NIVELES, ['#9e9e9e', '#4caf50', '#ff9800', '#d32f2f']):
            s = bv == nv
            ejes[1].scatter(U[s, 0], U[s, 1], s=6, alpha=0.6, c=col, label=nv)
        ejes[0].set_title(f'Grupos: {elegido.metodo} ({elegido.espacio})')
        ejes[1].set_title('Nivel de N de cada ventana')
        for a in ejes:
            a.set(xlabel='UMAP 1', ylabel='UMAP 2')
            a.legend(fontsize=8, markerscale=2)
            a.grid(alpha=0.3)
        fig.suptitle(f'{EQUIPO}: ventanas marcadas en el mapa UMAP 2D del latente (validación)')
        fig.tight_layout()
        fig.savefig(salida('05c_umap.png'), dpi=150)
        plt.close(fig)
        print('Mapa UMAP 2D por grupo y por nivel: 05c_umap.png')


# =============================================================================
# 9. DECODIFICAR LOS GRUPOS: ¿CÓMO ES LA VENTANA TÍPICA DE CADA UNO?
# =============================================================================
# El latente promedio de cada grupo pasa por el decodificador; se compara con el
# de las ventanas sin alarma que el modelo no marca (la operación normal típica).
if not _temporales.empty:
    print('\n' + '=' * 72)
    print('9. LA VENTANA TÍPICA DE CADA GRUPO (latente promedio decodificado)')
    print('=' * 72)
    modelo.eval()

    def decodificar(Z):
        with torch.no_grad():
            z = torch.tensor(Z.mean(axis=0, keepdims=True), dtype=torch.float32, device=device)
            x = modelo.decode(z)[0].cpu().numpy()                     # (60, 14), estandarizado
        return x * std_n.to_numpy() + media_n.to_numpy()             # unidades del sensor

    proto = {'normal': decodificar(Z_va[(~al_va) & (~mv)])}
    for gr in sorted(g for g in set(l_va) if g >= 0):
        proto[f'grupo {gr}'] = decodificar(Z_va[mv][l_va == gr])
    # variables que más se apartan de la ventana normal típica, en desviaciones estándar
    desvio = {n: np.abs(p - proto['normal']).mean(axis=0) / std_n.to_numpy()
              for n, p in proto.items() if n != 'normal'}
    filas_p = []
    for n, p in proto.items():
        for j, v in enumerate(VARIABLES):
            for t in range(W):
                filas_p.append({'prototipo': n, 'minuto': round(t * 10 / 60, 2), 'variable': v,
                                'valor': round(float(p[t, j]), 4)})
    pd.DataFrame(filas_p).to_csv(salida('05c_prototipos.csv'), sep=';', decimal=',', index=False,
                                 encoding='utf-8-sig')
    for n, dv in desvio.items():
        top = np.argsort(-dv)[:3]
        print(f'  {n:<9} se aparta de lo normal sobre todo en: ' + ', '.join(
            f'{VARIABLES[j]} ({proto[n][:, j].mean() - proto["normal"][:, j].mean():+.1f}, '
            f'{dv[j]:.1f} desv.)' for j in top))
    importantes = list(dict.fromkeys(VARIABLES[j] for dv in desvio.values() for j in np.argsort(-dv)[:2]))[:6]
    minutos = np.arange(W) * 10 / 60
    fig, ejes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, v in zip(ejes.ravel(), importantes):
        j = VARIABLES.index(v)
        ax.plot(minutos, proto['normal'][:, j], 'k--', lw=1.5, label='normal típica')
        for n in proto:
            if n != 'normal':
                ax.plot(minutos, proto[n][:, j], lw=1.6, label=n)
        ax.set_title(v)
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=True)
        ax.set_xlabel('Minuto de la ventana')
    for ax in ejes.ravel()[len(importantes):]:
        ax.axis('off')
    ejes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f'{EQUIPO}: ventana típica de cada grupo ({elegido.metodo}), decodificada del latente promedio')
    fig.tight_layout()
    fig.savefig(salida('05c_prototipos.png'), dpi=150)
    plt.close(fig)
    print('Entre paréntesis: diferencia promedio con la ventana normal típica, en las unidades del')
    print('sensor, y en desviaciones estándar. Gráfico: 05c_prototipos.png')

print(f'\nGuardado: 05c_agrupamiento.csv, 05c_perfil_grupos.csv, 05c_silueta.png, 05c_umap.png, '
      f'05c_prototipos.csv, 05c_prototipos.png, 05c_grupos_ventanas.csv y 05c_eleccion.json en {RUTA_EQ}')
