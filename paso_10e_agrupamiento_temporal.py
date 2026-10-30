# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 10e — AGRUPAMIENTO DEL LATENTE TEMPORAL CON ÍNDICE DE SILUETA
=============================================================================
Entrada:  04g_latente_temporal.csv, 04g_modelos/, 10_nivel_ventanas.csv
Salida:   10e_agrupamiento.csv, 10e_perfil_grupos.csv, 10e_silueta.png
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
MAX_RUIDO = 0.50                 # DBSCAN/HDBSCAN con más ruido no se eligen
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
ARCHIVO_04G = f'{RUTA_EQ}/04g_latente_temporal.csv'
ARCHIVO_NIVEL = f'{RUTA_EQ}/10_nivel_ventanas.csv'
for _f in [ARCHIVO_LIMPIO, ARCHIVO_CORTE, ARCHIVO_04G, ARCHIVO_NIVEL]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste los pasos 01, 04g y 10?')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# 1. MODELO ELEGIDO EN EL 04g
# =============================================================================
print('=' * 72)
print(f'PASO 10e — AGRUPAMIENTO DEL LATENTE TEMPORAL ({EQUIPO})')
print('=' * 72)
r04 = pd.read_csv(ARCHIVO_04G, sep=';', decimal=',')
_temp = r04[r04.red != 'transformer'].groupby(['red', 'latent_dim']).auc_val.mean()
if _temp.empty:
    raise SystemExit('El 04g no tiene corridas de redes temporales.')
RED = os.environ.get('MODELO_RED', _temp.idxmax()[0])
K = int(os.environ.get('MODELO_K', _temp.idxmax()[1]))
SEMILLA = int(os.environ.get('SEMILLA', r04[(r04.red == RED) & (r04.latent_dim == K)].semilla.min()))
print(f'Modelo temporal: {RED}, latent_dim = {K}, semilla {SEMILLA} '
      f'(por defecto, el de mayor AUC de validación en el 04g)')
RUTA_MOD = salida(f'04g_modelos/{RED}_k{K}_s{SEMILLA}.pt')
RUTA_REF = salida(f'04g_modelos/transformer_k64_s{SEMILLA}.pt')
RUTA_ERR = salida(f'04g_errores/{RED}_k{K}_s{SEMILLA}.npz')
for _f in [RUTA_MOD, RUTA_ERR]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}: corre el 04g con esa red, k y semilla.')


# =============================================================================
# 2. LAS MISMAS VENTANAS DEL 04g
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


def familia_dominante(cuenta):
    fam = np.column_stack([cuenta[:, [IDS.index(r) for r in rs if r in IDS]].sum(axis=1)
                           for rs in FAMILIAS.values()])
    nombres = np.array(list(FAMILIAS))
    return np.where(fam.sum(axis=1) > 0, nombres[fam.argmax(axis=1)], 'sin_alarma')


tipo_va, tipo_te = familia_dominante(cu_va), familia_dominante(cu_te)


# =============================================================================
# 3. MODELOS (mismas clases del 04g) Y LATENTE
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

    def forward(self, x):
        h = self.encode(x).unsqueeze(1).repeat(1, self.W, 1)
        return self.salida(self.decoder(self.pos_enc(h)))


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

    def forward(self, x):
        return self.salida(self.decoder(self.pos_enc(self.desde_latente(self.encode(x)))))


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

    def forward(self, x):
        return self.decoder(self.encode(x).transpose(1, 2)).transpose(1, 2)


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

    def forward(self, x):
        h, _ = self.lstm_dec(self.desde_latente(self.encode(x)))
        return self.conv_out(self.drop(h).transpose(1, 2)).transpose(1, 2)


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

# verificación: mismas ventanas y mismo modelo que el 04g
_g = np.load(RUTA_ERR)
_dif = max(np.max(np.abs(ev_va - _g['ev_va']) / np.clip(_g['ev_va'], 1e-9, None)),
           np.max(np.abs(ev_te - _g['ev_te']) / np.clip(_g['ev_te'], 1e-9, None)))
print(f'Error recalculado contra el del 04g: diferencia relativa máxima {_dif:.2e}')
if _dif > TOL_ERROR:
    raise SystemExit('No coincide con el 04g: no son las mismas ventanas o el mismo modelo.')

e_va, e_te = ev_va.mean(axis=1), ev_te.mean(axis=1)
umbral = float(np.percentile(e_va[~al_va], PERCENTIL_UMBRAL))
mv, mt = e_va > umbral, e_te > umbral
print(f'Umbral P{PERCENTIL_UMBRAL} (ventanas sin alarma de validación): {umbral:.5f}')
print(f'Ventanas marcadas: validación {mv.sum():,} | prueba {mt.sum():,}')

if os.path.exists(RUTA_REF):
    Zr_va, _ = latente_y_error(cargar('transformer', 64, RUTA_REF), X_va)
    Zr_te, _ = latente_y_error(cargar('transformer', 64, RUTA_REF), X_te)
else:
    Zr_va = Zr_te = None
    print('No está el modelo actual del 04g: se omite la comparación con el latente de 64.')


# =============================================================================
# 4. NIVEL DE CADA VENTANA (paso 10)
# =============================================================================
_nv = pd.read_csv(ARCHIVO_NIVEL, usecols=['timestamp_inicio', 'N_empinada', BANDA],
                  parse_dates=['timestamp_inicio']).drop_duplicates('timestamp_inicio')
_mapa_b = _nv.set_index('timestamp_inicio')[BANDA]
_mapa_n = _nv.set_index('timestamp_inicio')['N_empinada']
b_va, b_te = ts_va.map(_mapa_b).to_numpy(dtype=object), ts_te.map(_mapa_b).to_numpy(dtype=object)
n_va, n_te = ts_va.map(_mapa_n).to_numpy(dtype=float), ts_te.map(_mapa_n).to_numpy(dtype=float)
_cob = np.mean(pd.notna(np.concatenate([b_va, b_te])))
print(f'Ventanas con nivel del paso 10: {_cob:.1%}')
if _cob < 0.9:
    print('OJO: menos del 90 % cruzó con 10_nivel_ventanas.csv. Revisa que sea del mismo limpio.')


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
    return p.transform(a_va), p.transform(a_te), n


REPRES = {
    f'temporal_resumen (media, desv, mín, máx y pendiente de las {K} dimensiones)':
        (resumen_temporal(Z_va[mv]), resumen_temporal(Z_te[mt])),
    f'temporal_aplanado (60 x {K} valores)':
        (Z_va[mv].reshape(mv.sum(), -1), Z_te[mt].reshape(mt.sum(), -1)),
}
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


def anota(rep, metodo, k, l_va, l_te, Xv, Xt):
    fila = {'representacion': rep, 'metodo': metodo, 'grupos': k,
            'ruido_pct': round(100 * float((l_va < 0).mean()), 1),
            'silueta_val': round(silueta(Xv, l_va), 3),
            'silueta_test': round(silueta(Xt, l_te), 3),
            'ARI_nivel_val': round(ari(bv, l_va, pos_v), 3),
            'ARI_nivel_test': round(ari(bt, l_te, pos_t), 3),
            'NMI_nivel_test': round(normalized_mutual_info_score(bt[pos_t], l_te[pos_t]), 3),
            'NMI_tipo_test': round(normalized_mutual_info_score(tt[pos_t], l_te[pos_t]), 3)}
    filas.append(fila)
    etiquetas[(rep, metodo)] = (l_va, l_te)


for rep, (A_va, A_te) in REPRES.items():
    Xv, Xt, n = preparar(A_va, A_te)
    print(f'\n{rep}: {A_va.shape[1]} valores -> {n} componentes principales')
    for k in K_RANGO:
        if k >= len(Xv):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xv)
        anota(rep, f'K-means k={k}', k, km.labels_, km.predict(Xt), Xv, Xt)
    min_s = 2 * Xv.shape[1]                        # Sander et al. (1998)
    dk = (np.sort(NearestNeighbors(n_neighbors=min_s).fit(Xv).kneighbors(Xv)[0][:, -1])
          if len(Xv) > min_s else None)
    for q in (CUANTILES_EPS if dk is not None else []):
        eps = float(np.quantile(dk, q))
        db = DBSCAN(eps=eps, min_samples=min_s).fit(Xv)
        g = len(set(db.labels_) - {-1})
        l_te = (KNeighborsClassifier(1).fit(Xv, db.labels_).predict(Xt) if g else np.full(len(Xt), -1))
        anota(rep, f'DBSCAN eps=P{int(100 * q)}', g, db.labels_, l_te, Xv, Xt)
    if HDBSCAN is not None:
        h = HDBSCAN(min_cluster_size=int(min(30, max(5, len(Xv) // 10))), copy=True).fit(Xv)
        g = len(set(h.labels_) - {-1})
        l_te = (KNeighborsClassifier(1).fit(Xv, h.labels_).predict(Xt) if g else np.full(len(Xt), -1))
        anota(rep, 'HDBSCAN', g, h.labels_, l_te, Xv, Xt)
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
        anota(rep, f'GMM k={kb} (BIC)', kb, gm.predict(Xv), gm.predict(Xt), Xv, Xt)

agrup = pd.DataFrame(filas)
agrup.to_csv(salida('10e_agrupamiento.csv'), sep=';', decimal=',', index=False, encoding='utf-8-sig')
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
print('\nSilueta: sobre 0,50 hay grupos razonables; 0,25 o menos, no hay grupos claros.')
print('ARI de nivel: 0 = azar, 1 = los mismos grupos que leve / media / grave.')
print('NMI de tipo mayor que NMI de nivel: los grupos se parecen más al tipo de alarma.')

fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
for rep in REPRES:
    s = agrup[(agrup.representacion == rep) & agrup.metodo.str.startswith('K-means')]
    etq = rep.split(' (')[0]
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
fig.savefig(salida('10e_silueta.png'), dpi=150)
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
else:
    elegido = _temporales.loc[_temporales.silueta_val.idxmax()]
    l_va, _ = etiquetas[(elegido.representacion, elegido.metodo)]
    print(f'{elegido.representacion} · {elegido.metodo} · silueta {elegido.silueta_val}')
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
    perfil.to_csv(salida('10e_perfil_grupos.csv'), sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print('\nSi un grupo concentra un tipo de alarma y unas variables, describe un modo de')
    print('comportamiento aunque no coincida con leve / media / grave.')

print(f'\nGuardado: 10e_agrupamiento.csv, 10e_perfil_grupos.csv y 10e_silueta.png en {RUTA_EQ}')
