# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 03a — AUTOENCODER TEMPORAL: ELECCIÓN DE latent_dim (k) Y DE LA ARQUITECTURA
=============================================================================
"""
import os
import sys
import json
import math
import time

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
if RUTA_BASE not in sys.path:
    sys.path.append(RUTA_BASE)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reglas_engine import evaluar_reglas, POR_ID

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', device)
if device.type != 'cuda':
    print('AVISO: sin GPU esto tarda muchísimo. Activa el entorno con GPU en')
    print('       Entorno de ejecución > Cambiar tipo de entorno de ejecución.')


# =============================================================================
# 0. CONSTANTES
# =============================================================================
W = 60
STRIDE_TRAIN = 12
STRIDE_EVAL = W
EPOCAS_MAX = int(os.environ.get('EPOCAS_MAX', 100))   # solo para pruebas rápidas
PACIENCIA = 15
SEMILLAS = [int(s) for s in os.environ.get('SEMILLAS', '42').split(',')]
LATENTES = [int(k) for k in os.environ.get('LATENTES', '2,4,7,8,12').split(',')]
REDES = os.environ.get('REDES', 'transformer_t,cnn_t,cnn_lstm').split(',')
REF = 'transformer'                  # el modelo anterior: latente de 64 sin tiempo
K_REF = 64                           # su latente: 64 números, sin eje de tiempo
TOLERANCIA_CODO = 0.10
PERCENTIL_UMBRAL = 95
N_BOOT = 1000
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
ARCHIVO_SALIDA = f'{RUTA_EQ}/03a_latente_temporal.csv'
CARPETA_ERRORES = f'{RUTA_EQ}/03a_errores'
CARPETA_MODELOS = f'{RUTA_EQ}/03a_modelos'
for _f in [ARCHIVO_LIMPIO, ARCHIVO_CORTE]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste el paso 01?')
os.makedirs(CARPETA_ERRORES, exist_ok=True)
os.makedirs(CARPETA_MODELOS, exist_ok=True)


# =============================================================================
# 1. DATOS — las mismas ventanas y normalización del paso 03b
# =============================================================================
print('\n' + '=' * 72)
print(f'PASO 03a — ELECCIÓN DE latent_dim, LATENTE (60, k) ({EQUIPO})')
print('=' * 72)


def cargar_limpio(ruta):
    cabecera = pd.read_csv(ruta, nrows=0)
    col = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col], low_memory=False)
    return df.set_index(col).sort_index().rename_axis('timestamp')


df = cargar_limpio(ARCHIVO_LIMPIO)
VARIABLES = [v for v in VARIABLES if v in df.columns]
N_VARS = len(VARIABLES)
print(f'Filas: {len(df):,}   Variables: {N_VARS}')

corte = json.load(open(ARCHIVO_CORTE))
T_TRAIN, T_VAL = pd.Timestamp(corte['T_train']), pd.Timestamp(corte['T_val'])
print(f'Corte único: T_train = {T_TRAIN} | T_val = {T_VAL}')

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
        reglas_tramo.append(r[IDS].to_numpy().astype(bool))
print(f'Tramos útiles: {len(tramos):,} | reglas evaluadas: {len(IDS)}')


def variables_de(regla):
    """Las variables que mira una regla (umbral, compuesta, diferencia o máximo)."""
    r = POR_ID[regla]
    v = [r[c] for c in ('var', 'var2', 'a', 'b') if c in r] + list(r.get('vars', []))
    return [x for x in v if x in VARIABLES]


# matriz regla x variable: 1 si la regla mira esa variable
R2V = np.array([[v in variables_de(rg) for v in VARIABLES] for rg in IDS], dtype=bool)

_ini = np.array([t.index[0] for t in tramos])
conjunto_tramo = np.where(_ini <= T_TRAIN, 'train', np.where(_ini <= T_VAL, 'val', 'test'))
concat = pd.concat([t for t, c in zip(tramos, conjunto_tramo) if c == 'train'])[VARIABLES]
media_n, std_n = concat.mean(), concat.std().replace(0, 1.0)


def construir(conjunto, stride):
    X, reglas, tramo, pos, inicio = [], [], [], [], []
    for k, (t, rg, c) in enumerate(zip(tramos, reglas_tramo, conjunto_tramo)):
        if c != conjunto:
            continue
        arr = ((t[VARIABLES] - media_n) / std_n).to_numpy(dtype=np.float32)
        for j, i in enumerate(range(0, len(arr) - W + 1, stride)):
            X.append(arr[i:i + W])
            reglas.append(rg[i:i + W].any(axis=0))
            tramo.append(k)
            pos.append(j)
            inicio.append(t.index[i])
    reglas = np.array(reglas, dtype=bool)
    return (np.stack(X), reglas, reglas.any(axis=1), np.array(tramo), np.array(pos),
            pd.DatetimeIndex(inicio))


X_tr, _, al_tr, _, _, _ = construir('train', STRIDE_TRAIN)
X_va, rg_va, al_va, tr_va, po_va, ts_va = construir('val', STRIDE_EVAL)
X_te, rg_te, al_te, tr_te, po_te, ts_te = construir('test', STRIDE_EVAL)
X_tr_n = X_tr[~al_tr]
X_va_n = X_va[~al_va]
print(f'Entrenamiento: {len(X_tr):,} ventanas, {len(X_tr_n):,} normales')
print(f'Validación   : {len(X_va):,} ventanas ({int(al_va.sum()):,} con alarma)')
print(f'Prueba       : {len(X_te):,} ventanas ({int(al_te.sum()):,} con alarma)')


def siguiente_con_alarma(alarma, tramo, pos):
    s = pd.DataFrame({'a': alarma, 't': tramo, 'p': pos}).sort_values(['t', 'p'])
    sig = s.groupby('t').a.shift(-1)
    out = pd.Series(np.nan, index=s.index)
    out[sig.notna()] = sig[sig.notna()].astype(float)
    return out.sort_index().to_numpy()


fut_te = siguiente_con_alarma(al_te, tr_te, po_te)


class VD(Dataset):
    def __init__(self, X): self.X = torch.from_numpy(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


# =============================================================================
# 2. LAS ARQUITECTURAS — todas reciben y devuelven (B, 60, 14)
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
    """REFERENCIA: el modelo anterior (latente de 64, sin eje de tiempo)."""
    def __init__(self, n_vars, W, k=None, d_model=64, num_layers=2):
        super().__init__()
        self.W = W
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        self.encoder = nn.TransformerEncoder(capa_transformer(d_model), num_layers=num_layers)
        self.decoder = nn.TransformerEncoder(capa_transformer(d_model), num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def encode(self, x):                                       # (B, 60, 14)
        return self.encoder(self.pos_enc(self.embed(x))).mean(dim=1)   # (B, 64)

    def forward(self, x):
        h = self.encode(x).unsqueeze(1).repeat(1, self.W, 1)   # (B, 60, 64)
        return self.salida(self.decoder(self.pos_enc(h)))      # (B, 60, 14)


class AETransformerT(nn.Module):
    """El Transformer con el cuello de botella POR INSTANTE (el del paso 03b)."""
    def __init__(self, n_vars, W, k, d_model=64, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        self.encoder = nn.TransformerEncoder(capa_transformer(d_model), num_layers=num_layers)
        self.a_latente = nn.Linear(d_model, k)                 # cuello de botella: 64 -> k
        self.desde_latente = nn.Linear(k, d_model)             # vuelta: k -> 64
        self.decoder = nn.TransformerEncoder(capa_transformer(d_model), num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def encode(self, x):                                       # (B, 60, 14)
        h = self.encoder(self.pos_enc(self.embed(x)))          # (B, 60, 64)
        return self.a_latente(h)                               # (B, 60, k)

    def forward(self, x):
        h = self.pos_enc(self.desde_latente(self.encode(x)))   # (B, 60, 64)
        return self.salida(self.decoder(h))                    # (B, 60, 14)


class AECNNT(nn.Module):
    """Convolucional 1D sin reducir el tiempo (paso 1, relleno 'same')."""
    def __init__(self, n_vars, W, k, canales=32, nucleo=5, dropout=0.1):
        super().__init__()
        p = nucleo // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(n_vars, canales, nucleo, padding=p), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(canales, k, nucleo, padding=p))         # cuello de botella: 32 -> k
        self.decoder = nn.Sequential(
            nn.Conv1d(k, canales, nucleo, padding=p), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(canales, n_vars, nucleo, padding=p))

    def encode(self, x):                                       # (B, 60, 14)
        return self.encoder(x.transpose(1, 2)).transpose(1, 2)  # (B, 60, k)

    def forward(self, x):
        z = self.encode(x).transpose(1, 2)                     # (B, k, 60)
        return self.decoder(z).transpose(1, 2)                 # (B, 60, 14)


class AECNNLSTM(nn.Module):
    def __init__(self, n_vars, W, k, canales=32, oculto=64, nucleo=5, dropout=0.1):
        super().__init__()
        p = nucleo // 2
        self.conv_in = nn.Sequential(nn.Conv1d(n_vars, canales, nucleo, padding=p), nn.ReLU())
        self.lstm_enc = nn.LSTM(canales, oculto, batch_first=True)
        self.a_latente = nn.Linear(oculto, k)                  # cuello de botella: 64 -> k
        self.desde_latente = nn.Linear(k, oculto)
        self.lstm_dec = nn.LSTM(oculto, oculto, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(oculto, n_vars, nucleo, padding=p)

    def encode(self, x):                                       # (B, 60, 14)
        h = self.conv_in(x.transpose(1, 2)).transpose(1, 2)    # (B, 60, 32)
        h, _ = self.lstm_enc(h)                                # (B, 60, 64)
        return self.a_latente(self.drop(h))                    # (B, 60, k)

    def forward(self, x):
        h, _ = self.lstm_dec(self.desde_latente(self.encode(x)))   # (B, 60, 64)
        return self.conv_out(self.drop(h).transpose(1, 2)).transpose(1, 2)   # (B, 60, 14)


CLASES = {'transformer': AETransformer, 'transformer_t': AETransformerT,
          'cnn_t': AECNNT, 'cnn_lstm': AECNNLSTM}


def crear(red, k):
    return CLASES[red](N_VARS, W, k).to(device)


# Las formas, para mostrar que el latente conserva el tiempo
print('\nFormas con una ventana de prueba, x = (1, 60, 14):')
_x = torch.zeros(1, W, N_VARS, device=device)
for _r in [REF] + REDES:
    _k = K_REF if _r == REF else LATENTES[0]
    _m = crear(_r, _k).eval()
    with torch.no_grad():
        _z, _y = _m.encode(_x), _m(_x)
    print(f'  {_r:<14} latente {tuple(_z.shape)!s:<14} salida {tuple(_y.shape)}   '
          f'({sum(p.numel() for p in _m.parameters()):,} parámetros con k = {_k})')


# =============================================================================
# 3. MÉTRICAS
# =============================================================================
def errores_por_variable(modelo, X, batch=256):
    """(n, 14): error cuadrático medio de cada variable en cada ventana."""
    modelo.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.from_numpy(X[i:i + batch]).to(device)
            out.append(((modelo(b) - b) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


def auc(puntaje, positivo):
    """AUC por rangos (Mann-Whitney); los empates valen 0,5."""
    positivo = np.asarray(positivo, dtype=bool)
    n1, n0 = positivo.sum(), (~positivo).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(puntaje).rank(method='average').to_numpy()
    return (r[positivo].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def anticipacion(e_va, e_te):
    """Definición del paso 06: ventanas quietas, alarma en la ventana siguiente."""
    u = np.percentile(e_va[~al_va], PERCENTIL_UMBRAL)
    q = (~al_te) & ~np.isnan(fut_te)
    y = fut_te[q].astype(bool)
    marca = e_te[q] > u
    p1 = y[marca].mean() if marca.any() else np.nan
    p0 = y[~marca].mean() if (~marca).any() else np.nan
    return {'auc_anticipacion': auc(e_te[q], y),
            'rr_10min': p1 / p0 if p0 and p0 > 0 else np.nan}


def fila_metricas(e_va, e_te):
    return dict({'auc_val': auc(e_va, al_va), 'auc_test': auc(e_te, al_te),
                 'err_normal_val': float(np.median(e_va[~al_va])),
                 'veces_anomala_normal_val': float(np.median(e_va[al_va]) / np.median(e_va[~al_va]))},
                **anticipacion(e_va, e_te))


# =============================================================================
# 4. ENTRENAMIENTO — el mismo protocolo del paso 03b
# =============================================================================
def entrenar(red, k, semilla):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    modelo = crear(red, k)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    gen = torch.Generator().manual_seed(semilla)
    loader = DataLoader(VD(X_tr_n), batch_size=64, shuffle=True, generator=gen)
    t0, mejor_perdida, mejor_epoca, mejor_estado, espera = time.time(), np.inf, 0, None, 0
    for ep in range(1, EPOCAS_MAX + 1):
        modelo.train()
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            loss = crit(modelo(b), b)
            loss.backward()
            opt.step()
        perdida_val = float(errores_por_variable(modelo, X_va_n).mean())
        if perdida_val < mejor_perdida - 1e-5:
            mejor_perdida, mejor_epoca, espera = perdida_val, ep, 0
            mejor_estado = {kk: v.detach().clone() for kk, v in modelo.state_dict().items()}
        else:
            espera += 1
        if ep % 10 == 0 or ep == 1:
            print(f'    época {ep:>3}  pérdida normales val = {perdida_val:.5f}')
        if espera >= PACIENCIA:
            print(f'    parada temprana en la época {ep} (mínimo en la {mejor_epoca})')
            break
    modelo.load_state_dict(mejor_estado)
    ev_va, ev_te = errores_por_variable(modelo, X_va), errores_por_variable(modelo, X_te)
    return modelo, ev_va, ev_te, {
        'red': red, 'latent_dim': k, 'semilla': semilla,
        'forma_latente': f'({k})' if red == REF else f'(60, {k})',
        'numeros_latente': k if red == REF else W * k,
        'compresion': round(W * N_VARS / (k if red == REF else W * k), 2),
        'parametros': sum(p.numel() for p in modelo.parameters()),
        'epocas_corridas': ep, 'mejor_epoca': mejor_epoca,
        'perdida_val': round(mejor_perdida, 6),
        'minutos': round((time.time() - t0) / 60, 2)}


def nombre(red, k, semilla):
    return f'{red}_k{k}_s{semilla}'


if os.path.exists(ARCHIVO_SALIDA):
    hechas = pd.read_csv(ARCHIVO_SALIDA, sep=';', decimal=',')
    _otro = hechas.T_train.astype(str) != str(T_TRAIN)
    if _otro.any():
        print(f'{int(_otro.sum())} filas guardadas con otro corte: se vuelven a correr.')
        hechas = hechas[~_otro].reset_index(drop=True)
else:
    hechas = pd.DataFrame()
ya = set(zip(hechas.get('red', []), hechas.get('latent_dim', []), hechas.get('semilla', [])))

CORRIDAS = ([(REF, K_REF, s) for s in SEMILLAS]
            + [(r, k, s) for r in REDES for k in LATENTES for s in SEMILLAS])
print('\n' + '=' * 72)
print(f'4. {len(CORRIDAS)} corridas: el modelo actual + {len(REDES)} redes x '
      f'{len(LATENTES)} valores de k x {len(SEMILLAS)} semilla(s)')
print('=' * 72)
for red, k, semilla in CORRIDAS:
    clave = nombre(red, k, semilla)
    if (red, k, semilla) in ya and os.path.exists(f'{CARPETA_ERRORES}/{clave}.npz'):
        print(f'[salto] {clave}')
        continue
    print(f'\n--- {red} · latent_dim {k} · semilla {semilla} ---')
    modelo, ev_va, ev_te, info = entrenar(red, k, semilla)
    torch.save(modelo.state_dict(), f'{CARPETA_MODELOS}/{clave}.pt')
    np.savez_compressed(f'{CARPETA_ERRORES}/{clave}.npz',
                        ev_va=ev_va.astype(np.float32), ev_te=ev_te.astype(np.float32))
    fila = dict(info, **{kk: round(float(v), 5) for kk, v in
                         fila_metricas(ev_va.mean(axis=1), ev_te.mean(axis=1)).items()},
                T_train=str(T_TRAIN))
    hechas = pd.concat([hechas, pd.DataFrame([fila])], ignore_index=True) if len(hechas) else pd.DataFrame([fila])
    hechas = hechas.drop_duplicates(['red', 'latent_dim', 'semilla'], keep='last')
    hechas.to_csv(ARCHIVO_SALIDA, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'    pérdida normales val {fila["perdida_val"]}  AUC val {fila["auc_val"]}  '
          f'AUC prueba {fila["auc_test"]}  ({fila["minutos"]} min)  [guardado]')

hechas = hechas[hechas.red.isin([REF] + REDES) & hechas.semilla.isin(SEMILLAS)
                & (hechas.latent_dim.isin(LATENTES) | (hechas.red == REF))]


# =============================================================================
# 5. LA CURVA DEL CODO Y LA ELECCIÓN DE k (en validación)
# =============================================================================
print('\n' + '=' * 72)
print('5. PÉRDIDA DE NORMALES Y AUC CONTRA latent_dim (validación)')
print('=' * 72)
METRICAS = ['perdida_val', 'auc_val', 'auc_test', 'veces_anomala_normal_val',
            'auc_anticipacion', 'rr_10min']
prom = (hechas.groupby(['red', 'latent_dim'])[METRICAS + ['parametros', 'compresion']]
        .mean().round(4).reset_index())
pd.set_option('display.width', 220)
print(prom.to_string(index=False))
ref = prom[prom.red == REF]

eleccion = []
for red in REDES:
    p = prom[prom.red == red].sort_values('latent_dim')
    if p.empty:
        continue
    mejor = p.perdida_val.min()
    k_codo = int(p[p.perdida_val <= mejor * (1 + TOLERANCIA_CODO)].latent_dim.min())
    k_auc = int(p.loc[p.auc_val.idxmax(), 'latent_dim'])
    eleccion.append({'red': red, 'k_codo': k_codo, 'k_auc': k_auc})
    print(f'  {red:<14} k del codo = {k_codo} (pérdida a menos de {TOLERANCIA_CODO:.0%} de la mejor) | '
          f'k con mayor AUC de validación = {k_auc}')
eleccion = pd.DataFrame(eleccion)

fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
for red in REDES:
    p = prom[prom.red == red].sort_values('latent_dim')
    if p.empty:
        continue
    ejes[0].plot(p.latent_dim, p.perdida_val, 'o-', label=red)
    ejes[1].plot(p.latent_dim, p.auc_val, 'o-', label=red)
if len(ref):
    ejes[0].axhline(ref.perdida_val.iloc[0], color='k', ls='--', lw=1, label='anterior (64, sin tiempo)')
    ejes[1].axhline(ref.auc_val.iloc[0], color='k', ls='--', lw=1, label='anterior (64, sin tiempo)')
ejes[0].set(xlabel='latent_dim (k)', ylabel='Pérdida en ventanas normales de validación (MSE)',
            title='Curva del codo')
ejes[1].set(xlabel='latent_dim (k)', ylabel='AUC de detección, validación',
            title='¿Detecta mejor?')
for a in ejes:
    a.set_xticks(LATENTES)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
fig.suptitle(f'{EQUIPO}: latente (60, k) contra el anterior (64, sin tiempo)')
fig.tight_layout()
fig.savefig(f'{RUTA_EQ}/03a_codo.png', dpi=150)
plt.close(fig)
print(f'Gráfico: {RUTA_EQ}/03a_codo.png')


# =============================================================================
# 6. MEJOR k DE CADA RED CONTRA EL MODELO ACTUAL (intervalo bootstrap)
# =============================================================================
print('\n' + '=' * 72)
print('6. MEJOR k (por AUC de validación) DE CADA RED CONTRA EL ANTERIOR')
print('=' * 72)


def errores_ventana(red, k, conj):
    sub = hechas[(hechas.red == red) & (hechas.latent_dim == k)]
    return [np.load(f'{CARPETA_ERRORES}/{nombre(red, k, int(s))}.npz')[conj].mean(axis=1)
            for s in sub.semilla]


def bootstrap(errs, alarma, tramo, ref_key):
    grupos = pd.Series(np.arange(len(tramo))).groupby(tramo).indices
    claves = np.array(list(grupos))
    rng = np.random.default_rng(42)
    boot = {c: [] for c in errs}
    for _ in range(N_BOOT):
        sel = np.concatenate([grupos[c] for c in rng.choice(claves, len(claves))])
        for c, lista in errs.items():
            boot[c].append(np.nanmean([auc(e[sel], alarma[sel]) for e in lista]))
    return {c: np.nanpercentile(np.array(boot[c]) - np.array(boot[ref_key]), [2.5, 97.5])
            for c in errs if c != ref_key}


filas_res = []
if len(ref) and len(eleccion):
    configs = {(REF, K_REF)} | {(r.red, r.k_auc) for r in eleccion.itertuples()}
    for conj, alarma, tramo, etiqueta in [('ev_va', al_va, tr_va, 'val'), ('ev_te', al_te, tr_te, 'test')]:
        errs = {c: errores_ventana(*c, conj) for c in configs}
        ic = bootstrap(errs, alarma, tramo, (REF, K_REF))
        for c in configs:
            if c == (REF, K_REF):
                continue
            a = np.mean([auc(e, alarma) for e in errs[c]])
            b = np.mean([auc(e, alarma) for e in errs[(REF, K_REF)]])
            filas_res.append({'red': c[0], 'latent_dim': c[1], 'conjunto': etiqueta,
                              'auc': round(a, 4), 'auc_anterior': round(b, 4), 'dif': round(a - b, 4),
                              'ic95_inf': round(float(ic[c][0]), 4), 'ic95_sup': round(float(ic[c][1]), 4)})
resumen = pd.DataFrame(filas_res)
if len(resumen):
    resumen['lectura'] = np.where(resumen.ic95_inf > 0, 'mejor que el anterior',
                                  np.where(resumen.ic95_sup < 0, 'peor que el anterior', 'empate'))
    print(resumen.to_string(index=False))
    resumen = resumen.merge(eleccion, on='red', how='left')
    resumen.to_csv(f'{RUTA_EQ}/03a_resumen.csv', sep=';', decimal=',', index=False, encoding='utf-8-sig')
print('Validación decide; prueba solo se informa. Con una semilla es indicativo: para')
print('decidir, correr el k elegido con SEMILLAS=42,7,2024 y aplicar la regla: 2 veces la dispersión e IC sin el 0.')


# =============================================================================
# 7. ERROR POR VARIABLE: ¿QUÉ VARIABLES EXPLICAN QUE LA VENTANA SEA ANÓMALA?
# =============================================================================
print('\n' + '=' * 72)
print('7. ERROR POR VARIABLE: ¿LA VARIABLE CON MÁS ERROR ES LA DE LA REGLA QUE SONÓ?')
print('=' * 72)


def atribucion(ev, reglas, alarma, ev_ref_quietas):
    top = ev.argmax(axis=1)
    dist0 = np.bincount(ev_ref_quietas.argmax(axis=1), minlength=N_VARS) / len(ev_ref_quietas)
    filas = []
    act_fam = {f: reglas[:, [IDS.index(r) for r in rs if r in IDS]].any(axis=1)
               for f, rs in FAMILIAS.items()}
    grupos = {'todas las anomalas': alarma}
    for f in FAMILIAS:
        otras = np.column_stack([act_fam[g] for g in FAMILIAS if g != f]).any(axis=1)
        grupos[f'solo {f}'] = act_fam[f] & ~otras
    for nombre_g, m in grupos.items():
        m = m & alarma
        if m.sum() == 0:
            continue
        vars_ok = (reglas[m].astype(int) @ R2V.astype(int)) > 0          # (n, 14)
        acierta = vars_ok[np.arange(m.sum()), top[m]]
        azar = (vars_ok * dist0).sum(axis=1)
        filas.append({'grupo': nombre_g, 'ventanas': int(m.sum()),
                      'acierta_pct': round(100 * acierta.mean(), 1),
                      'azar_pct': round(100 * azar.mean(), 1)})
    return filas


mejores = [(REF, K_REF)] + [(r.red, r.k_auc) for r in eleccion.itertuples()] if len(eleccion) else [(REF, K_REF)]
s0 = SEMILLAS[0]
filas_at, filas_var = [], []
for red, k in mejores:
    ruta = f'{CARPETA_ERRORES}/{nombre(red, k, s0)}.npz'
    if not os.path.exists(ruta):
        continue
    z = np.load(ruta)
    u_var = np.percentile(z['ev_va'][~al_va], PERCENTIL_UMBRAL, axis=0)     # umbral por variable
    for conj, ev, reglas, alarma, ts in [('val', z['ev_va'], rg_va, al_va, ts_va),
                                         ('test', z['ev_te'], rg_te, al_te, ts_te)]:
        for f in atribucion(ev, reglas, alarma, z['ev_va'][~al_va]):
            filas_at.append(dict(red=red, latent_dim=k, conjunto=conj, **f))
        sobre = ev > u_var
        orden = np.argsort(-ev, axis=1)
        for i in np.where(alarma)[0]:
            filas_var.append({
                'red': red, 'latent_dim': k, 'conjunto': conj, 'inicio_ventana': ts[i],
                'error_ventana': round(float(ev[i].mean()), 6),
                'var_1': VARIABLES[orden[i, 0]], 'var_2': VARIABLES[orden[i, 1]],
                'var_3': VARIABLES[orden[i, 2]],
                'aporte_var_1_pct': round(100 * float(ev[i, orden[i, 0]] / ev[i].sum()), 1),
                'variables_sobre_su_umbral': ', '.join(np.array(VARIABLES)[sobre[i]]),
                'reglas_que_sonaron': ', '.join(np.array(IDS)[reglas[i]])})
at = pd.DataFrame(filas_at)
if len(at):
    print(at.to_string(index=False))
    at.to_csv(f'{RUTA_EQ}/03a_atribucion.csv', sep=';', decimal=',', index=False, encoding='utf-8-sig')
    pd.DataFrame(filas_var).to_csv(f'{RUTA_EQ}/03a_variables_por_ventana.csv', sep=';', decimal=',',
                                   index=False, encoding='utf-8-sig')
print('\nCómo leerlo: si "acierta" está muy por encima del azar, el error por variable')
print('señala la variable de la regla que sonó: sirve para decir qué revisar. La prueba')
print('equivalente con el índice W dio 71 a 76 % en lubricación, contra 4 a 6 % de azar.')
print(f'\nGuardado: {ARCHIVO_SALIDA}, 03a_resumen.csv, 03a_atribucion.csv,')
print(f'03a_variables_por_ventana.csv, 03a_codo.png y las carpetas 03a_errores y 03a_modelos.')
