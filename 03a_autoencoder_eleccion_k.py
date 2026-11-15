# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 03a — AUTOENCODER TEMPORAL: ELECCIÓN DE latent_dim (k) Y DE LA ARQUITECTURA
=============================================================================
Entrena el autoencoder del 03b con varios k (y semillas) y elige k con una regla
fijada de antemano que solo usa validación (prueba solo se informa):
  1. mayor AUC de validación (promedio entre semillas);
  2. empate con el mejor si el IC 95 % de la diferencia (bootstrap por bloques de 6 h)
     incluye el 0, o si la diferencia es menor que 2 desviaciones entre semillas; entre
     los empatados, mayor ventaja de localización en validación (la variable que más se
     pasa de su umbral es de la regla que sonó, menos lo que acertaría el azar, como el 04c);
  3. si también empatan en localización, el menor k (más compresión).
Se corre por etapas y se retoma solo (lo ya entrenado no se repite):
  A. todos los k con SEMILLAS=42;
  B. SEMILLAS=42,7,2024 con los candidatos que imprime la etapa A;
  C. opcional: REDES=cnn_t,cnn_lstm con el k elegido.
Una corrida que llegó al tope de épocas sin activar la parada temprana se repite con el
tope actual (EPOCAS_MAX, 300 por defecto).
Entrada:  {EQUIPO}_limpio.csv y {EQUIPO}_corte_split.json (paso 01)
Salida:   03a_latente_temporal.csv (una fila por corrida), 03a_eleccion.csv (una fila por red y k,
          con la decisión), 03a_barrido.png, 03a_resumen.csv (contra el modelo anterior),
          03a_atribucion.csv, 03a_variables_por_ventana.csv, 03a_k_comun.csv (cuando ya están
          los tres equipos) y las carpetas 03a_errores y 03a_modelos
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
EPOCAS_MAX = int(os.environ.get('EPOCAS_MAX', 300))   # tope alto: que pare la parada temprana
PACIENCIA = 15
SEMILLAS = [int(s) for s in os.environ.get('SEMILLAS', '42').split(',')]
LATENTES = [int(k) for k in os.environ.get('LATENTES', '2,4,6,7,8,10,12').split(',')]
REDES = os.environ.get('REDES', 'transformer_t').split(',')
REF = 'transformer'                  # el modelo anterior: latente de 64 sin tiempo
K_REF = 64                           # su latente: 64 números, sin eje de tiempo
RED_OFICIAL = 'transformer_t'        # la del paso 03b
PERCENTIL_UMBRAL = 95
N_BOOT = 1000
BLOQUE = 36                          # ventanas por bloque del bootstrap: 36 x 10 min = 6 h (como el 04c)
EQUIPOS = ['CR009', 'CR010', 'CR011']
COLOR = {'transformer_t': '#0F9ED5', 'cnn_t': '#E97132', 'cnn_lstm': '#196B24', REF: '#7F7F7F'}
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
ARCHIVO_ELECCION = f'{RUTA_EQ}/03a_eleccion.csv'
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
bl_va = tr_va.astype(np.int64) * 100000 + po_va // BLOQUE     # bloques de 6 h para el bootstrap
bl_te = tr_te.astype(np.int64) * 100000 + po_te // BLOQUE
_t_mitad = ts_va.sort_values()[len(ts_va) // 2]
va_1, va_2 = np.asarray(ts_va < _t_mitad), np.asarray(ts_va >= _t_mitad)   # mitades cronológicas
if max(LATENTES) >= N_VARS:
    print(f'OJO: k = {max(LATENTES)} no comprime ({N_VARS} variables por instante); conviene k < {N_VARS}.')

_cfg = f'{RUTA_EQ}/autoencoder_W{W}_config.json'
K_ACTUAL = None
if os.path.exists(_cfg):
    _c = json.load(open(_cfg))
    K_ACTUAL = int(_c['latent_dim']) if _c.get('tipo') == 'temporal' else None
print(f'k del modelo oficial (03b): {K_ACTUAL if K_ACTUAL else "sin config temporal del 03b"}')


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


def ventaja_por_ventana(ev, reglas, alarma, ev_va):
    """En cada ventana con alarma: 100 si la variable que más se pasa de su umbral P95 es de una
    regla que sonó, menos lo que acertaría eligiendo como en las normales (el azar del 04c)."""
    ref = np.percentile(ev_va[~al_va], PERCENTIL_UMBRAL, axis=0)
    dist0 = np.bincount(np.argmax(ev_va[~al_va] - ref, axis=1), minlength=N_VARS) / int((~al_va).sum())
    top = np.argmax(ev - ref, axis=1)
    ok = (reglas.astype(int) @ R2V.astype(int)) > 0                   # variables de las reglas que sonaron
    v = 100 * (ok[np.arange(len(ev)), top] - (ok * dist0).sum(axis=1))
    return np.where(alarma, v, np.nan)


def metricas(ev_va, ev_te):
    """Todas las métricas de una corrida, desde el error de cada variable en cada ventana."""
    e_va, e_te = ev_va.mean(axis=1), ev_te.mean(axis=1)
    u = np.percentile(e_va[~al_va], PERCENTIL_UMBRAL)                # el umbral del paso 04
    out = {'auc_val': auc(e_va, al_va), 'auc_test': auc(e_te, al_te),
           'err_normal_val': float(np.median(e_va[~al_va])),
           'veces_anomala_normal_val': float(np.median(e_va[al_va]) / np.median(e_va[~al_va]))}
    for c, e, a in [('val', e_va, al_va), ('test', e_te, al_te)]:
        marca = e > u
        vp, fp, fn = int((marca & a).sum()), int((marca & ~a).sum()), int((~marca & a).sum())
        out[f'espec_{c}'] = 1 - fp / max(int((~a).sum()), 1)         # normales reconocidas como normales
        out[f'recall_{c}'] = vp / max(int(a.sum()), 1)
        out[f'precision_{c}'] = vp / (vp + fp) if vp + fp else np.nan
        out[f'f1_{c}'] = 2 * vp / (2 * vp + fp + fn) if vp + fp + fn else np.nan
    # estabilidad sin mirar prueba: umbral con la 1.a mitad de validación, especificidad en la 2.a
    n1, n2 = ~al_va & va_1, ~al_va & va_2
    out['espec_val_2mitad'] = (float((e_va[n2] <= np.percentile(e_va[n1], PERCENTIL_UMBRAL)).mean())
                               if n1.any() and n2.any() else np.nan)
    out['ventaja_val'] = float(np.nanmean(ventaja_por_ventana(ev_va, rg_va, al_va, ev_va)))
    out['ventaja_test'] = float(np.nanmean(ventaja_por_ventana(ev_te, rg_te, al_te, ev_va)))
    out.update(anticipacion(e_va, e_te))
    return out


METRICAS = ['perdida_val', 'auc_val', 'ventaja_val', 'espec_val', 'recall_val', 'precision_val', 'f1_val',
            'espec_val_2mitad', 'auc_test', 'espec_test', 'recall_test', 'precision_test', 'f1_test',
            'ventaja_test', 'veces_anomala_normal_val', 'auc_anticipacion', 'rr_10min']


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
        'epocas_corridas': ep, 'mejor_epoca': mejor_epoca, 'epocas_max': EPOCAS_MAX,
        'convergio': ep - mejor_epoca >= PACIENCIA,
        'perdida_val': round(mejor_perdida, 6),
        'minutos': round((time.time() - t0) / 60, 2)}


def nombre(red, k, semilla):
    return f'{red}_k{k}_s{semilla}'


def convergio(f):
    """La parada temprana se activó (PACIENCIA épocas sin mejorar): no la cortó el tope de épocas."""
    return (f.epocas_corridas - f.mejor_epoca) >= PACIENCIA


if os.path.exists(ARCHIVO_SALIDA):
    hechas = pd.read_csv(ARCHIVO_SALIDA, sep=';', decimal=',', encoding='utf-8-sig')
    _otro = hechas.T_train.astype(str) != str(T_TRAIN)
    if _otro.any():
        print(f'{int(_otro.sum())} filas guardadas con otro corte: se vuelven a correr.')
        hechas = hechas[~_otro].reset_index(drop=True)
else:
    hechas = pd.DataFrame()

# el modelo anterior solo con la primera semilla: es la línea base, no se elige
CORRIDAS = ([(REF, K_REF, SEMILLAS[0])]
            + [(r, k, s) for r in REDES for k in LATENTES for s in SEMILLAS])
print('\n' + '=' * 72)
print(f'4. {len(CORRIDAS)} corridas: el modelo anterior + {len(REDES)} red(es) x '
      f'{len(LATENTES)} valores de k x {len(SEMILLAS)} semilla(s) | tope {EPOCAS_MAX} épocas')
print('=' * 72)
for red, k, semilla in CORRIDAS:
    clave = nombre(red, k, semilla)
    previa = (hechas[(hechas.red == red) & (hechas.latent_dim == k) & (hechas.semilla == semilla)]
              if len(hechas) else hechas)
    if len(previa) and os.path.exists(f'{CARPETA_ERRORES}/{clave}.npz'):
        p = previa.iloc[-1]
        if convergio(p) or p.epocas_corridas >= EPOCAS_MAX:
            print(f'[salto] {clave}')
            continue
        print(f'[repite] {clave}: llegó al tope de {int(p.epocas_corridas)} épocas sin activar la parada '
              f'temprana; se entrena de nuevo con tope {EPOCAS_MAX}.')
    print(f'\n--- {red} · latent_dim {k} · semilla {semilla} ---')
    modelo, ev_va, ev_te, info = entrenar(red, k, semilla)
    torch.save(modelo.state_dict(), f'{CARPETA_MODELOS}/{clave}.pt')
    np.savez_compressed(f'{CARPETA_ERRORES}/{clave}.npz',
                        ev_va=ev_va.astype(np.float32), ev_te=ev_te.astype(np.float32))
    fila = dict(info, **{kk: round(float(v), 5) for kk, v in metricas(ev_va, ev_te).items()},
                T_train=str(T_TRAIN))
    hechas = pd.concat([hechas, pd.DataFrame([fila])], ignore_index=True) if len(hechas) else pd.DataFrame([fila])
    hechas = hechas.drop_duplicates(['red', 'latent_dim', 'semilla'], keep='last')
    hechas.to_csv(ARCHIVO_SALIDA, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'    pérdida normales val {fila["perdida_val"]}  AUC val {fila["auc_val"]}  '
          f'AUC prueba {fila["auc_test"]}  ({fila["minutos"]} min, '
          + ('convergió' if fila['convergio'] else f'NO convergió en {EPOCAS_MAX} épocas') + ')  [guardado]')

# las métricas de todas las corridas desde sus errores guardados (también las de antes de este cambio)
filas = []
for _, f in hechas.iterrows():
    ruta = f'{CARPETA_ERRORES}/{nombre(f.red, int(f.latent_dim), int(f.semilla))}.npz'
    if not os.path.exists(ruta):
        print(f'Falta {ruta}: esa corrida queda fuera del resumen.')
        continue
    z = np.load(ruta)
    if len(z['ev_va']) != len(X_va) or len(z['ev_te']) != len(X_te):
        print(f'{ruta} no calza con las ventanas actuales: queda fuera (córrela de nuevo).')
        continue
    f = f.copy()
    for kk, v in metricas(z['ev_va'], z['ev_te']).items():
        f[kk] = round(float(v), 5)
    f['convergio'] = bool(convergio(f))
    filas.append(f)
hechas = pd.DataFrame(filas).reset_index(drop=True)
if hechas.empty:
    raise SystemExit('No hay corridas con sus errores guardados: no hay con qué elegir k.')
hechas.to_csv(ARCHIVO_SALIDA, sep=';', decimal=',', index=False, encoding='utf-8-sig')


# =============================================================================
# 5. RESUMEN POR k (PROMEDIO ENTRE SEMILLAS) Y REGLA DE DECISIÓN (SOLO VALIDACIÓN)
# =============================================================================
print('\n' + '=' * 72)
print('5. ELECCIÓN DE k: AUC DE VALIDACIÓN, LUEGO LOCALIZACIÓN, LUEGO EL MENOR k')
print('=' * 72)
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)      # medias de remuestreos sin alarmas
pd.set_option('display.width', 220)

ok = hechas[hechas.convergio.astype(bool)]
if (~hechas.convergio.astype(bool)).any():
    print('No convergieron (quedan fuera de la decisión; córrelas con EPOCAS_MAX más alto):')
    print(hechas.loc[~hechas.convergio.astype(bool),
                     ['red', 'latent_dim', 'semilla', 'epocas_corridas', 'mejor_epoca']].to_string(index=False))
if ok.empty:
    raise SystemExit('Ninguna corrida convergió todavía: no hay con qué elegir k.')

_agg = {}
for c in METRICAS:
    _agg[c], _agg[c + '_sd'] = (c, 'mean'), (c, 'std')
tab = (ok.groupby(['red', 'latent_dim'])
         .agg(semillas=('semilla', 'nunique'), parametros=('parametros', 'first'),
              compresion=('compresion', 'first'), **_agg).reset_index())


def remuestreos(bloques, n):
    """Índices de N_BOOT remuestreos por bloques de 6 h (los mismos para todas las configuraciones)."""
    grupos = pd.Series(np.arange(n)).groupby(bloques).indices
    claves = np.array(list(grupos))
    rng = np.random.default_rng(42)
    return [np.concatenate([grupos[c] for c in rng.choice(claves, len(claves))]) for _ in range(N_BOOT)]


MUESTRAS = {'val': remuestreos(bl_va, len(X_va)), 'test': remuestreos(bl_te, len(X_te))}
ALARMA, REGLAS = {'val': al_va, 'test': al_te}, {'val': rg_va, 'test': rg_te}


def cargar(red, k, conj):
    """Error total y ventaja de localización por ventana, una lista por semilla convergida."""
    e, v = [], []
    for s in ok[(ok.red == red) & (ok.latent_dim == k)].semilla:
        z = np.load(f'{CARPETA_ERRORES}/{nombre(red, k, int(s))}.npz')
        ev = z['ev_va'] if conj == 'val' else z['ev_te']
        e.append(ev.mean(axis=1))
        v.append(ventaja_por_ventana(ev, REGLAS[conj], ALARMA[conj], z['ev_va']))
    return e, v


def boot(lista, conj, que):
    """Valor en cada remuestreo, promedio entre semillas."""
    a = ALARMA[conj]
    if que == 'auc':
        return np.array([np.nanmean([auc(x[s], a[s]) for x in lista]) for s in MUESTRAS[conj]])
    return np.array([np.nanmean([np.nanmean(x[s]) for x in lista]) for s in MUESTRAS[conj]])


def empate(dif_boot, dif, sd_a, sd_b, n_a, n_b):
    """Empate: el IC 95 % de la diferencia incluye el 0, o la diferencia no supera 2 desviaciones
    entre semillas (solo si ambos tienen 2 semillas o más)."""
    lo, hi = np.nanpercentile(dif_boot, [2.5, 97.5])
    ruido = 2 * np.nanmax([sd_a, sd_b]) if min(n_a, n_b) >= 2 else 0.0
    return bool(lo <= 0 <= hi or abs(dif) < ruido), lo, hi


partes = []
for red in [r for r in tab.red.unique() if r != REF]:
    t = tab[tab.red == red].sort_values('latent_dim').reset_index(drop=True)
    for c in ['dif_auc', 'ic_auc_inf', 'ic_auc_sup', 'dif_ventaja', 'ic_ven_inf', 'ic_ven_sup']:
        t[c] = np.nan
    t['empate_auc'], t['empate_ventaja'] = False, False
    datos = {int(k): cargar(red, int(k), 'val') for k in t.latent_dim}
    b_auc = {k: boot(d[0], 'val', 'auc') for k, d in datos.items()}
    b_ven = {k: boot(d[1], 'val', 'ventaja') for k, d in datos.items()}
    i1 = t.auc_val.idxmax()                                        # 1. mayor AUC de validación
    for i, f in t.iterrows():
        e, lo, hi = empate(b_auc[int(f.latent_dim)] - b_auc[int(t.latent_dim[i1])], f.auc_val - t.auc_val[i1],
                           f.auc_val_sd, t.auc_val_sd[i1], f.semillas, t.semillas[i1])
        t.loc[i, ['dif_auc', 'ic_auc_inf', 'ic_auc_sup']] = [f.auc_val - t.auc_val[i1], lo, hi]
        t.at[i, 'empate_auc'] = e
    t.at[i1, 'empate_auc'] = True                                  # el mejor siempre está en su grupo
    t1 = t[t.empate_auc]
    i2 = t1.ventaja_val.idxmax()                                   # 2. entre los empatados, localización
    for i, f in t1.iterrows():
        e, lo, hi = empate(b_ven[int(f.latent_dim)] - b_ven[int(t.latent_dim[i2])],
                           f.ventaja_val - t.ventaja_val[i2], f.ventaja_val_sd, t.ventaja_val_sd[i2],
                           f.semillas, t.semillas[i2])
        t.loc[i, ['dif_ventaja', 'ic_ven_inf', 'ic_ven_sup']] = [f.ventaja_val - t.ventaja_val[i2], lo, hi]
        t.at[i, 'empate_ventaja'] = e
    t.at[i2, 'empate_ventaja'] = True
    t2 = t[t.empate_ventaja]
    k_el = int(t2.latent_dim.min())                                # 3. el menor k
    t['elegido'] = t.latent_dim == k_el
    otros1 = [str(int(x)) for x in t1.latent_dim if x != k_el]
    otros2 = [str(int(x)) for x in t2.latent_dim if x != k_el]
    if len(t) == 1:
        motivo = 'es el único k con corridas que convergieron'
    elif len(t1) == 1:
        motivo = 'mayor AUC de validación; los demás k quedan claramente por debajo'
    elif len(t2) == 1:
        motivo = f'empata en AUC con k = {", ".join(otros1)} y localiza mejor que ellos'
    else:
        motivo = f'empata en AUC y en localización con k = {", ".join(otros2)}; es el menor k'
    t['motivo'] = np.where(t.elegido, motivo, '')
    t['lectura'] = np.select([t.elegido, t.empate_auc & t.empate_ventaja, t.empate_auc],
                             ['elegido', 'empata en todo; se prefiere el menor k', 'empata en AUC, localiza peor'],
                             'AUC claramente menor')
    t['k_actual'] = K_ACTUAL if red == RED_OFICIAL else np.nan
    partes.append(t)
    print(f'\n{red}:')
    print(t[['latent_dim', 'semillas', 'auc_val', 'auc_val_sd', 'ic_auc_inf', 'ic_auc_sup', 'ventaja_val',
             'perdida_val', 'espec_val_2mitad', 'auc_test', 'espec_test', 'lectura']].round(4).to_string(index=False))
    print(f'  -> k elegido = {k_el}: {motivo}.')
    cand = t1.sort_values('auc_val', ascending=False).latent_dim.astype(int).tolist()[:3]
    for x in [k_el] + ([K_ACTUAL] if red == RED_OFICIAL and K_ACTUAL in set(t.latent_dim) else []):
        if x not in cand:
            cand.append(x)
    if (t[t.latent_dim.isin(cand)].semillas < 3).any():
        print(f'  Para confirmar con semillas (etapa B): SEMILLAS=42,7,2024 '
              f'LATENTES={",".join(map(str, sorted(cand)))} REDES={red}')

el = pd.concat(partes + [tab[tab.red == REF].assign(lectura='modelo anterior (referencia)', elegido=False)],
               ignore_index=True)
COLS = (['red', 'latent_dim', 'semillas', 'elegido', 'lectura', 'motivo', 'auc_val', 'auc_val_sd', 'dif_auc',
         'ic_auc_inf', 'ic_auc_sup', 'empate_auc', 'ventaja_val', 'ventaja_val_sd', 'dif_ventaja', 'ic_ven_inf',
         'ic_ven_sup', 'empate_ventaja', 'perdida_val', 'perdida_val_sd', 'espec_val', 'espec_val_2mitad',
         'recall_val', 'precision_val', 'f1_val', 'auc_test', 'auc_test_sd', 'espec_test', 'espec_test_sd',
         'recall_test', 'precision_test', 'f1_test', 'ventaja_test', 'auc_anticipacion', 'rr_10min',
         'compresion', 'parametros', 'k_actual'])
el = el[[c for c in COLS if c in el.columns]]
el.round(5).to_csv(ARCHIVO_ELECCION, sep=';', decimal=',', index=False, encoding='utf-8-sig')
print('\nespec_*: normales reconocidas como normales con el umbral P95 (en validación es 0,95 por construcción).')
print('espec_val_2mitad: umbral con la 1.a mitad de validación y especificidad en la 2.a: estabilidad sin')
print('mirar prueba. Prueba solo se informa.')

f = el[(el.red == RED_OFICIAL) & (el.elegido == True)]
if len(f):
    k_el = int(f.latent_dim.iloc[0])
    print(f'\nk ELEGIDO PARA {EQUIPO} ({RED_OFICIAL}): {k_el} ({f.motivo.iloc[0]}).')
    if K_ACTUAL == k_el:
        print('Es el mismo k del 03b: no hay que reentrenar. Corre 04 en adelante con PUNTO_PARADA=perdida,')
        print('el mismo criterio con que se eligió k (época de menor pérdida, sin mirar alarmas).')
    elif K_ACTUAL is None:
        print(f'No hay configuración temporal del 03b: entrénalo con LATENT_DIM={k_el}.')
    else:
        print(f'El 03b usa k = {K_ACTUAL}: reentrena con LATENT_DIM={k_el} y corre 04 en adelante con '
              'PUNTO_PARADA=perdida.')


# =============================================================================
# 6. EL k ELEGIDO DE CADA RED CONTRA EL MODELO ANTERIOR (64, SIN TIEMPO)
# =============================================================================
print('\n' + '=' * 72)
print('6. EL k ELEGIDO CONTRA EL MODELO ANTERIOR (latente de 64 sin tiempo)')
print('=' * 72)
filas_res = []
elegidos = el[el.elegido == True]
if (ok.red == REF).any() and len(elegidos):
    for conj in ['val', 'test']:
        e_ref = cargar(REF, K_REF, conj)[0]
        b_ref = boot(e_ref, conj, 'auc')
        a_ref = np.mean([auc(x, ALARMA[conj]) for x in e_ref])
        for f in elegidos.itertuples():
            e = cargar(f.red, int(f.latent_dim), conj)[0]
            lo, hi = np.nanpercentile(boot(e, conj, 'auc') - b_ref, [2.5, 97.5])
            a = np.mean([auc(x, ALARMA[conj]) for x in e])
            filas_res.append({'red': f.red, 'latent_dim': int(f.latent_dim), 'conjunto': conj,
                              'auc': round(a, 4), 'auc_anterior': round(a_ref, 4), 'dif': round(a - a_ref, 4),
                              'ic95_inf': round(float(lo), 4), 'ic95_sup': round(float(hi), 4)})
resumen = pd.DataFrame(filas_res)
if len(resumen):
    resumen['lectura'] = np.where(resumen.ic95_inf > 0, 'mejor que el anterior',
                                  np.where(resumen.ic95_sup < 0, 'peor que el anterior', 'empate'))
    print(resumen.to_string(index=False))
    resumen.to_csv(f'{RUTA_EQ}/03a_resumen.csv', sep=';', decimal=',', index=False, encoding='utf-8-sig')
else:
    print('Falta el modelo anterior convergido o un k elegido: se omite.')


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


filas_at, filas_var = [], []
for red, k in [(REF, K_REF)] + [(f.red, int(f.latent_dim)) for f in elegidos.itertuples()]:
    sub = ok[(ok.red == red) & (ok.latent_dim == k)]
    if sub.empty:
        continue
    s0 = 42 if 42 in set(sub.semilla) else int(sub.semilla.min())
    z = np.load(f'{CARPETA_ERRORES}/{nombre(red, k, s0)}.npz')
    u_var = np.percentile(z['ev_va'][~al_va], PERCENTIL_UMBRAL, axis=0)     # umbral por variable
    for conj, ev, reglas, alarma, ts in [('val', z['ev_va'], rg_va, al_va, ts_va),
                                         ('test', z['ev_te'], rg_te, al_te, ts_te)]:
        for f in atribucion(ev, reglas, alarma, z['ev_va'][~al_va]):
            filas_at.append(dict(red=red, latent_dim=k, semilla=s0, conjunto=conj, **f))
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
print('señala la variable de la regla que sonó: sirve para decir qué revisar.')


# =============================================================================
# 8. GRÁFICO DEL BARRIDO
# =============================================================================
fig, ejes = plt.subplots(2, 2, figsize=(12, 8.5))
ref_t = el[el.red == REF]
PANELES = [('perdida_val', None, 'Pérdida en normales de validación (MSE)', 'Reconstrucción de lo normal'),
           ('auc_val', 'empate_auc', 'AUC de validación', '1.er criterio: detección'),
           ('ventaja_val', 'empate_ventaja', 'Ventaja de localización, validación (puntos)',
            '2.º criterio, entre los empatados: localización')]
for ax, (col, col_emp, ylab, titulo) in zip(ejes.flat, PANELES):
    for red in [r for r in el.red.unique() if r != REF]:
        t = el[el.red == red].sort_values('latent_dim')
        c = COLOR.get(red, '#595959')
        x, y = t.latent_dim.to_numpy(), t[col].to_numpy(dtype=float)
        ax.plot(x, y, '-', color=c, lw=2, label=red)
        ax.errorbar(x, y, yerr=t[col + '_sd'].fillna(0).to_numpy(dtype=float), fmt='none', ecolor=c,
                    capsize=3, lw=1)
        emp = t[col_emp].astype(bool).to_numpy() if col_emp else np.ones(len(t), bool)
        ax.scatter(x[emp], y[emp], s=55, color=c, zorder=3)
        ax.scatter(x[~emp], y[~emp], s=55, facecolor='white', edgecolor=c, linewidth=1.5, zorder=3)
        e = t.elegido.astype(bool).to_numpy()
        ax.scatter(x[e], y[e], s=220, marker='*', color='#1F1F1F', zorder=4,
                   label='k elegido' if red == RED_OFICIAL else None)
    if len(ref_t) and pd.notna(ref_t[col].iloc[0]):
        ax.axhline(ref_t[col].iloc[0], color=COLOR[REF], ls='--', lw=1.2, label='anterior (64, sin tiempo)')
    ax.set(xlabel='latent_dim (k)', ylabel=ylab, title=titulo)
    ax.set_xticks(sorted(el.loc[el.red != REF, 'latent_dim'].astype(int).unique()))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
ax = ejes[1, 1]
t = el[el.red == RED_OFICIAL].sort_values('latent_dim')
for col, etiqueta, c, estilo in [('auc_test', 'AUC de prueba', '#0F9ED5', '-'),
                                 ('espec_test', 'Especificidad de prueba', '#E97132', '-'),
                                 ('espec_val_2mitad', 'Especificidad, 2.ª mitad de validación', '#196B24', '--')]:
    ax.plot(t.latent_dim, t[col], estilo, color=c, marker='o', ms=6, lw=2, label=etiqueta)
ax.set(xlabel='latent_dim (k)', ylabel='Proporción', ylim=(0, 1.02),
       title=f'No deciden (solo se informan), {RED_OFICIAL}')
ax.set_xticks(sorted(t.latent_dim.astype(int).unique()))
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='lower left')
fig.suptitle(f'{EQUIPO}: elección de k (relleno = empata con el mejor; hueco = claramente peor; '
             'estrella = elegido; barras = desviación entre semillas)', fontsize=10)
fig.tight_layout()
fig.savefig(f'{RUTA_EQ}/03a_barrido.png', dpi=150)
plt.close(fig)
print(f'\nGráfico: {RUTA_EQ}/03a_barrido.png')


# =============================================================================
# 9. ¿UN k COMÚN PARA LOS TRES EQUIPOS? (cuando ya están los tres)
# =============================================================================
def carpeta(eq):
    if eq == 'CR010':
        return f'{RUTA_BASE}/CR010' if os.path.isdir(f'{RUTA_BASE}/CR010') else RUTA_BASE
    return f'{RUTA_BASE}/{eq}'


partes = []
for eq in EQUIPOS:
    ruta = f'{carpeta(eq)}/03a_eleccion.csv'
    if os.path.exists(ruta):
        t = pd.read_csv(ruta, sep=';', decimal=',', encoding='utf-8-sig')
        partes.append(t[t.red == RED_OFICIAL].assign(equipo=eq))
if len(partes) == len(EQUIPOS):
    print('\n' + '=' * 72)
    print('9. ¿UN k COMÚN PARA LOS TRES EQUIPOS?')
    print('=' * 72)
    t = pd.concat(partes, ignore_index=True)
    for c in ['empate_auc', 'empate_ventaja', 'elegido']:
        t[c] = t[c].astype(str).str.lower().eq('true')
    comun = (t.groupby('latent_dim')
               .agg(equipos=('equipo', 'nunique'), empata_auc_en=('empate_auc', 'sum'),
                    empata_localizacion_en=('empate_ventaja', 'sum'), elegido_en=('elegido', 'sum'),
                    auc_val_medio=('auc_val', 'mean'), ventaja_val_media=('ventaja_val', 'mean'),
                    auc_test_medio=('auc_test', 'mean'), espec_test_media=('espec_test', 'mean'))
               .reset_index())
    cand = comun[(comun.equipos == len(EQUIPOS)) & (comun.empata_auc_en == len(EQUIPOS))]
    if len(cand):
        mejor = cand.sort_values(['empata_localizacion_en', 'latent_dim'], ascending=[False, True]).iloc[0]
        k_comun = int(mejor.latent_dim)
        comun['k_comun'] = comun.latent_dim == k_comun
        print(comun.round(4).to_string(index=False))
        print(f'\nk común = {k_comun}: no pierde AUC de validación de forma clara en ningún equipo y empata en '
              f'localización en {int(mejor.empata_localizacion_en)} de {len(EQUIPOS)}.')
    else:
        comun['k_comun'] = False
        print(comun.round(4).to_string(index=False))
        print('\nNingún k empata con el mejor en los tres equipos: se usa el k elegido en cada uno.')
    print('k elegido por equipo: ' + ', '.join(f'{r.equipo} = {int(r.latent_dim)}'
                                              for r in t[t.elegido].itertuples()))
    comun.round(5).to_csv(f'{RUTA_BASE}/03a_k_comun.csv', sep=';', decimal=',', index=False, encoding='utf-8-sig')
else:
    print(f'\nk común: falta el 03a de {len(EQUIPOS) - len(partes)} equipo(s); se calcula cuando estén los tres.')

print(f'\nGuardado: {ARCHIVO_SALIDA}, {ARCHIVO_ELECCION}, 03a_barrido.png, 03a_resumen.csv,')
print('03a_atribucion.csv, 03a_variables_por_ventana.csv y las carpetas 03a_errores y 03a_modelos.')
