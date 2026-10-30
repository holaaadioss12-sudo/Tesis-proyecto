# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04d — CONFIRMACIÓN CON VARIAS SEMILLAS
=============================================================================
"""

from google.colab import drive
drive.mount('/content/drive')
RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
_os.makedirs(RUTA_EQ, exist_ok=True)
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import sys
sys.path.append(RUTA_BASE)

import os
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from reglas_engine import evaluar_reglas

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', device)
if device.type != 'cuda':
    print('AVISO: sin GPU esto tarda muchísimo. Activá el entorno con GPU en')
    print('       Entorno de ejecución > Cambiar tipo de entorno de ejecución.')

W = 60
STRIDE_TRAIN = 12
STRIDE_EVAL = W
EPOCAS_MAX = 100
PACIENCIA = 15
SEMILLAS = [42, 7, 2024]
ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_SALIDA = f'{RUTA_EQ}/04d_confirmacion_semillas_v2.csv'
ARCHIVO_RESUMEN = f'{RUTA_EQ}/04d_resumen_v2.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

BASE = {'lr': 1e-3, 'num_layers': 2, 'd_model': 64, 'nhead': 4, 'dim_ff': 128}
ARCHIVO_04C = f'{RUTA_EQ}/04c_barrido_hiperparametros_v2.csv'
if not os.path.exists(ARCHIVO_04C):
    raise SystemExit(f'Falta {ARCHIVO_04C}: correr primero el 04c parchado.')
_b04c = pd.read_csv(ARCHIVO_04C, sep=';', decimal=',')
_b04c = (_b04c[_b04c.configuracion != 'base (la actual)']
         .sort_values('err_normal').head(2))
CONFIGS = [dict(BASE, nombre='base (la actual)')]
for _, _r in _b04c.iterrows():
    CONFIGS.append(dict(BASE, lr=float(_r.lr), num_layers=int(_r.capas),
                        d_model=int(_r.d_model), dim_ff=2 * int(_r.d_model),
                        nombre=_r.configuracion))
print('Configuraciones a confirmar:', [c['nombre'] for c in CONFIGS])

# =============================================================================
# 1. DATOS — el mismo pipeline del paso 04 y del 04c
# =============================================================================
print('\n' + '=' * 72)
print('1. PREPARANDO LOS DATOS (una sola vez)')
print('=' * 72)

np.random.seed(42)


def cargar_limpio(ruta):
    cabecera = pd.read_csv(ruta, nrows=0)
    col = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col], low_memory=False)
    return df.set_index(col).sort_index().rename_axis('timestamp')


df = cargar_limpio(ARCHIVO_LIMPIO)
VARIABLES = [v for v in VARIABLES if v in df.columns]
print(f'Filas: {len(df):,}   Variables: {len(VARIABLES)}')

# -----------------------------------------------------------------------------
import json
_corte = json.load(open(f'{RUTA_EQ}/{EQUIPO}_corte_split.json'))
_T_TRAIN, _T_VAL = pd.Timestamp(_corte['T_train']), pd.Timestamp(_corte['T_val'])
_f_tr = _corte.get('frac_train_efectiva')
print(f'Corte único: T_train = {_T_TRAIN} | T_val = {_T_VAL} | '
      f'frac_train_efectiva = {_f_tr}')
if _f_tr is None or abs(_f_tr - _corte.get('frac_train', 0.70)) > 0.05:
    raise SystemExit('El corte del JSON está desviado (o no dice dónde está). '
                     'Correr 01 -> 05 con el pipeline parchado antes de esto.')

idx = df.index.to_series()
df['_tramo'] = (idx.diff() > pd.Timedelta('10s')).cumsum().values
tramos, mascaras = [], []
for _, g in df.groupby('_tramo'):
    if len(g) >= W:
        t = g.drop(columns='_tramo')
        r = evaluar_reglas(t, verbose=False)
        ids = [c for c in r.columns if c not in ('n_reglas_activadas', 'estado')]
        tramos.append(t)
        mascaras.append(pd.Series(r[ids].to_numpy().any(axis=1), index=t.index))
print(f'Tramos útiles: {len(tramos):,}')
_n_al = sum(int(m.sum()) for m in mascaras)
_n_tot = sum(len(m) for m in mascaras)
print(f'Muestras en estado de alarma: {_n_al:,} ({100 * _n_al / _n_tot:.1f} %)')

_ini = np.array([t.index[0] for t in tramos])
_s_tr = _ini <= _T_TRAIN
_s_va = (_ini > _T_TRAIN) & (_ini <= _T_VAL)
tr_t = [t for t, k in zip(tramos, _s_tr) if k]
va_t = [t for t, k in zip(tramos, _s_va) if k]
tr_m = [m for m, k in zip(mascaras, _s_tr) if k]
va_m = [m for m, k in zip(mascaras, _s_va) if k]
print(f'Tramos: train={len(tr_t)} | val={len(va_t)}   '
      f'(prueba no se toca en este experimento)')

concat = pd.concat(tr_t)[VARIABLES]
media_n, std_n = concat.mean(), concat.std().replace(0, 1.0)


def construir(tramos_, mascaras_, stride):
    X, normal = [], []
    for t, m in zip(tramos_, mascaras_):
        arr = ((t[VARIABLES] - media_n) / std_n).to_numpy(dtype=np.float32)
        mv = m.to_numpy()
        for i in range(0, len(arr) - W + 1, stride):
            X.append(arr[i:i + W])
            normal.append(not mv[i:i + W].any())
    return np.stack(X), np.array(normal)


X_tr, nm_tr = construir(tr_t, tr_m, STRIDE_TRAIN)
X_va, nm_va = construir(va_t, va_m, STRIDE_EVAL)
X_tr_n = X_tr[nm_tr]
X_va_n, X_va_a = X_va[nm_va], X_va[~nm_va]
print(f'Entrenamiento: {len(X_tr):,} ventanas, {len(X_tr_n):,} normales')
print(f'Validación   : {len(X_va):,} ventanas, {len(X_va_n):,} normales, '
      f'{len(X_va_a):,} anómalas')


class VD(Dataset):
    def __init__(self, X): self.X = torch.from_numpy(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


# =============================================================================
# 2. EL MODELO — idéntico al paso 04
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
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
        cap = lambda: nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(cap(), num_layers=num_layers)
        self.decoder = nn.TransformerEncoder(cap(), num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def forward(self, x):
        h = self.encoder(self.pos_enc(self.embed(x))).mean(dim=1)
        h = h.unsqueeze(1).repeat(1, self.W, 1)
        return self.salida(self.decoder(self.pos_enc(h)))


def errores(modelo, X, batch=256):
    modelo.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.from_numpy(X[i:i + batch]).to(device)
            out.extend(((modelo(b) - b) ** 2).mean(dim=(1, 2)).cpu().tolist())
    return np.array(out)


def auc_mann_whitney(e_norm, e_anom):
    """AUC sin sklearn: el estadístico U normalizado."""
    todos = np.concatenate([e_norm, e_anom])
    rangos = pd.Series(todos).rank(method='average').to_numpy()
    r_anom = rangos[len(e_norm):].sum()
    na, nn_ = len(e_anom), len(e_norm)
    return (r_anom - na * (na + 1) / 2) / (na * nn_)


def metricas(e_norm, e_anom):
    umbral = np.percentile(e_norm, 90)        # el P90 que ya se usa
    return {
        'separacion_mediana': np.median(e_anom) / np.median(e_norm),
        'separacion_media': e_anom.mean() / e_norm.mean(),
        'auc': auc_mann_whitney(e_norm, e_anom),
        'recall_al_10pct': (e_anom > umbral).mean(),
        'err_normal': float(np.median(e_norm)),
        'err_anomala': float(np.median(e_anom)),
    }


def entrenar(cfg, semilla):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    modelo = AutoencoderTransformer(
        n_vars=len(VARIABLES), W=W, d_model=cfg['d_model'], nhead=cfg['nhead'],
        num_layers=cfg['num_layers'], dim_ff=cfg['dim_ff']).to(device)
    n_param = sum(p.numel() for p in modelo.parameters())
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg['lr'])
    crit = nn.MSELoss()
    loader = DataLoader(VD(X_tr_n), batch_size=64, shuffle=True)

    t0 = time.time()
    mejor = None
    espera = 0
    for ep in range(1, EPOCAS_MAX + 1):
        modelo.train()
        perdida = 0.0
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            loss = crit(modelo(b), b)
            loss.backward()
            opt.step()
            perdida += loss.item() * b.size(0)
        perdida /= len(X_tr_n)

        e_n = errores(modelo, X_va_n)
        m = metricas(e_n, errores(modelo, X_va_a))
        m['perdida_val'] = float(e_n.mean())
        if mejor is None or m['perdida_val'] < mejor['perdida_val'] - 1e-5:
            mejor = dict(m, epoca=ep, perdida_train=perdida)
            espera = 0
        else:
            espera += 1
        if ep % 5 == 0 or ep == 1:
            print(f'    época {ep:>2}  train={perdida:.5f}  '
                  f'sep_mediana={m["separacion_mediana"]:5.2f}  '
                  f'sep_media={m["separacion_media"]:6.2f}  '
                  f'AUC={m["auc"]:.4f}  recall@10%={m["recall_al_10pct"]:.3f}')
        if espera >= PACIENCIA:
            print(f'    early stopping en la época {ep} (sin mejora en {PACIENCIA})')
            break

    return {
        'configuracion': cfg['nombre'], 'semilla': semilla,
        'lr': cfg['lr'], 'capas': cfg['num_layers'], 'd_model': cfg['d_model'],
        'parametros': n_param, 'epocas_corridas': ep,
        'mejor_epoca': mejor['epoca'],
        'perdida_val': round(mejor['perdida_val'], 6),
        'separacion_mediana': round(mejor['separacion_mediana'], 4),
        'separacion_media': round(mejor['separacion_media'], 4),
        'auc': round(mejor['auc'], 5),
        'recall_al_10pct': round(mejor['recall_al_10pct'], 4),
        'err_normal': round(mejor['err_normal'], 6),
        'err_anomala': round(mejor['err_anomala'], 6),
        'perdida_train': round(mejor['perdida_train'], 6),
        'minutos': round((time.time() - t0) / 60, 2),
    }


# =============================================================================
# 3. LAS CORRIDAS — con reanudación si Colab se corta
# =============================================================================
print('\n' + '=' * 72)
print(f'3. {len(CONFIGS)} configuraciones x {len(SEMILLAS)} semillas x '
      f'hasta {EPOCAS_MAX} épocas = {len(CONFIGS)*len(SEMILLAS)} corridas')
print('=' * 72)

if os.path.exists(ARCHIVO_SALIDA):
    hechas = pd.read_csv(ARCHIVO_SALIDA, sep=';', decimal=',')
    if 'T_train' in hechas.columns:
        _otro = hechas.T_train.notna() & (hechas.T_train.astype(str) != str(_T_TRAIN))
        if _otro.any():
            print(f'  {int(_otro.sum())} filas guardadas con OTRO corte: se descartan '
                  f'y se vuelven a correr.')
            hechas = hechas[~_otro].reset_index(drop=True)
    else:
        print('  (filas sin T_train: anteriores a este parche, mismo archivo _v2)')
    ya = set(zip(hechas.configuracion, hechas.semilla))
    print(f'Ya había {len(ya)} corridas hechas. Se saltan.')
else:
    hechas = pd.DataFrame()
    ya = set()

for cfg in CONFIGS:
    for s in SEMILLAS:
        if (cfg['nombre'], s) in ya:
            print(f'\n[salto] {cfg["nombre"]} · semilla {s}')
            continue
        print(f'\n--- {cfg["nombre"]} · semilla {s} ---')
        fila = entrenar(cfg, s)
        fila['T_train'] = str(_T_TRAIN)
        hechas = pd.concat([hechas, pd.DataFrame([fila])], ignore_index=True)
        hechas.to_csv(ARCHIVO_SALIDA, sep=';', decimal=',', index=False,
                      encoding='utf-8-sig')
        print(f'    -> sep_mediana {fila["separacion_mediana"]}  '
              f'AUC {fila["auc"]}  ({fila["minutos"]} min)  [guardado]')

# =============================================================================
# 4. RESUMEN — promedio y dispersión por configuración
# =============================================================================
print('\n' + '=' * 72)
print('4. RESUMEN')
print('=' * 72)

METRICAS = ['separacion_mediana', 'separacion_media', 'auc',
            'recall_al_10pct', 'err_normal']
g = hechas.groupby('configuracion')[METRICAS].agg(['mean', 'std']).round(5)
print(g.to_string())

base_nom = 'base (la actual)'
if base_nom in set(hechas.configuracion):
    print('\n--- ¿La diferencia supera al ruido de la semilla? ---')
    b = hechas[hechas.configuracion == base_nom]
    for nombre, grupo in hechas.groupby('configuracion'):
        if nombre == base_nom:
            continue
        print(f'\n  {nombre}:')
        for m in METRICAS:
            dif = grupo[m].mean() - b[m].mean()
            # dispersión conjunta de las dos configuraciones
            disp = np.sqrt((grupo[m].std(ddof=1)**2 + b[m].std(ddof=1)**2) / 2)
            veces = dif / disp if disp > 0 else np.nan
            signo = '+' if dif >= 0 else ''
            if abs(veces) >= 2:
                fallo = 'DIFERENCIA CLARA'
            elif abs(veces) >= 1:
                fallo = 'tendencia, no concluyente'
            else:
                fallo = 'dentro del ruido'
            print(f'    {m:20s} {signo}{dif:+.5f}  '
                  f'({veces:+.1f} veces la dispersión)  -> {fallo}')

resumen = hechas.groupby('configuracion')[METRICAS + ['parametros']].mean().round(5)
resumen['corridas'] = hechas.groupby('configuracion').size()
resumen.to_csv(ARCHIVO_RESUMEN, sep=';', decimal=',', encoding='utf-8-sig')
print(f'\nGuardado: {ARCHIVO_SALIDA}')
print(f'Guardado: {ARCHIVO_RESUMEN}')
print('\nREGLA DE DECISIÓN: solo cambiar la configuración si la mejora supera')
print('2 veces la dispersión EN AUC y EN recall. Si solo mejora la separación')
print('pero no el AUC, no es una mejora de detección: es un cambio de escala.')
print('\nAun si alguna configuración pasa esa regla, el modelo de producción NO')
print('cambia: prueba ya se miró. Se reporta como sensibilidad, y una mejora')
print('clara queda como recomendación para un reentrenamiento futuro.')
