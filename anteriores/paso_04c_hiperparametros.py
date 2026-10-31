# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04c — BARRIDO DE HIPERPARÁMETROS
=============================================================================
"""

from google.colab import drive
drive.mount('/content/drive')
RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and _os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'     # CR010 en su subcarpeta, como en los demás pasos
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
import matplotlib.pyplot as plt

from reglas_engine import evaluar_reglas

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', device)
if device.type != 'cuda':
    print('AVISO: sin GPU esto va a tardar muchísimo. Activá el entorno con GPU')
    print('       en Entorno de ejecución > Cambiar tipo de entorno de ejecución.')

W = 60
STRIDE_TRAIN = 12
STRIDE_EVAL = W
EPOCAS_MAX = 100
PACIENCIA = 15
ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_SALIDA = f'{RUTA_EQ}/04c_barrido_hiperparametros_v2.csv'
ARCHIVO_FIG = f'{RUTA_EQ}/04c_barrido_v2.png'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

# --- la configuración actual, la que hay que superar -----------------------
BASE = {'lr': 1e-3, 'num_layers': 2, 'd_model': 64, 'nhead': 4, 'dim_ff': 128}

# --- el barrido: una variable a la vez -------------------------------------
CONFIGS = [dict(BASE, nombre='base (la actual)')]
for lr in (3e-4, 3e-3):
    CONFIGS.append(dict(BASE, lr=lr, nombre=f'lr = {lr:g}'))
for nl in (3, 4):
    CONFIGS.append(dict(BASE, num_layers=nl, nombre=f'capas = {nl}'))
for dm in (32, 128):
    # dim_ff se mantiene en 2x d_model, que es la proporción de la base
    CONFIGS.append(dict(BASE, d_model=dm, dim_ff=2 * dm, nombre=f'd_model = {dm}'))


# =============================================================================
# 1. DATOS — exactamente el mismo pipeline del paso 04
# =============================================================================
print('\n' + '=' * 72)
print('1. PREPARANDO LOS DATOS (una sola vez para todas las configuraciones)')
print('=' * 72)


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
# 2. EL MODELO — igual que el paso 04, pero con los tamaños parametrizados
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


def error_medio(modelo, X, batch=256):
    modelo.eval()
    errores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.from_numpy(X[i:i + batch]).to(device)
            errores.extend(((modelo(b) - b) ** 2).mean(dim=(1, 2)).cpu().tolist())
    return np.array(errores)


def auc_mann_whitney(e_norm, e_anom):
    """AUC sin sklearn: el estadístico U normalizado."""
    todos = np.concatenate([e_norm, e_anom])
    rangos = pd.Series(todos).rank(method='average').to_numpy()
    r_anom = rangos[len(e_norm):].sum()
    na, nn_ = len(e_anom), len(e_norm)
    return (r_anom - na * (na + 1) / 2) / (na * nn_)


# -----------------------------------------------------------------------------
def entrenar_y_evaluar(cfg):
    """Entrena una configuración y devuelve sus métricas."""
    torch.manual_seed(SEED)
    modelo = AutoencoderTransformer(
        n_vars=len(VARIABLES), W=W, d_model=cfg['d_model'], nhead=cfg['nhead'],
        num_layers=cfg['num_layers'], dim_ff=cfg['dim_ff']).to(device)
    n_param = sum(p.numel() for p in modelo.parameters())
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg['lr'])
    crit = nn.MSELoss()
    loader = DataLoader(VD(X_tr_n), batch_size=64, shuffle=True)

    t0 = time.time()
    mejor = {'perdida_val': float('inf')}
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

        e_n, e_a = error_medio(modelo, X_va_n), error_medio(modelo, X_va_a)
        perdida_val = float(e_n.mean())      # la pérdida de validación del paso 04
        e_norm, e_anom = float(np.median(e_n)), float(np.median(e_a))
        razon = e_anom / e_norm if e_norm > 0 else float('nan')
        auc = auc_mann_whitney(e_n, e_a)
        recall = float((e_a > np.percentile(e_n, 90)).mean())
        if perdida_val < mejor['perdida_val'] - 1e-5:
            mejor = {'epoca': ep, 'perdida_val': perdida_val, 'err_norm': e_norm,
                     'err_anom': e_anom, 'razon': razon, 'auc': auc,
                     'recall': recall, 'train': perdida}
            espera = 0
        else:
            espera += 1
        if ep == 1 or ep % 5 == 0:
            print(f'    época {ep:>3}  train={perdida:.5f}  val={perdida_val:.5f}  '
                  f'normal={e_norm:.5f}  razón={razon:5.2f}x  AUC={auc:.4f}')
        if espera >= PACIENCIA:
            print(f'    early stopping en la época {ep} (sin mejora en {PACIENCIA})')
            break

    if mejor['epoca'] == ep:
        print('    OJO: el mínimo cayó en la última época corrida; la pérdida')
        print('    seguía bajando y esta configuración quedó corta de épocas.')
    return {
        'configuracion': cfg['nombre'],
        'lr': cfg['lr'], 'capas': cfg['num_layers'], 'd_model': cfg['d_model'],
        'parametros': n_param,
        'epocas_corridas': ep,
        'mejor_epoca': mejor['epoca'],
        'err_normal': round(mejor['err_norm'], 6),
        'perdida_val': round(mejor['perdida_val'], 6),
        'auc': round(mejor['auc'], 5),
        'recall_al_10pct': round(mejor['recall'], 4),
        'separacion': round(mejor['razon'], 3),
        'err_anomala': round(mejor['err_anom'], 6),
        'perdida_train': round(mejor['train'], 6),
        'brecha_train_val': round(mejor['perdida_val'] - mejor['train'], 6),
        'minutos': round((time.time() - t0) / 60, 1),
    }


# =============================================================================
# 3. EL BARRIDO — con reanudación si Colab se corta
# =============================================================================
print('\n' + '=' * 72)
print(f'3. BARRIDO — {len(CONFIGS)} configuraciones, hasta {EPOCAS_MAX} épocas '
      f'cada una (early stopping, paciencia {PACIENCIA})')
print('=' * 72)

if os.path.exists(ARCHIVO_SALIDA):
    hechas = pd.read_csv(ARCHIVO_SALIDA, sep=';', decimal=',')
    ya = set(hechas.configuracion)
    print(f'Ya había {len(ya)} configuraciones hechas. Se saltan.')
else:
    hechas = pd.DataFrame()
    ya = set()

for cfg in CONFIGS:
    if cfg['nombre'] in ya:
        print(f'\n[salto] {cfg["nombre"]}')
        continue
    print(f'\n--- {cfg["nombre"]} ---')
    fila = entrenar_y_evaluar(cfg)
    hechas = pd.concat([hechas, pd.DataFrame([fila])], ignore_index=True)
    hechas.to_csv(ARCHIVO_SALIDA, sep=';', decimal=',', index=False,
                  encoding='utf-8-sig')
    print(f'    -> error normal {fila["err_normal"]} | AUC {fila["auc"]} en la '
          f'época {fila["mejor_epoca"]}  ({fila["minutos"]} min)  [guardado]')


# =============================================================================
# 4. RESULTADOS
# =============================================================================
print('\n' + '=' * 72)
print('4. RESULTADOS DEL BARRIDO')
print('=' * 72)
hechas = hechas.sort_values('err_normal')
print(hechas[['configuracion', 'parametros', 'mejor_epoca', 'err_normal',
              'auc', 'recall_al_10pct', 'separacion', 'brecha_train_val',
              'minutos']].to_string(index=False))

base_fila = hechas[hechas.configuracion == 'base (la actual)']
if len(base_fila):
    b = base_fila.iloc[0]
    m = hechas.iloc[0]
    d_err = 100 * (m.err_normal / b.err_normal - 1)
    print()
    print(f'Configuración actual : error normal {b.err_normal:.6f} | AUC {b.auc:.4f}')
    print(f'Menor error          : error normal {m.err_normal:.6f} | AUC {m.auc:.4f}'
          f'  ({m.configuracion})')
    print(f'Cambio en el error   : {d_err:+.1f} %   |   cambio en AUC: '
          f'{m.auc - b.auc:+.4f}')
    print()
    if m.configuracion == b.configuracion or d_err > -5:
        print('  Ninguna configuración baja el error en normales más de un 5 %.')
        print('  Eso TAMBIÉN es un resultado: la configuración actual está en una')
        print('  zona plana, y el desempeño no depende de un ajuste afortunado de')
        print('  hiperparámetros. Es un argumento de robustez.')
    else:
        print('  Hay configuraciones que reconstruyen mejor. Con UNA semilla eso es')
        print('  una dirección, no una conclusión: el 04d las repite con 3 semillas.')
        print('  Mirá también brecha_train_val: si se abre, es sobreajuste.')
    print()
    print('  Correlación entre error en normales y AUC en el barrido:')
    from scipy.stats import spearmanr
    rho, p = spearmanr(hechas.err_normal, hechas.auc)
    print(f'    rho de Spearman = {rho:+.2f} (p = {p:.3f}, n = {len(hechas)})')
    print('    negativo = reconstruir mejor lo normal va con detectar mejor.')

# --- gráfico ---------------------------------------------------------------
fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
orden = hechas.sort_values('err_normal', ascending=False)
ejes[0].barh(orden.configuracion, orden.err_normal, color='#2a78d6')
if len(base_fila):
    ejes[0].axvline(b.err_normal, color='#d03b3b', ls='--', lw=1.5)
    ejes[0].text(b.err_normal, -0.6, ' actual', color='#d03b3b', fontsize=9)
ejes[0].set_xlabel('Error de reconstrucción en normales de validación (mediana)')
ejes[0].set_title('Qué configuración reconstruye mejor lo normal', loc='left',
                  fontweight='bold')
ejes[0].grid(alpha=0.3, axis='x')

ejes[1].scatter(hechas.err_normal, hechas.auc, s=70, color='#eb6834')
for _, r in hechas.iterrows():
    ejes[1].annotate(r.configuracion, (r.err_normal, r.auc),
                     fontsize=7.5, xytext=(4, 4), textcoords='offset points')
ejes[1].set_xlabel('Error en normales (menor es mejor)')
ejes[1].set_ylabel('AUC en validación')
ejes[1].set_title('¿Reconstruir mejor es detectar mejor?', loc='left',
                  fontweight='bold')
ejes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ARCHIVO_FIG, dpi=150)
plt.show()

print(f'\nGuardado: {ARCHIVO_SALIDA}')
print(f'Guardado: {ARCHIVO_FIG}')
print('\n' + '=' * 72)
print('LISTO. El modelo de producción NO cambia: el conjunto de prueba ya se')
print('miró, y elegir otra configuración ahora sería ajustarla a ese resultado.')
print('Esto se reporta como análisis de sensibilidad de hiperparámetros.')
print('=' * 72)
