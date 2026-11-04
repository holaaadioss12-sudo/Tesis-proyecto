# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 03b — AUTOENCODER TEMPORAL: ENTRENAMIENTO (SOLO VENTANAS NORMALES)
=============================================================================
"""

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:                  # fuera de Colab
    RUTA_BASE = __import__('os').environ.get('RUTA_BASE', '.')

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and _os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'     # CR010 en su subcarpeta, como en los demás pasos
_os.makedirs(RUTA_EQ, exist_ok=True)
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import sys
sys.path.append(RUTA_BASE)

import math
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
    print('\n⚠ NO HAY GPU. Entorno de ejecución -> Cambiar tipo de entorno -> GPU (T4).')

W = 60
STRIDE_TRAIN = 12    # ventanas cada 2 min -- traslape alto, más ejemplos
STRIDE_EVAL = W      # sin traslape en val/test
EPOCHS_MAX = int(_os.environ.get('EPOCAS_MAX', 100))
PACIENCIA = 15       # épocas sin mejorar la pérdida de validación

# Autoencoder temporal: latente de 60 x LATENT_DIM (reduce variables, conserva el tiempo).
# TIPO_AE = 'colapsado' vuelve al modelo anterior (latente de 64 sin eje de tiempo).
TIPO_AE = _os.environ.get('TIPO_AE', 'temporal')
LATENT_DIM = int(_os.environ.get('LATENT_DIM', 4))

ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
RUTA_MODELO = f'{RUTA_EQ}/autoencoder_W{W}_normales.pt'          # menor pérdida
RUTA_MODELO_SEP = f'{RUTA_EQ}/autoencoder_W{W}_normales_sep.pt'  # mayor separación
RUTA_CONFIG = f'{RUTA_EQ}/autoencoder_W{W}_config.json'
RUTA_ESTADO = f'{RUTA_EQ}/autoencoder_W{W}_estado_entrenamiento.pt'   # para retomar si se corta
REENTRENAR = _os.environ.get('REENTRENAR', '0') == '1'                 # 1: ignora el estado guardado
RUTA_NORMALIZADOR = f'{RUTA_EQ}/normalizador_W{W}.npz'
RUTA_CURVA = f'{RUTA_EQ}/03b_curva_entrenamiento.png'
RUTA_EPOCAS = f'{RUTA_EQ}/03b_comparacion_epocas.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

# Modelo anterior de CR010 (latente de 64, entrenado con todas las ventanas)
REF = ({'val_mse': 0.03057, 'test_mse': 0.03707,
        'err_normal': 0.0087, 'err_anomalo': 0.0302, 'razon': 3.46, 'auc': 0.8358}
       if EQUIPO == 'CR010' else None)


# =============================================================================
# 1. CARGA Y SEGMENTACIÓN
# =============================================================================
print('=' * 70)
print(f'PASO 03b — AUTOENCODER TEMPORAL, ENTRENAMIENTO (W = {W})')
print('=' * 70)


def cargar_limpio(ruta):
    cabecera = pd.read_csv(ruta, nrows=0)
    col_tiempo = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col_tiempo], low_memory=False)
    return df.set_index(col_tiempo).sort_index().rename_axis('timestamp')


df = cargar_limpio(ARCHIVO_LIMPIO)
VARIABLES = [v for v in VARIABLES if v in df.columns]
print(f'Dataset: {df.shape[0]:,} filas | {len(VARIABLES)} variables')


def segmentar_tramos_continuos(df, freq='10s', max_gap=1):
    idx = df.index.to_series()
    gaps = idx.diff() > pd.Timedelta(freq) * max_gap
    df = df.copy()
    df['tramo_id'] = gaps.cumsum()
    return [g.drop(columns='tramo_id') for _, g in df.groupby('tramo_id')]


tramos = segmentar_tramos_continuos(df)
tramos = [t for t in tramos if len(t) >= W]
print(f'Tramos útiles: {len(tramos):,}')


# =============================================================================
# 2. EVALUAR LAS REGLAS — para saber qué ventana es normal
# =============================================================================
print('\n' + '=' * 70)
print('2. EVALUANDO REGLAS PARA IDENTIFICAR LAS VENTANAS NORMALES')
print('=' * 70)

mascaras_activas = []      # una por tramo: bool por paso
ids_reglas = None

for i, t in enumerate(tramos):
    resultado = evaluar_reglas(t, verbose=False)
    if ids_reglas is None:
        ids_reglas = [c for c in resultado.columns
                      if c not in ('n_reglas_activadas', 'estado')]
    mascaras_activas.append(resultado[ids_reglas].to_numpy().any(axis=1))
    if (i + 1) % 500 == 0:
        print(f'  {i + 1:,} / {len(tramos):,} tramos')

print(f'Reglas evaluadas: {len(ids_reglas)}')
pasos_en_alarma = sum(int(m.sum()) for m in mascaras_activas)
pasos_totales = sum(len(m) for m in mascaras_activas)
print(f'Pasos con alguna regla activa: {pasos_en_alarma:,} de {pasos_totales:,} '
      f'({100 * pasos_en_alarma / pasos_totales:.1f} %)')


# =============================================================================
# 3. SPLIT CRONOLÓGICO
# =============================================================================
import json as _json
_corte = _json.load(open(f'{RUTA_EQ}/{EQUIPO}_corte_split.json'))
_T_TRAIN, _T_VAL = pd.Timestamp(_corte['T_train']), pd.Timestamp(_corte['T_val'])

_ini = np.array([t.index[0] for t in tramos])
s_tr = _ini <= _T_TRAIN
s_va = (_ini > _T_TRAIN) & (_ini <= _T_VAL)
s_te = _ini > _T_VAL
tramos_train = [t for t, m in zip(tramos, s_tr) if m]
tramos_val = [t for t, m in zip(tramos, s_va) if m]
tramos_test = [t for t, m in zip(tramos, s_te) if m]
masc_train = [m for m, k in zip(mascaras_activas, s_tr) if k]
masc_val = [m for m, k in zip(mascaras_activas, s_va) if k]
masc_test = [m for m, k in zip(mascaras_activas, s_te) if k]
print(f'\nTramos: train={len(tramos_train)} | val={len(tramos_val)} | '
      f'test={len(tramos_test)}')
print(f'Muestras: train={sum(len(t) for t in tramos_train):,} | '
      f'val={sum(len(t) for t in tramos_val):,} | '
      f'test={sum(len(t) for t in tramos_test):,}')
print(f'  corte unico: T_train = {_T_TRAIN} | T_val = {_T_VAL}')


# =============================================================================
# 4. NORMALIZACIÓN Z-SCORE — ajustada solo con train
# =============================================================================
def ajustar_normalizador(tramos_train, variables):
    concat = pd.concat(tramos_train)[variables]
    return {'media': concat.mean(), 'std': concat.std()}


def aplicar_normalizador(tramo, variables, params):
    tramo = tramo.copy()
    tramo[variables] = (tramo[variables] - params['media']) / params['std']
    return tramo


params_norm = ajustar_normalizador(tramos_train, VARIABLES)
tramos_train = [aplicar_normalizador(t, VARIABLES, params_norm) for t in tramos_train]
tramos_val = [aplicar_normalizador(t, VARIABLES, params_norm) for t in tramos_val]
tramos_test = [aplicar_normalizador(t, VARIABLES, params_norm) for t in tramos_test]

np.savez(RUTA_NORMALIZADOR,
         media=params_norm['media'].to_numpy(),
         std=params_norm['std'].to_numpy(),
         variables=np.array(VARIABLES, dtype=object))
print(f'Normalizador guardado: {RUTA_NORMALIZADOR}')


# =============================================================================
# 5. VENTANAS, CADA UNA CON SU ETIQUETA DE NORMALIDAD
# =============================================================================
def construir_ventanas(tramos, mascaras, variables, W, stride):
    ventanas, es_normal = [], []
    for t, m in zip(tramos, mascaras):
        arr = t[variables].to_numpy(dtype=np.float32)
        acum = np.concatenate(([0], np.cumsum(m.astype(np.int32))))
        for ini in range(0, len(arr) - W + 1, stride):
            ventanas.append(arr[ini:ini + W])
            es_normal.append((acum[ini + W] - acum[ini]) == 0)
    if not ventanas:
        return (np.empty((0, W, len(variables)), dtype=np.float32),
                np.empty(0, dtype=bool))
    return np.stack(ventanas), np.array(es_normal, dtype=bool)


X_train, norm_train = construir_ventanas(tramos_train, masc_train, VARIABLES, W, STRIDE_TRAIN)
X_val, norm_val = construir_ventanas(tramos_val, masc_val, VARIABLES, W, STRIDE_EVAL)
X_test, norm_test = construir_ventanas(tramos_test, masc_test, VARIABLES, W, STRIDE_EVAL)

print('\n' + '=' * 70)
print('5. VENTANAS')
print('=' * 70)
for nombre, X, nm in [('train', X_train, norm_train), ('val', X_val, norm_val),
                      ('test', X_test, norm_test)]:
    print(f'  {nombre:>5}: {len(X):>7,} ventanas | normales {int(nm.sum()):>7,} '
          f'({100 * nm.mean():.1f} %) | anómalas {int((~nm).sum()):>7,}')

# --- EL FILTRO: entrenar solo con las normales ---------------------------
X_train_normal = X_train[norm_train]
X_val_normal = X_val[norm_val]

print(f'\nVentanas de entrenamiento: {len(X_train):>7,} (todas)')
print(f'Se entrena con         : {len(X_train_normal):>7,} ventanas (solo normales)')
print(f'Validación para parar  : {len(X_val_normal):>7,} ventanas normales')

if len(X_train_normal) < 3000:
    print('\n⚠ Quedan pocas ventanas normales para entrenar. Con unos 136.000 parámetros')
    print('  hay riesgo de sobreajuste. Si pasa, bajar STRIDE_TRAIN a 6 para')
    print('  generar más ventanas, o reducir d_model de 64 a 32.')



class VentanasDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


train_loader = DataLoader(VentanasDataset(X_train_normal), batch_size=64, shuffle=True)
val_norm_loader = DataLoader(VentanasDataset(X_val_normal), batch_size=256, shuffle=False)
val_full_loader = DataLoader(VentanasDataset(X_val), batch_size=256, shuffle=False)
test_loader = DataLoader(VentanasDataset(X_test), batch_size=256, shuffle=False)


# =============================================================================
# 6. EL MODELO
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
        h = self.embed(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return h.mean(dim=1)

    def decode(self, latente):
        h = latente.unsqueeze(1).repeat(1, self.W, 1)
        h = self.pos_enc(h)
        h = self.decoder(h)
        return self.salida(h)

    def forward(self, x):
        return self.decode(self.encode(x))


class AutoencoderTransformerTemporal(nn.Module):
    """Latente (60, latent_dim): el cuello de botella reduce las variables en cada instante."""
    def __init__(self, n_vars, W, latent_dim=4, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1):
        super().__init__()
        self.W = W
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        capa_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(capa_encoder, num_layers=num_layers)
        self.a_latente = nn.Linear(d_model, latent_dim)       # cuello de botella: 64 -> k
        self.desde_latente = nn.Linear(latent_dim, d_model)   # vuelta: k -> 64
        capa_decoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(capa_decoder, num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def encode(self, x):                                       # (B, 60, 14)
        h = self.encoder(self.pos_enc(self.embed(x)))          # (B, 60, 64)
        return self.a_latente(h)                               # (B, 60, k)

    def decode(self, latente):
        h = self.pos_enc(self.desde_latente(latente))          # (B, 60, 64)
        return self.salida(self.decoder(h))                    # (B, 60, 14)

    def forward(self, x):
        return self.decode(self.encode(x))


n_vars = len(VARIABLES)
if TIPO_AE == 'temporal':
    modelo = AutoencoderTransformerTemporal(n_vars=n_vars, W=W, latent_dim=LATENT_DIM).to(device)
elif TIPO_AE == 'colapsado':
    modelo = AutoencoderTransformer(n_vars=n_vars, W=W).to(device)
else:
    raise SystemExit(f"TIPO_AE tiene que ser 'temporal' o 'colapsado', no {TIPO_AE!r}.")
n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
with torch.no_grad():
    _forma_latente = tuple(modelo.encode(torch.zeros(1, W, n_vars, device=device)).shape[1:])
_n_latente = int(np.prod(_forma_latente))
print(f'\nModelo: {TIPO_AE} | {n_params:,} parámetros entrenables')
print(f'Formas: entrada ({W}, {n_vars}) -> latente {_forma_latente} -> salida ({W}, {n_vars})')
print(f'Compresión: {W * n_vars} números -> {_n_latente} ({W * n_vars / _n_latente:.1f}x)')
print(f'Relación datos/parámetros: {len(X_train_normal) * W * n_vars / n_params:.0f} a 1')
_json.dump({'tipo': TIPO_AE, 'latent_dim': LATENT_DIM if TIPO_AE == 'temporal' else 64,
            'forma_latente': list(_forma_latente), 'd_model': 64, 'nhead': 4, 'num_layers': 2,
            'dim_ff': 128, 'dropout': 0.1, 'W': W, 'variables': VARIABLES},
           open(RUTA_CONFIG, 'w'), indent=1)
print(f'Configuración guardada: {RUTA_CONFIG}')


# =============================================================================
# 7. ENTRENAMIENTO
# =============================================================================
print('\n' + '=' * 70)
print('ENTRENANDO — solo con ventanas normales')
print('=' * 70)

optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
criterio = nn.MSELoss()
criterio_sin_reducir = nn.MSELoss(reduction='none')


def error_por_ventana(loader):
    """Error de reconstrucción de cada ventana del loader."""
    modelo.eval()
    errores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            e = criterio_sin_reducir(modelo(batch), batch).mean(dim=(1, 2))
            errores.append(e.cpu().numpy())
    return np.concatenate(errores)


mejor_val = float('inf')
mejor_separacion = -float('inf')
mejor_epoca = 0
mejor_epoca_sep = 0
espera = 0
historial = {'train': [], 'val_normal': [], 'err_norm': [], 'err_anom': [],
             'razon': [], 'auc': []}
epoca_inicial, terminado = 1, False

# si una corrida anterior se cortó, se retoma desde la última época guardada
_config_actual = {'tipo': TIPO_AE, 'latent_dim': LATENT_DIM}
if _os.path.exists(RUTA_ESTADO) and not REENTRENAR:
    _est = torch.load(RUTA_ESTADO, map_location=device, weights_only=False)
    if _est['config'] == _config_actual:
        modelo.load_state_dict(_est['modelo'])
        optimizador.load_state_dict(_est['optimizador'])
        mejor_val, mejor_separacion = _est['mejor_val'], _est['mejor_separacion']
        mejor_epoca, mejor_epoca_sep, espera = _est['mejor_epoca'], _est['mejor_epoca_sep'], _est['espera']
        historial, terminado = _est['historial'], _est['terminado']
        epoca_inicial = _est['epoca'] + 1
        torch.set_rng_state(_est['rng_cpu'].cpu())
        if torch.cuda.is_available() and _est.get('rng_cuda'):
            torch.cuda.set_rng_state_all([r.cpu() for r in _est['rng_cuda']])
        print(f'\nSe encontró un entrenamiento guardado: '
              + ('ya había terminado, no se repite.' if terminado or epoca_inicial > EPOCHS_MAX
                 else f'se retoma desde la época {epoca_inicial}.'))
    else:
        print(f'\nHay un entrenamiento guardado con otra configuración ({_est["config"]}): se empieza de cero.')

for epoch in range(epoca_inicial, EPOCHS_MAX + 1):
    if terminado:
        break
    modelo.train()
    perdida_train = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizador.zero_grad()
        loss = criterio(modelo(batch), batch)
        loss.backward()
        optimizador.step()
        perdida_train += loss.item() * batch.size(0)
    perdida_train /= len(train_loader.dataset)

    # error en las normales de validación -> criterio de parada
    modelo.eval()
    perdida_val = 0.0
    with torch.no_grad():
        for batch in val_norm_loader:
            batch = batch.to(device)
            perdida_val += criterio(modelo(batch), batch).item() * batch.size(0)
    perdida_val /= len(val_norm_loader.dataset)

    # diagnóstico: separación sobre TODA la validación
    err_val = error_por_ventana(val_full_loader)
    e_norm = float(np.median(err_val[norm_val]))
    e_anom = float(np.median(err_val[~norm_val]))
    razon = e_anom / e_norm if e_norm > 0 else float('nan')
    orden = np.argsort(err_val)
    rangos = np.empty(len(err_val), dtype=float)
    rangos[orden] = np.arange(1, len(err_val) + 1)
    n_pos, n_neg = int((~norm_val).sum()), int(norm_val.sum())
    auc = ((rangos[~norm_val].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
           if n_pos and n_neg else float('nan'))

    for k, v in [('train', perdida_train), ('val_normal', perdida_val),
                 ('err_norm', e_norm), ('err_anom', e_anom),
                 ('razon', razon), ('auc', auc)]:
        historial[k].append(v)

    print(f'Epoch {epoch:3d} | train={perdida_train:.5f} | val_norm={perdida_val:.5f} '
          f'| normal={e_norm:.5f} anómala={e_anom:.5f} | razón={razon:5.2f}x | AUC={auc:.4f}')

    # checkpoint 1: menor pérdida de reconstrucción sobre las normales
    if perdida_val < mejor_val - 1e-5:
        mejor_val = perdida_val
        espera = 0
        torch.save(modelo.state_dict(), RUTA_MODELO)
        mejor_epoca = epoch
    else:
        espera += 1

    if razon > mejor_separacion:
        mejor_separacion = razon
        torch.save(modelo.state_dict(), RUTA_MODELO_SEP)
        mejor_epoca_sep = epoch

    terminado = espera >= PACIENCIA
    torch.save({'config': _config_actual, 'epoca': epoch, 'terminado': terminado,
                'modelo': modelo.state_dict(), 'optimizador': optimizador.state_dict(),
                'mejor_val': mejor_val, 'mejor_separacion': mejor_separacion,
                'mejor_epoca': mejor_epoca, 'mejor_epoca_sep': mejor_epoca_sep, 'espera': espera,
                'historial': historial, 'rng_cpu': torch.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               RUTA_ESTADO)
    if terminado:
        print(f'\nEarly stopping en la época {epoch} (sin mejora en {PACIENCIA})')
        break

pd.DataFrame(historial).assign(epoca=range(1, len(historial['train']) + 1)).to_csv(
    RUTA_EPOCAS, sep=';', decimal=',', index=False, encoding='utf-8-sig')
print(f'\nCurva por época guardada: {RUTA_EPOCAS}')

# --- La comprobación que faltaba ------------------------------------------
_ep_final = len(historial['train'])
if not terminado:
    print('\n' + '!' * 70)
    print(f'ATENCIÓN: llegó al máximo de {EPOCHS_MAX} épocas sin activar la parada temprana.')
    print(f'Mejor época: {mejor_epoca}; {espera} de {PACIENCIA} épocas sin mejorar.')
    print('El entrenamiento no terminó por el criterio de validación. Para seguir,')
    print('corre de nuevo con EPOCAS_MAX más alto: retoma desde la época siguiente.')
    print('!' * 70)
else:
    print(f'\nConvergió: parada temprana en la época {_ep_final}, mínimo de pérdida '
          f'en la época {mejor_epoca} ({PACIENCIA} épocas sin mejorar).')


# =============================================================================
# 8. RESULTADOS Y COMPARACIÓN CON EL MODELO ANTERIOR
# =============================================================================
def evaluar(ruta):
    modelo.load_state_dict(torch.load(ruta, map_location=device))
    modelo.eval()
    ev = error_por_ventana(val_full_loader)
    et = error_por_ventana(test_loader)
    en = float(np.median(ev[norm_val]))
    ea = float(np.median(ev[~norm_val]))
    orden = np.argsort(ev)
    rangos = np.empty(len(ev), dtype=float)
    rangos[orden] = np.arange(1, len(ev) + 1)
    npos, nneg = int((~norm_val).sum()), int(norm_val.sum())
    auc = (rangos[~norm_val].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return {'err_norm': en, 'err_anom': ea, 'razon': ea / en,
            'auc': auc, 'mse_test': float(et.mean())}


m_perdida = evaluar(RUTA_MODELO)
m_separacion = evaluar(RUTA_MODELO_SEP)

print('\n' + '=' * 70)
print('RESULTADOS — TRES MODELOS')
print('=' * 70)
print(f'Checkpoint por menor pérdida  : época {mejor_epoca}')
print(f'Checkpoint por mayor separación: época {mejor_epoca_sep}')
print()
_ref = REF or {}
_col = (lambda v: f'{v:>12.4f}') if REF else (lambda v: '')
cab = f'{"":<34}' + (f'{"ANTERIOR":>12}' if REF else '') + f'{"menor pérdida":>16}{"mayor separación":>19}'
print(cab)
print('-' * len(cab))
for nombre, clave, ref in [
        ('Error mediano, normales', 'err_norm', _ref.get('err_normal')),
        ('Error mediano, anómalas', 'err_anom', _ref.get('err_anomalo')),
        ('Razón anómala / normal', 'razon', _ref.get('razon')),
        ('AUC normal vs. anómala', 'auc', _ref.get('auc'))]:
    print(f'{nombre:<34}{_col(ref)}{m_perdida[clave]:>16.4f}{m_separacion[clave]:>19.4f}')
print(f'{"MSE en test":<34}{_col(_ref.get("test_mse"))}'
      f'{m_perdida["mse_test"]:>16.4f}{m_separacion["mse_test"]:>19.4f}')

print()
mejor = max([('menor pérdida', m_perdida), ('mayor separación', m_separacion)],
            key=lambda x: x[1]['razon'])
print(f'El mejor detector es el checkpoint por {mejor[0]}: '
      f'razón {mejor[1]["razon"]:.2f}x, AUC {mejor[1]["auc"]:.4f}')
if REF and m_separacion['razon'] > REF['razon']:
    print(f'Frente al modelo entrenado con todo ({REF["razon"]:.2f}x), la mejora es de '
          f'{m_separacion["razon"] / REF["razon"]:.1f} veces.')

razon_v = m_separacion['razon']

# --- ¿hay efecto Ayala? ---------------------------------------------------
razones = np.array(historial['razon'])
pico = int(np.nanargmax(razones)) + 1
print()
print(f'Separación máxima durante el entrenamiento: {np.nanmax(razones):.2f}x '
      f'en la época {pico}')
if pico < mejor_epoca:
    print(f'  ⚠ El pico de separación ({pico}) llegó ANTES que el mínimo de pérdida '
          f'({mejor_epoca}).')
    print('    Es el efecto que describe Ayala: entrenar de más empeora la detección.')
    print('    Conviene reportar las dos curvas y discutir cuál checkpoint conviene.')
else:
    print('  El mínimo de pérdida y el máximo de separación coinciden o van juntos.')
    print('  No aparece el efecto que describe Ayala en este caso.')


# =============================================================================
# 9. CURVAS
# =============================================================================
fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

ejes[0].plot(historial['train'], label='train (solo normales)', linewidth=1.8)
ejes[0].plot(historial['val_normal'], label='validación (solo normales)', linewidth=1.8)
ejes[0].axvline(mejor_epoca - 1, color='gray', linestyle=':', label=f'mejor época ({mejor_epoca})')
ejes[0].set_xlabel('Época'); ejes[0].set_ylabel('MSE')
ejes[0].set_title('Pérdida de reconstrucción')
ejes[0].legend(); ejes[0].grid(alpha=.3)

ejes[1].plot(historial['razon'], color='#E8833A', linewidth=1.8, label='razón anómala/normal')
if REF:
    ejes[1].axhline(REF['razon'], color='gray', linestyle='--', linewidth=1,
                    label=f'modelo anterior ({REF["razon"]:.2f}x)')
ejes[1].axvline(pico - 1, color='#2E7D32', linestyle=':', label=f'pico ({pico})')
ejes[1].set_xlabel('Época'); ejes[1].set_ylabel('Razón')
ejes[1].set_title('Separación entre ventanas normales y anómalas')
ejes[1].legend(); ejes[1].grid(alpha=.3)

plt.tight_layout()
plt.savefig(RUTA_CURVA, dpi=150)
print(f'\nCurvas guardadas: {RUTA_CURVA}')

print('\n' + '=' * 70)
print('LISTO.')
print(f'      Checkpoint por menor pérdida  : {RUTA_MODELO}')
print(f'      Checkpoint por mayor separación: {RUTA_MODELO_SEP}')
print('Siguiente: 04_deteccion.py, que calcula el error de cada ventana y')
print('lo guarda en el archivo de errores por ventana.')
print('=' * 70)
