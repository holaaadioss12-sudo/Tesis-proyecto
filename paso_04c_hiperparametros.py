# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04c — BARRIDO DE HIPERPARÁMETROS
=============================================================================
Lo pidió el profesor en la reunión:

  "lo que autoencoder, como son redes neuronales, juega bien modificando la
   tasa de aprendizaje, es un valor que se puede cambiar, y la cantidad de
   capas también un poquito más, o la cantidad de nodos que tiene la red"

Entradas: CR010_limpio.csv       (paso 01)
          reglas_engine.py
Salidas:  04c_barrido_hiperparametros.csv   (se escribe DESPUÉS DE CADA CORRIDA)
          04c_barrido.png

NO toca el paso 04 ni sus checkpoints. Es un experimento aparte.

CÓMO ESTÁ DISEÑADO EL BARRIDO, Y POR QUÉ
----------------------------------------
No es una grilla completa. Es "una variable a la vez" (OFAT) alrededor de la
configuración actual. Razones:

  1. Una grilla de 3 x 3 x 3 son 27 corridas. Con el límite de GPU de Colab
     eso no se termina. OFAT son 7 corridas y responde la pregunta que importa:
     ¿mejora si muevo ESTO?
  2. Es lo que se puede explicar y defender en una memoria. "Partí de la
     configuración base y moví un parámetro a la vez" es una frase clara.
  3. Si alguna variable muestra una mejora grande, después se puede refinar
     solo en esa dirección.

La métrica para elegir NO es la pérdida de validación: es la SEPARACIÓN entre
el error de las ventanas anómalas y el de las normales. Es la misma lógica por
la que el paso 04 guarda dos checkpoints: el objetivo del sistema es detectar,
no reconstruir.

SOBRE "MÁS CAPAS / MÁS NODOS"
-----------------------------
Subir d_model de 64 a 128 multiplica los parámetros por algo más de 3. Con
~18.700 ventanas normales de entrenamiento, la relación datos/parámetros baja
de ~116 a 1 hasta ~30 a 1. Puede mejorar o puede sobreajustar; por eso el
script reporta también la brecha entre entrenamiento y validación, que es
donde el sobreajuste se ve. Si esa brecha se abre, más capacidad es peor
aunque la pérdida baje.

EL SCRIPT GUARDA DESPUÉS DE CADA CORRIDA
----------------------------------------
Si Colab se corta a mitad de camino, no se pierde lo hecho: basta volver a
correrlo y las configuraciones ya evaluadas se saltan solas.
=============================================================================
"""

from google.colab import drive
drive.mount('/content/drive')
RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'

# PARCHE EQUIPO -- el mismo pipeline para CR009, CR010 y CR011.
# El equipo se elige con la variable de entorno EQUIPO. En Colab, una celda
#     import os; os.environ['EQUIPO'] = 'CR009'
# antes de correr. Sin ella es CR010, y CR010 sigue leyendo y escribiendo en
# la carpeta raiz, igual que antes, para no mover nada de lo ya hecho. Los
# otros equipos trabajan en su propia subcarpeta: si no, sobrescribirian los
# modelos y resultados de CR010 (autoencoder_W60_*.pt, normalizador_W60.npz,
# B1_*, 06_*, 04c_* no llevan el nombre del equipo).
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
# PARCHE 2 -- se entrena COMO EL PASO 04 DE PRODUCCIÓN: hasta 100 épocas con
# early stopping (paciencia 15) sobre la pérdida en las ventanas normales de
# validación. Con un número fijo y corto de épocas el experimento medía qué
# configuración aprende más RÁPIDO, no cuál reconstruye MEJOR; el modelo de
# producción llegó a su mínimo recién en la época 93.
EPOCAS_MAX = 100
PACIENCIA = 15
ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
# PARCHE 1b -- archivos de salida NUEVOS (_v2). Estos scripts retoman desde
# su CSV si ya existe: con el nombre viejo se saltarían todas las corridas y
# devolverían los resultados del corte 70/15/15. Los viejos quedan como estaban.
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

# PARCHE 1 -- MISMAS VENTANAS NORMALES Y MISMO CORTE QUE EL PASO 04
# -----------------------------------------------------------------------------
# ANTES: las reglas se evaluaban sobre el dataframe entero y el corte era el
# 70/15/15 de la CANTIDAD de tramos. El paso 04 de producción hace otra cosa:
#   - evalúa las reglas TRAMO POR TRAMO: una regla con ventana móvil no debe
#     "ver" a través de un hueco de datos, y eso cambia qué ventana es normal;
#   - corta por FECHA con T_train y T_val de CR010_corte_split.json
#     (PARCHE 2 del pipeline), el mismo corte del normalizador y de la Fase B.
# Con el corte viejo este experimento entrenaba y validaba sobre períodos
# distintos a los del modelo que se defiende, y sus números no se podían
# comparar con él. Ahora las ventanas normales, el período de entrenamiento
# y el de validación son exactamente los del paso 04.
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
    """AUC sin sklearn: el estadístico U normalizado. Empates valen 0,5.
    Copia literal de la del 04d."""
    todos = np.concatenate([e_norm, e_anom])
    rangos = pd.Series(todos).rank(method='average').to_numpy()
    r_anom = rangos[len(e_norm):].sum()
    na, nn_ = len(e_anom), len(e_norm)
    return (r_anom - na * (na + 1) / 2) / (na * nn_)


# PARCHE 3 -- QUÉ SE MIDE Y EN QUÉ ÉPOCA
# -----------------------------------------------------------------------------
# ANTES: cada configuración se resumía en la época de máxima SEPARACIÓN
# (razón de medianas anómala/normal) y solo se reportaba eso. El 04d y el
# 04e mostraron que la separación no predice la detección (rho = -0,25,
# p = 0,52) y que el error en las ventanas normales sí (rho = -1,0). La
# pregunta del profesor, además, es si el ERROR DE RECONSTRUCCIÓN mejora.
# AHORA: cada configuración se resume en la época de MÍNIMA pérdida sobre
# las normales de validación (el mismo criterio que el early stopping y que
# el checkpoint autoencoder_W60_normales.pt), y en esa época se reporta:
#   err_normal     mediana del error en normales de validación (la misma
#                  cifra que el paso 05 da para val: 0,00639 en producción)
#   auc, recall    si reconstruir mejor se traduce en detectar mejor
#   separacion     la de antes, para poder comparar con el 04c viejo
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
# PARCHE 3 (cont.) -- se ordena por error en normales, que es la pregunta, y
# al lado va el AUC para ver si reconstruir mejor es también detectar mejor.
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
# PARCHE 4 -- el barrido se hizo DESPUÉS de mirar prueba, así que no puede
# cambiar el modelo que se defiende sin contaminar esa evaluación. Se reporta
# como análisis de sensibilidad.
print('LISTO. El modelo de producción NO cambia: el conjunto de prueba ya se')
print('miró, y elegir otra configuración ahora sería ajustarla a ese resultado.')
print('Esto se reporta como análisis de sensibilidad de hiperparámetros.')
print('=' * 72)
