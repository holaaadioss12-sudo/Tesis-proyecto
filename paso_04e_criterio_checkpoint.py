# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04e — ¿CON QUÉ CRITERIO HAY QUE ELEGIR EL CHECKPOINT?
=============================================================================
El paso 04 guarda dos checkpoints de la misma corrida:

  autoencoder_W60_normales.pt       el de MÍNIMA pérdida de validación
  autoencoder_W60_normales_sep.pt   el de MÁXIMA separación   <- el que se usa

Esa elección se hizo cuando la separación era la única métrica disponible.
El 04d mostró dos cosas que la ponen en duda:

  - la separación no predice la detección: sobre nueve corridas,
    rho(separación, AUC) = -0,25 con p = 0,52, o sea ninguna relación;
  - el error de reconstrucción en ventanas normales SÍ la predice, y de
    forma perfecta: rho = -1,000. Cuanto mejor reconstruye lo normal,
    mejor detecta.

Si eso vale también DENTRO de una corrida, época a época, entonces el
checkpoint de mínima pérdida —que ya está guardado y no se usa— podría ser
mejor detector que el que está en producción. Este script lo mide.

QUÉ HACE
--------
Entrena la configuración actual con 3 semillas y 20 épocas, y en CADA época
calcula todas las métricas. Después compara tres criterios de selección:

  1. mínima pérdida de validación   (el checkpoint "normales")
  2. máxima separación              (el checkpoint "normales_sep", el actual)
  3. máximo AUC                     (el candidato)

CÓMO SE EVITA HACER TRAMPA
--------------------------
El criterio elige la época mirando VALIDACIÓN, y después la métrica se
reporta sobre PRUEBA, que no se tocó para elegir nada. Si se eligiera y se
midiera sobre el mismo conjunto, el criterio que más se mueve ganaría solo
por azar. Esta es también la forma correcta de reportarlo en la memoria.

Entradas: CR010_limpio.csv       (paso 01)
          reglas_engine.py
Salidas:  04e_curvas_por_epoca.csv   (3 semillas x 20 épocas = 60 filas)
          04e_comparacion_criterios.csv

NO toca el paso 04 ni sus checkpoints. No reemplaza nada: mide y reporta.
Tiempo estimado: ~6 minutos con GPU.
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

from reglas_engine import evaluar_reglas

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', device)

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
SEMILLAS = [42, 7, 2024]
ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
# PARCHE 1b -- archivos de salida NUEVOS (_v2). Estos scripts retoman desde
# su CSV si ya existe: con el nombre viejo se saltarían todas las corridas y
# devolverían los resultados del corte 70/15/15. Los viejos quedan como estaban.
SALIDA_CURVAS = f'{RUTA_EQ}/04e_curvas_por_epoca_v2.csv'
SALIDA_COMPARA = f'{RUTA_EQ}/04e_comparacion_criterios_v2.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
CFG = {'lr': 1e-3, 'num_layers': 2, 'd_model': 64, 'nhead': 4, 'dim_ff': 128}

# =============================================================================
# 1. DATOS — el corte por fecha del paso 04 (PARCHE 1), y ahora SÍ se usa prueba
# =============================================================================
print('\n' + '=' * 72)
print('1. PREPARANDO LOS DATOS')
print('=' * 72)

np.random.seed(42)


def cargar_limpio(ruta):
    cab = pd.read_csv(ruta, nrows=0)
    col = 'timestamp' if 'timestamp' in cab.columns else 'Time'
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
_s_te = _ini > _T_VAL
te_t = [t for t, k in zip(tramos, _s_te) if k]
te_m = [m for m, k in zip(mascaras, _s_te) if k]
print(f'Tramos: train={len(tr_t)} | val={len(va_t)} | test={len(te_t)}')

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
X_te, nm_te = construir(te_t, te_m, STRIDE_EVAL)
X_tr_n = X_tr[nm_tr]
X_va_n, X_va_a = X_va[nm_va], X_va[~nm_va]
X_te_n, X_te_a = X_te[nm_te], X_te[~nm_te]
print(f'Entrenamiento: {len(X_tr_n):,} ventanas normales')
print(f'Validación   : {len(X_va_n):,} normales, {len(X_va_a):,} anómalas   '
      f'<- acá se ELIGE la época')
print(f'Prueba       : {len(X_te_n):,} normales, {len(X_te_a):,} anómalas   '
      f'<- acá se MIDE el resultado')


class VD(Dataset):
    def __init__(self, X): self.X = torch.from_numpy(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


# =============================================================================
# 2. MODELO — idéntico al paso 04
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
    todos = np.concatenate([e_norm, e_anom])
    rangos = pd.Series(todos).rank(method='average').to_numpy()
    r_anom = rangos[len(e_norm):].sum()
    na, nn_ = len(e_anom), len(e_norm)
    return (r_anom - na * (na + 1) / 2) / (na * nn_)


def metricas(e_norm, e_anom, prefijo):
    umbral = np.percentile(e_norm, 90)
    return {
        f'{prefijo}_err_normal': float(np.median(e_norm)),
        f'{prefijo}_err_anomala': float(np.median(e_anom)),
        f'{prefijo}_separacion': float(np.median(e_anom) / np.median(e_norm)),
        f'{prefijo}_auc': float(auc_mann_whitney(e_norm, e_anom)),
        f'{prefijo}_recall': float((e_anom > umbral).mean()),
    }


# =============================================================================
# 3. ENTRENAR Y REGISTRAR CADA ÉPOCA
# =============================================================================
print('\n' + '=' * 72)
print(f'2. ENTRENANDO — {len(SEMILLAS)} semillas x hasta {EPOCAS_MAX} épocas '
      f'(early stopping, paciencia {PACIENCIA})')
print('=' * 72)

if os.path.exists(SALIDA_CURVAS):
    curvas = pd.read_csv(SALIDA_CURVAS, sep=';', decimal=',')
    ya = set(curvas.semilla.unique())
    print(f'Ya había {len(ya)} semillas hechas. Se saltan.')
else:
    curvas = pd.DataFrame()
    ya = set()

for semilla in SEMILLAS:
    if semilla in ya:
        print(f'\n[salto] semilla {semilla}')
        continue
    print(f'\n--- semilla {semilla} ---')
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    modelo = AutoencoderTransformer(
        n_vars=len(VARIABLES), W=W, d_model=CFG['d_model'], nhead=CFG['nhead'],
        num_layers=CFG['num_layers'], dim_ff=CFG['dim_ff']).to(device)
    opt = torch.optim.Adam(modelo.parameters(), lr=CFG['lr'])
    crit = nn.MSELoss()
    loader = DataLoader(VD(X_tr_n), batch_size=64, shuffle=True)

    t0 = time.time()
    filas = []
    mejor_val, espera = float('inf'), 0
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

        e_vn = errores(modelo, X_va_n)
        fila = {'semilla': semilla, 'epoca': ep, 'perdida_train': perdida,
                'val_perdida': float(e_vn.mean())}
        fila.update(metricas(e_vn, errores(modelo, X_va_a), 'val'))
        fila.update(metricas(errores(modelo, X_te_n), errores(modelo, X_te_a), 'test'))
        filas.append(fila)
        print(f'  época {ep:>2}  val: err_n={fila["val_err_normal"]:.5f} '
              f'sep={fila["val_separacion"]:5.2f} AUC={fila["val_auc"]:.4f}  |  '
              f'test: AUC={fila["test_auc"]:.4f} recall={fila["test_recall"]:.4f}')
        # PARCHE 2 (cont.) -- el mismo early stopping del paso 04
        if fila['val_perdida'] < mejor_val - 1e-5:
            mejor_val, espera = fila['val_perdida'], 0
        else:
            espera += 1
        if espera >= PACIENCIA:
            print(f'  early stopping en la época {ep} (sin mejora en {PACIENCIA})')
            break

    curvas = pd.concat([curvas, pd.DataFrame(filas)], ignore_index=True)
    curvas.to_csv(SALIDA_CURVAS, sep=';', decimal=',', index=False,
                  encoding='utf-8-sig')
    print(f'  ({(time.time()-t0)/60:.1f} min)  [guardado]')

# =============================================================================
# 4. LOS TRES CRITERIOS
# =============================================================================
print('\n' + '=' * 72)
print('3. ¿QUÉ ÉPOCA ELIGE CADA CRITERIO, Y QUÉ DA EN PRUEBA?')
print('=' * 72)

CRITERIOS = [
    # PARCHE 3 -- 'val_perdida' (media del error en normales) es EXACTAMENTE
    # la pérdida con que el paso 04 guarda el checkpoint "normales"; antes se
    # usaba la mediana, que es parecida pero no la misma.
    ('mínima pérdida de validación', 'val_perdida', 'min',
     'el checkpoint "normales" (guardado, no se usa)'),
    ('máxima separación', 'val_separacion', 'max',
     'el checkpoint "normales_sep" (EL QUE ESTÁ EN PRODUCCIÓN)'),
    ('máximo AUC', 'val_auc', 'max',
     'el candidato'),
]

filas = []
for nombre, col, modo, nota in CRITERIOS:
    print(f'\n--- {nombre} --- {nota}')
    for semilla, g in curvas.groupby('semilla'):
        i = g[col].idxmin() if modo == 'min' else g[col].idxmax()
        r = curvas.loc[i]
        filas.append({'criterio': nombre, 'semilla': semilla,
                      'epoca_elegida': int(r.epoca),
                      'test_auc': r.test_auc, 'test_recall': r.test_recall,
                      'test_separacion': r.test_separacion,
                      'test_err_normal': r.test_err_normal})
        print(f'    semilla {semilla:>4}: época {int(r.epoca):>2}  ->  '
              f'test AUC {r.test_auc:.5f}  recall {r.test_recall:.4f}  '
              f'separación {r.test_separacion:.2f}')

comp = pd.DataFrame(filas)
res = comp.groupby('criterio')[['epoca_elegida', 'test_auc', 'test_recall',
                                'test_separacion']].agg(['mean', 'std']).round(5)
print('\n' + '=' * 72)
print('RESUMEN SOBRE EL CONJUNTO DE PRUEBA (promedio de las 3 semillas)')
print('=' * 72)
print(res.to_string())

actual = 'máxima separación'
base_auc = comp[comp.criterio == actual].test_auc
base_rec = comp[comp.criterio == actual].test_recall
print(f'\n--- Cada criterio contra el que está en producción ---')
for nombre, _, _, _ in CRITERIOS:
    if nombre == actual:
        continue
    g = comp[comp.criterio == nombre]
    for met, b, etq in [('test_auc', base_auc, 'AUC'),
                        ('test_recall', base_rec, 'recall')]:
        dif = g[met].mean() - b.mean()
        disp = np.sqrt((g[met].std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        # PARCHE 5 -- con dispersión cero (los dos criterios eligen la misma
        # época en todas las semillas) la razón era NaN, y NaN > -1 es False:
        # el script etiquetaba 'PEOR' a una diferencia de exactamente cero.
        if disp > 0:
            veces = dif / disp
        else:
            veces = 0.0 if dif == 0 else np.sign(dif) * np.inf
        tag = ('MEJORA CLARA' if veces >= 2 else
               'tendencia' if veces >= 1 else
               'dentro del ruido' if veces > -1 else 'PEOR')
        print(f'  {nombre:30s} {etq:7s} {dif:+.5f}  '
              f'({veces:+.1f} veces la dispersión)  -> {tag}')

comp.to_csv(SALIDA_COMPARA, sep=';', decimal=',', index=False,
            encoding='utf-8-sig')
print(f'\nGuardado: {SALIDA_CURVAS}')
print(f'Guardado: {SALIDA_COMPARA}')

print('\n' + '=' * 72)
print('CÓMO LEERLO')
print('=' * 72)
print('Si "máximo AUC" o "mínima pérdida" supera a "máxima separación" por más')
print('de 2 veces la dispersión EN AUC Y EN RECALL sobre PRUEBA, conviene')
print('cambiar el criterio de selección del checkpoint en el paso 04.')
print('Si los tres quedan dentro del ruido, el criterio actual está bien y el')
print('tema queda cerrado: tampoco eso es un fracaso, es una pregunta contestada.')
# PARCHE 4 -- prueba ya se miró con el modelo de producción: este script la
# usa para MEDIR, no para elegir. El paso 06 ya comparó los dos checkpoints
# reales en prueba (AUC 0,836 contra 0,828), y esto lo generaliza a 3 semillas.
print('\nEl modelo de producción no cambia por este resultado: se reporta.')
