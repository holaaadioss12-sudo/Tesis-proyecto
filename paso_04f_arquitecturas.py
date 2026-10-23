# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04f — COMPARACIÓN DE ARQUITECTURAS: ¿POR QUÉ UN TRANSFORMER?
=============================================================================
El objetivo específico 2 pide "evaluar y seleccionar algoritmos de Deep
Learning". Hasta aquí se entrenó una sola arquitectura (el autoencoder
Transformer) y se movieron sus hiperparámetros (04c, 04d, 04e). Este paso
compara arquitecturas distintas con EXACTAMENTE el mismo protocolo:

  transformer   el del paso 04: 2 + 2 capas, d_model 64, 4 cabezas
                (Vaswani et al., 2017). 135.758 parámetros.
  lstm          autoencoder LSTM: 2 capas que leen la ventana, el último
                estado se proyecta a 64 números y otras 2 capas reconstruyen
                desde ese resumen repetido en el tiempo (Malhotra et al.,
                2016). Es la referencia clásica para series de tiempo.
  cnn           autoencoder convolucional 1D: 3 convoluciones que reducen la
                ventana de 60 a 15 pasos, cuello de botella lineal de 64
                números y convoluciones transpuestas para volver. Es el tipo
                de autoencoder que usó la tesis anterior (Ayala, 2026).
  pca           análisis de componentes principales con 64 componentes sobre
                la ventana aplanada (60 x 14). Es un autoencoder LINEAL: si
                las redes no le ganan, la no linealidad no está aportando
                (Sakurada y Yairi, 2014). No se entrena con semillas.

Las tres redes tienen un presupuesto de parámetros parecido (~125 a 141 mil)
y el mismo cuello de botella (64 números), para que la comparación sea de
arquitectura y no de tamaño.

Protocolo (el mismo del paso 04 de producción y del 04d):
  - mismo corte por fecha (JSON del paso 01), mismas reglas tramo por tramo,
    misma estandarización (media y desviación del entrenamiento);
  - entrenamiento solo con ventanas normales (paso 12 entre ventanas);
  - Adam 0,001, lotes de 64, hasta 100 épocas, parada temprana con paciencia
    15 sobre la pérdida en las normales de validación, y se guarda el modelo
    de MÍNIMA pérdida (PARCHE P8);
  - 3 semillas (42, 7 y 2024) por red.

Qué se mide, para cada red y semilla:
  - AUC de detección en validación y en prueba (ventanas con alarma contra
    ventanas sin alarma, sin traslape);
  - anticipación con la definición del paso 06: entre las ventanas quietas de
    prueba, AUC del puntaje para "hay alarma en la ventana siguiente" y riesgo
    relativo a 10 minutos con el umbral P95 de las quietas de validación.

Regla de decisión: la SELECCIÓN se hace con el AUC de VALIDACIÓN, promediado
sobre semillas. Una red solo se considera mejor que el Transformer si cumple
dos condiciones: la diferencia supera 2 veces la dispersión entre semillas
(regla del 04d) y su intervalo bootstrap por tramos (1.000 remuestreos) no
incluye el 0. Con solo 3 semillas la dispersión puede salir muy chica, y la
segunda condición evita declarar ganadora a una red por las ventanas que
tocaron en validación. La prueba se informa para todas, con el mismo
intervalo, pero no decide nada.

Entradas: {EQUIPO}_limpio.csv, {EQUIPO}_corte_split.json   (paso 01)
          reglas_engine.py                                 (carpeta base)
Salidas:  04f_arquitecturas.csv      una fila por red y semilla (se escribe
                                     después de cada corrida; si Colab se
                                     corta, al volver a correr se retoma)
          04f_errores/*.npz          el error de cada ventana, por corrida
          04f_resumen.csv            promedio por red, diferencias e IC
          04f_niveles.csv            ¿el error de cada red reconoce el nivel
                                     leve / medio / grave? (sección 6; usa
                                     10_nivel_ventanas.csv del paso 10)
Si las salidas de una corrida anterior se movieron a una carpeta Cambios_...
del equipo, el script las trae de vuelta y retoma sin reentrenar.
Tiempo: unos 30 a 60 minutos con GPU para las 9 corridas.
=============================================================================
"""
import os
import sys
import json
import math
import time
import shutil

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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy.stats import spearmanr

from reglas_engine import evaluar_reglas

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
LATENTE = 64
EPOCAS_MAX = int(os.environ.get('EPOCAS_MAX', 100))   # solo para pruebas rápidas
PACIENCIA = 15
SEMILLAS = [int(s) for s in os.environ.get('SEMILLAS', '42,7,2024').split(',')]
REDES = ['transformer', 'lstm', 'cnn']
PERCENTIL_UMBRAL = 95
N_BOOT = 1000
BANDA = 'banda_N_empinada_natural'   # la escala principal del paso 10 (PARCHE P14)
NIVELES = ['1_leve', '2_media', '3_grave']
MIN_CLASE = 10
VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
ARCHIVO_SALIDA = f'{RUTA_EQ}/04f_arquitecturas.csv'
ARCHIVO_RESUMEN = f'{RUTA_EQ}/04f_resumen.csv'
CARPETA_ERRORES = f'{RUTA_EQ}/04f_errores'
for _f in [ARCHIVO_LIMPIO, ARCHIVO_CORTE]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste el paso 01?')


def buscar(nombre, carpeta=False):
    """La copia más reciente del archivo (o carpeta) en la carpeta del equipo
    o en sus subcarpetas (las salidas se van moviendo a carpetas Cambios_...).
    Nunca entra a la carpeta de otro equipo. None si no está."""
    hallados = []
    for raiz, carpetas, archivos in os.walk(RUTA_EQ):
        carpetas[:] = [c for c in carpetas if not c.upper().startswith('CR0')]
        if nombre in (carpetas if carpeta else archivos):
            hallados.append(os.path.join(raiz, nombre))
    return max(hallados, key=os.path.getmtime) if hallados else None


# Si la corrida anterior se movió a una carpeta Cambios_..., se trae de vuelta
# para retomar sin reentrenar.
if not os.path.exists(ARCHIVO_SALIDA):
    _prev = buscar('04f_arquitecturas.csv')
    if _prev:
        shutil.copy2(_prev, ARCHIVO_SALIDA)
        _err_prev = os.path.join(os.path.dirname(_prev), '04f_errores')
        if os.path.isdir(_err_prev):
            shutil.copytree(_err_prev, CARPETA_ERRORES, dirs_exist_ok=True)
        print(f'Corrida anterior encontrada en {os.path.dirname(_prev)}: se retoma desde ahí.')
os.makedirs(CARPETA_ERRORES, exist_ok=True)


# =============================================================================
# 1. DATOS — el mismo pipeline del paso 04 y del 04d
# =============================================================================
print('\n' + '=' * 72)
print(f'PASO 04f — COMPARACIÓN DE ARQUITECTURAS ({EQUIPO})')
print('=' * 72)


def cargar_limpio(ruta):
    cabecera = pd.read_csv(ruta, nrows=0)
    col = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col], low_memory=False)
    return df.set_index(col).sort_index().rename_axis('timestamp')


df = cargar_limpio(ARCHIVO_LIMPIO)
VARIABLES = [v for v in VARIABLES if v in df.columns]
print(f'Filas: {len(df):,}   Variables: {len(VARIABLES)}')

corte = json.load(open(ARCHIVO_CORTE))
T_TRAIN, T_VAL = pd.Timestamp(corte['T_train']), pd.Timestamp(corte['T_val'])
print(f'Corte único: T_train = {T_TRAIN} | T_val = {T_VAL}')

# Reglas evaluadas tramo por tramo, como el paso 04 (una regla con ventana
# móvil no debe "ver" a través de un hueco de datos).
idx = df.index.to_series()
df['_tramo'] = (idx.diff() > pd.Timedelta('10s')).cumsum().values
tramos, mascaras = [], []
for _, g in df.groupby('_tramo'):
    if len(g) >= W:
        t = g.drop(columns='_tramo')
        r = evaluar_reglas(t, verbose=False)
        ids = [c for c in r.columns if c not in ('n_reglas_activadas', 'estado')]
        tramos.append(t)
        mascaras.append(r[ids].to_numpy().any(axis=1))
print(f'Tramos útiles: {len(tramos):,}')

_ini = np.array([t.index[0] for t in tramos])
conjunto_tramo = np.where(_ini <= T_TRAIN, 'train', np.where(_ini <= T_VAL, 'val', 'test'))
concat = pd.concat([t for t, c in zip(tramos, conjunto_tramo) if c == 'train'])[VARIABLES]
media_n, std_n = concat.mean(), concat.std().replace(0, 1.0)


def construir(conjunto, stride):
    """Ventanas del conjunto, con su tramo, su posición, si tienen alarma y la
    hora en que empiezan (para cruzarlas con el nivel del paso 10)."""
    X, alarma, tramo, pos, inicio = [], [], [], [], []
    for k, (t, m, c) in enumerate(zip(tramos, mascaras, conjunto_tramo)):
        if c != conjunto:
            continue
        arr = ((t[VARIABLES] - media_n) / std_n).to_numpy(dtype=np.float32)
        for j, i in enumerate(range(0, len(arr) - W + 1, stride)):
            X.append(arr[i:i + W])
            alarma.append(bool(m[i:i + W].any()))
            tramo.append(k)
            pos.append(j)
            inicio.append(t.index[i])
    return (np.stack(X), np.array(alarma), np.array(tramo), np.array(pos),
            pd.DatetimeIndex(inicio))


X_tr, al_tr, _, _, _ = construir('train', STRIDE_TRAIN)
X_va, al_va, tr_va, po_va, ts_va = construir('val', STRIDE_EVAL)
X_te, al_te, tr_te, po_te, ts_te = construir('test', STRIDE_EVAL)
X_tr_n = X_tr[~al_tr]
X_va_n = X_va[~al_va]
print(f'Entrenamiento: {len(X_tr):,} ventanas, {len(X_tr_n):,} normales')
print(f'Validación   : {len(X_va):,} ventanas ({int(al_va.sum()):,} con alarma)')
print(f'Prueba       : {len(X_te):,} ventanas ({int(al_te.sum()):,} con alarma)')


def siguiente_con_alarma(alarma, tramo, pos):
    """1 si la ventana siguiente del mismo tramo tiene alarma, 0 si no, NaN si
    el tramo se acaba (la ventana queda censurada, como en el paso 06)."""
    s = pd.DataFrame({'a': alarma, 't': tramo, 'p': pos}).sort_values(['t', 'p'])
    sig = s.groupby('t').a.shift(-1)
    out = pd.Series(np.nan, index=s.index)
    out[sig.notna()] = sig[sig.notna()].astype(float)
    return out.sort_index().to_numpy()


fut_va = siguiente_con_alarma(al_va, tr_va, po_va)
fut_te = siguiente_con_alarma(al_te, tr_te, po_te)


class VD(Dataset):
    def __init__(self, X): self.X = torch.from_numpy(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


# =============================================================================
# 2. LAS ARQUITECTURAS
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


class AETransformer(nn.Module):
    """Idéntico al del paso 04."""
    def __init__(self, n_vars, W, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.W = W
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        capa = lambda: nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                                  dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(capa(), num_layers=num_layers)
        self.decoder = nn.TransformerEncoder(capa(), num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def forward(self, x):
        z = self.encoder(self.pos_enc(self.embed(x))).mean(dim=1)
        h = z.unsqueeze(1).repeat(1, self.W, 1)
        return self.salida(self.decoder(self.pos_enc(h)))


class AELSTM(nn.Module):
    """Codificador LSTM -> último estado -> 64 números -> decodificador LSTM."""
    def __init__(self, n_vars, W, oculto=64, capas=2, latente=LATENTE, dropout=0.1):
        super().__init__()
        self.W = W
        self.encoder = nn.LSTM(n_vars, oculto, capas, batch_first=True, dropout=dropout)
        self.a_latente = nn.Linear(oculto, latente)
        self.decoder = nn.LSTM(latente, oculto, capas, batch_first=True, dropout=dropout)
        self.salida = nn.Linear(oculto, n_vars)

    def forward(self, x):
        h, _ = self.encoder(x)
        z = self.a_latente(h[:, -1])
        d, _ = self.decoder(z.unsqueeze(1).repeat(1, self.W, 1))
        return self.salida(d)


class AECNN(nn.Module):
    """Convoluciones 1D: 60 -> 30 -> 15 pasos, cuello lineal de 64 números."""
    def __init__(self, n_vars, W, canales=48, latente=LATENTE, dropout=0.1):
        super().__init__()
        assert W % 4 == 0, 'W tiene que ser múltiplo de 4'
        self.c, self.l = canales, W // 4
        self.encoder = nn.Sequential(
            nn.Conv1d(n_vars, canales, 5, padding=2), nn.ReLU(),
            nn.Conv1d(canales, canales, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(canales, canales, 5, stride=2, padding=2), nn.ReLU(),
            nn.Dropout(dropout), nn.Flatten(), nn.Linear(canales * self.l, latente))
        self.desde_latente = nn.Sequential(nn.Linear(latente, canales * self.l), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(canales, canales, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(canales, canales, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(canales, n_vars, 5, padding=2))

    def forward(self, x):
        z = self.encoder(x.transpose(1, 2))
        h = self.desde_latente(z).view(-1, self.c, self.l)
        return self.decoder(h).transpose(1, 2)


def crear(red):
    n = len(VARIABLES)
    return {'transformer': AETransformer, 'lstm': AELSTM, 'cnn': AECNN}[red](n, W).to(device)


for _r in REDES:
    print(f'  {_r:<12} {sum(p.numel() for p in crear(_r).parameters()):>8,} parámetros')


# =============================================================================
# 3. MÉTRICAS
# =============================================================================
def errores(modelo, X, batch=256):
    modelo.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.from_numpy(X[i:i + batch]).to(device)
            out.append(((modelo(b) - b) ** 2).mean(dim=(1, 2)).cpu().numpy())
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
                 'err_normal_val': float(np.median(e_va[~al_va]))},
                **anticipacion(e_va, e_te))


# =============================================================================
# 4. ENTRENAMIENTO — el protocolo del paso 04
# =============================================================================
def entrenar(red, semilla):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    modelo = crear(red)
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
        perdida_val = float(errores(modelo, X_va_n).mean())
        if perdida_val < mejor_perdida - 1e-5:
            mejor_perdida, mejor_epoca, espera = perdida_val, ep, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            espera += 1
        if ep % 10 == 0 or ep == 1:
            print(f'    época {ep:>3}  pérdida normales val = {perdida_val:.5f}')
        if espera >= PACIENCIA:
            print(f'    parada temprana en la época {ep} (mínimo en la {mejor_epoca})')
            break
    modelo.load_state_dict(mejor_estado)
    e_va, e_te = errores(modelo, X_va), errores(modelo, X_te)
    return e_va, e_te, {'red': red, 'semilla': semilla,
                        'parametros': sum(p.numel() for p in modelo.parameters()),
                        'epocas_corridas': ep, 'mejor_epoca': mejor_epoca,
                        'perdida_val': round(mejor_perdida, 6),
                        'minutos': round((time.time() - t0) / 60, 2)}


if os.path.exists(ARCHIVO_SALIDA):
    hechas = pd.read_csv(ARCHIVO_SALIDA, sep=';', decimal=',')
    _otro = hechas.T_train.astype(str) != str(T_TRAIN)
    if _otro.any():
        print(f'{int(_otro.sum())} filas guardadas con otro corte: se vuelven a correr.')
        hechas = hechas[~_otro].reset_index(drop=True)
else:
    hechas = pd.DataFrame()
ya = set(zip(hechas.get('red', []), hechas.get('semilla', [])))

print('\n' + '=' * 72)
print(f'4. {len(REDES)} redes x {len(SEMILLAS)} semillas + PCA')
print('=' * 72)
CORRIDAS = [(r, s) for r in REDES for s in SEMILLAS] + [('pca', 0)]
for red, semilla in CORRIDAS:
    ruta_err = f'{CARPETA_ERRORES}/{red}_{semilla}.npz'
    if (red, semilla) in ya and os.path.exists(ruta_err):
        print(f'[salto] {red} · semilla {semilla}')
        continue
    print(f'\n--- {red} · semilla {semilla} ---')
    if red == 'pca':
        t0 = time.time()
        pca = PCA(n_components=LATENTE, random_state=42).fit(X_tr_n.reshape(len(X_tr_n), -1))

        def err_pca(X):
            Xf = X.reshape(len(X), -1)
            return ((pca.inverse_transform(pca.transform(Xf)) - Xf) ** 2).mean(axis=1)
        e_va, e_te = err_pca(X_va), err_pca(X_te)
        info = {'red': 'pca', 'semilla': 0, 'parametros': int(pca.components_.size + pca.mean_.size),
                'epocas_corridas': 0, 'mejor_epoca': 0,
                'perdida_val': round(float(e_va[~al_va].mean()), 6),
                'minutos': round((time.time() - t0) / 60, 2)}
    else:
        e_va, e_te, info = entrenar(red, semilla)
    np.savez_compressed(ruta_err, e_va=e_va.astype(np.float32), e_te=e_te.astype(np.float32))
    fila = dict(info, **{k: round(float(v), 5) for k, v in fila_metricas(e_va, e_te).items()},
                T_train=str(T_TRAIN))
    hechas = pd.concat([hechas, pd.DataFrame([fila])], ignore_index=True) if len(hechas) else pd.DataFrame([fila])
    hechas = hechas.drop_duplicates(['red', 'semilla'], keep='last')
    hechas.to_csv(ARCHIVO_SALIDA, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'    AUC val {fila["auc_val"]}  AUC prueba {fila["auc_test"]}  '
          f'RR 10 min {fila["rr_10min"]}  ({fila["minutos"]} min)  [guardado]')


# =============================================================================
# 5. RESUMEN: SELECCIÓN EN VALIDACIÓN, PRUEBA CON INTERVALO
# =============================================================================
print('\n' + '=' * 72)
print('5. RESUMEN')
print('=' * 72)
REF = 'transformer'
METRICAS = ['auc_val', 'auc_test', 'auc_anticipacion', 'rr_10min', 'err_normal_val']
g = hechas.groupby('red')
resumen = g[METRICAS].mean().round(4)
resumen['auc_val_sd'] = g.auc_val.std(ddof=1).round(4)
resumen['parametros'] = g.parametros.first()
resumen['corridas'] = g.size()
resumen = resumen.reindex([r for r in REDES + ['pca'] if r in resumen.index])


def bootstrap_contra_ref(clave, alarma, tramo, filtro=None):
    """Intervalo del 95 % de (AUC de la red - AUC del Transformer), pareado:
    se remuestrean tramos completos y las dos redes se evalúan sobre las
    mismas ventanas. El AUC de cada red es el promedio de sus semillas.
    filtro: ventanas que entran (para la anticipación, las quietas de prueba
    que tienen ventana siguiente en el mismo tramo)."""
    if filtro is None:
        filtro = np.ones(len(alarma), dtype=bool)
    alarma, tramo = np.asarray(alarma)[filtro].astype(bool), np.asarray(tramo)[filtro]
    errs = {red: [np.load(f'{CARPETA_ERRORES}/{red}_{int(s)}.npz')[clave][filtro] for s in sub.semilla]
            for red, sub in hechas.groupby('red')}
    if REF not in errs:
        return {}
    grupos = pd.Series(np.arange(len(tramo))).groupby(tramo).indices
    claves = np.array(list(grupos))
    rng = np.random.default_rng(42)
    boot = {r: [] for r in errs}
    for _ in range(N_BOOT):
        sel = np.concatenate([grupos[c] for c in rng.choice(claves, len(claves))])
        for r, lista in errs.items():
            boot[r].append(np.nanmean([auc(e[sel], alarma[sel]) for e in lista]))
    return {r: np.nanpercentile(np.array(boot[r]) - np.array(boot[REF]), [2.5, 97.5])
            for r in errs if r != REF}


ic_val = bootstrap_contra_ref('e_va', al_va, tr_va)
ic_te = bootstrap_contra_ref('e_te', al_te, tr_te)
_q = (~al_te) & ~np.isnan(fut_te)
ic_an = bootstrap_contra_ref('e_te', np.nan_to_num(fut_te), tr_te, filtro=_q)
for r in resumen.index:
    if r == REF or r not in ic_val:
        continue
    for tag, ic, col in [('val', ic_val, 'auc_val'), ('test', ic_te, 'auc_test'),
                         ('anticipacion', ic_an, 'auc_anticipacion')]:
        resumen.loc[r, f'dif_{tag}'] = round(float(resumen.loc[r, col] - resumen.loc[REF, col]), 4)
        resumen.loc[r, f'ic95_{tag}_inf'] = round(float(ic[r][0]), 4)
        resumen.loc[r, f'ic95_{tag}_sup'] = round(float(ic[r][1]), 4)

pd.set_option('display.width', 220)
print(resumen.to_string())

# Regla de decisión, solo con VALIDACIÓN. Una red le gana al Transformer si
# cumple las dos condiciones: (1) la diferencia supera 2 veces la dispersión
# entre semillas (regla del 04d: no es suerte de la semilla) y (2) el
# intervalo bootstrap por tramos no incluye el 0 (no es suerte de las
# ventanas que tocaron en validación).
print('\n--- Selección en VALIDACIÓN ---')
if REF in resumen.index:
    b = hechas[hechas.red == REF].auc_val
    for r in resumen.index:
        if r == REF or r not in ic_val:
            continue
        a = hechas[hechas.red == r].auc_val
        dif = a.mean() - b.mean()
        sd = np.sqrt((np.nan_to_num(a.std(ddof=1)) ** 2 + np.nan_to_num(b.std(ddof=1)) ** 2) / 2)
        veces = dif / sd if sd > 0 else np.nan
        inf, sup = ic_val[r]
        fuera_del_cero = inf > 0 or sup < 0
        if np.isnan(veces):
            juicio = 'faltan semillas para medir la dispersión'
        elif abs(veces) >= 2 and fuera_del_cero:
            juicio = 'DIFERENCIA CLARA (' + ('mejor' if dif > 0 else 'peor') + ' que el Transformer)'
        elif abs(veces) >= 1 or fuera_del_cero:
            juicio = 'tendencia, no concluyente'
        else:
            juicio = 'empate'
        resumen.loc[r, 'juicio_validacion'] = juicio
        print(f'  {r:<12} AUC val {a.mean():.4f} contra {b.mean():.4f}: {dif:+.4f} '
              f'({veces:+.1f} veces la dispersión; IC95 [{inf:+.4f}, {sup:+.4f}]) -> {juicio}')
resumen.to_csv(ARCHIVO_RESUMEN, sep=';', decimal=',', encoding='utf-8-sig')

print('\nSi ninguna red tiene DIFERENCIA CLARA a favor, se mantiene el Transformer.')
print('La prueba y la anticipación (dif_test, dif_anticipacion y sus IC) solo se informan: no cambian la selección.')
print(f'\nGuardado: {ARCHIVO_SALIDA}, {ARCHIVO_RESUMEN} y la carpeta {CARPETA_ERRORES}')


# =============================================================================
# 6. ¿EL ERROR DE CADA RED RECONOCE EL NIVEL (LEVE, MEDIO, GRAVE)?
# =============================================================================
# Ninguna de estas redes entrega un nivel: todas se entrenan solo con ventanas
# normales y dan UN número, el error de reconstrucción. Lo que se puede medir
# es si ese número ORDENA los niveles del paso 05 (N con pesos 9/3/1 y cortes
# de K-means, archivo 10_nivel_ventanas.csv del paso 10) y cuánto acierta el
# nivel si se le ponen cortes:
#   veces_*      error mediano de cada nivel / el de las ventanas con N = 0
#                (validación; es la tabla del paso 10 para cada red)
#   spearman     correlación de rangos entre el error y el nivel 0-1-2-3
#   auc_leve_media, auc_media_grave
#                ¿el error separa niveles VECINOS? (0,5 = no los distingue)
#   f1_macro     regresión logística sobre log(error), entrenada con las
#                ventanas de validación con N > 0 y medida en las de prueba;
#                se compara con un azar estratificado (el mismo esquema que
#                el XGBoost del paso 10, pero con un solo número)
# Todo es descriptivo: no cambia la selección de la sección 5.
# =============================================================================
print('\n' + '=' * 72)
print('6. ¿EL ERROR DE CADA RED RECONOCE EL NIVEL?')
print('=' * 72)
ruta_nivel = buscar('10_nivel_ventanas.csv')
if ruta_nivel is None:
    print('No encontré 10_nivel_ventanas.csv (paso 10): se salta esta sección.')
else:
    print(f'Nivel de cada ventana: {ruta_nivel}')
    _nv = pd.read_csv(ruta_nivel, usecols=['timestamp_inicio', BANDA], parse_dates=['timestamp_inicio'])
    _mapa = _nv.drop_duplicates('timestamp_inicio').set_index('timestamp_inicio')[BANDA]
    b_va = ts_va.map(_mapa).to_numpy(dtype=object)
    b_te = ts_te.map(_mapa).to_numpy(dtype=object)
    for _n, _b, _a in [('validación', b_va, al_va), ('prueba', b_te, al_te)]:
        _ok = pd.notna(_b)
        _coinc = ((_b[_ok] != '0_normal') == _a[_ok]).mean()
        print(f'  {_n}: {_ok.mean():.1%} de las ventanas con nivel; la alarma de este paso '
              f'coincide con N > 0 en el {_coinc:.1%}')
    if pd.notna(b_va).mean() < 0.9:
        print('  OJO: menos del 90 % de las ventanas cruzó con el paso 10. Revisa que el')
        print('  10_nivel_ventanas.csv sea del mismo limpio y del mismo corte.')
    COD = {'0_normal': 0, '1_leve': 1, '2_media': 2, '3_grave': 3}
    m_va3, m_te3 = np.isin(b_va, NIVELES), np.isin(b_te, NIVELES)
    _n_va = pd.Series(b_va[m_va3]).value_counts().reindex(NIVELES).fillna(0)
    _n_te = pd.Series(b_te[m_te3]).value_counts().reindex(NIVELES).fillna(0)
    print(f'  ventanas por nivel, validación: { {k: int(v) for k, v in _n_va.items()} }')
    print(f'  ventanas por nivel, prueba    : { {k: int(v) for k, v in _n_te.items()} }')
    hay_f1 = (_n_va >= MIN_CLASE).all() and (_n_te >= MIN_CLASE).all()
    if not hay_f1:
        print(f'  Algún nivel tiene menos de {MIN_CLASE} ventanas: no se calcula el F1.')

    def metricas_nivel(e_va, e_te):
        f = {}
        ok = pd.notna(b_va)
        med = pd.Series(e_va[ok]).groupby(b_va[ok]).median()
        for k in NIVELES:
            f[f'veces_{k}'] = med.get(k, np.nan) / med.get('0_normal', np.nan)
        f['spearman'] = spearmanr(e_va[ok], pd.Series(b_va[ok]).map(COD)).correlation
        for a, b in [('1_leve', '2_media'), ('2_media', '3_grave')]:
            m = np.isin(b_va, [a, b])
            f[f'auc_{a[2:]}_{b[2:]}'] = auc(e_va[m], b_va[m] == b)
        if hay_f1:
            lr = LogisticRegression(class_weight='balanced', max_iter=1000)
            lr.fit(np.log(e_va[m_va3] + 1e-12).reshape(-1, 1), b_va[m_va3].astype(str))
            p = lr.predict(np.log(e_te[m_te3] + 1e-12).reshape(-1, 1))
            f['f1_macro'] = f1_score(b_te[m_te3].astype(str), p, average='macro')
            f['bal_acc'] = balanced_accuracy_score(b_te[m_te3].astype(str), p)
        return f

    filas_nv = []
    for _, fila in hechas.iterrows():
        z = np.load(f'{CARPETA_ERRORES}/{fila.red}_{int(fila.semilla)}.npz')
        filas_nv.append(dict(red=fila.red, semilla=int(fila.semilla), **metricas_nivel(z['e_va'], z['e_te'])))
    niveles = (pd.DataFrame(filas_nv).groupby('red').mean(numeric_only=True)
               .drop(columns='semilla').reindex([r for r in REDES + ['pca'] if r in set(hechas.red)]))
    if hay_f1:
        _y_tr, _y_te = b_va[m_va3].astype(str), b_te[m_te3].astype(str)
        _dc = DummyClassifier(strategy='stratified', random_state=42).fit(np.zeros((len(_y_tr), 1)), _y_tr)
        _p = _dc.predict(np.zeros((len(_y_te), 1)))
        niveles.loc['azar_estratificado', ['f1_macro', 'bal_acc']] = [
            f1_score(_y_te, _p, average='macro'), balanced_accuracy_score(_y_te, _p)]
    niveles = niveles.round(3)
    print('\nPromedio de las semillas (veces, Spearman y AUC en validación; F1 en prueba):')
    print(niveles.to_string())
    _xgb = buscar('10_xgb.csv')
    if _xgb:
        print('\nComo referencia, el XGBoost del paso 10 (42 variables de la señal, no un solo número):')
        print(pd.read_csv(_xgb).to_string(index=False))
    niveles.to_csv(f'{RUTA_EQ}/04f_niveles.csv', sep=';', decimal=',', encoding='utf-8-sig')
    print('\nCómo leerlo: si las veces suben de nivel en nivel, el error ORDENA la gravedad.')
    print('Si auc_leve_media o auc_media_grave están cerca de 0,5, ese par de niveles')
    print('vecinos no se distingue con el error. Si el F1 macro apenas supera al azar,')
    print('el error no basta para decir si una ventana es leve, media o grave.')
    print(f'Guardado: {RUTA_EQ}/04f_niveles.csv')
