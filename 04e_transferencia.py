# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04e — TRANSFERENCIA: ¿LOS TRES CHANCADORES SE COMPORTAN IGUAL?
=============================================================================
Cada autoencoder, entrenado con las ventanas normales de su equipo, se aplica a
las ventanas de validación y prueba de los otros dos. Si el modelo de otro equipo
reconstruye las normales casi tan bien como el propio y separa igual las ventanas
con alarma, los equipos se comportan igual; si no, conviene un modelo por equipo.
Dos variantes: normalizador del equipo evaluado (se transfiere la forma de la
dinámica) y normalizador del modelo (se transfiere todo). El umbral P95 se
recalibra con las normales de validación del equipo evaluado.
Entrada:  por equipo, {EQ}_limpio.csv y {EQ}_corte_split.json (paso 01), normalizador_W60.npz,
          autoencoder_W60_normales[_sep].pt y su config (paso 03b), {EQ}_error_reconstruccion_W60.csv (paso 04)
Salida:   04e_transferencia.csv, 04e_transferencia_variables.csv y 04e_transferencia.png en la carpeta base
Es descriptivo: no se elige nada; se informan validación y prueba.
"""
import os
import json
import math

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score


# =============================================================================
# 0. CONSTANTES
# =============================================================================
EQUIPOS = os.environ.get('EQUIPOS', 'CR009,CR010,CR011').split(',')
W = 60
PERCENTIL = 95
LLAVES = ['tramo_id', 'ventana_id']
VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
PROPIO, DEL_EQUIPO, DEL_MODELO = 'propio', 'del equipo evaluado', 'del modelo'
AZULES = LinearSegmentedColormap.from_list('azules', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#104281'])

PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
if PUNTO_PARADA not in ('perdida', 'separacion'):
    raise SystemExit(f"PUNTO_PARADA tiene que ser 'perdida' o 'separacion', no {PUNTO_PARADA!r}.")
NOMBRE_PT = f'autoencoder_W{W}_normales_sep.pt' if PUNTO_PARADA == 'separacion' else f'autoencoder_W{W}_normales.pt'
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ruta_eq(eq):
    if eq == 'CR010':
        return f'{RUTA_BASE}/CR010' if os.path.isdir(f'{RUTA_BASE}/CR010') else RUTA_BASE
    return f'{RUTA_BASE}/{eq}'


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


# =============================================================================
# 1. MODELOS (los mismos del paso 04)
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
    def __init__(self, n_vars, W, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.W = W
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True), num_layers)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True), num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def forward(self, x):
        h = self.encoder(self.pos_enc(self.embed(x))).mean(dim=1).unsqueeze(1).repeat(1, self.W, 1)
        return self.salida(self.decoder(self.pos_enc(h)))


class AutoencoderTransformerTemporal(nn.Module):
    def __init__(self, n_vars, W, latent_dim=4, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.W = W
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True), num_layers)
        self.a_latente = nn.Linear(d_model, latent_dim)
        self.desde_latente = nn.Linear(latent_dim, d_model)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True), num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def forward(self, x):
        z = self.a_latente(self.encoder(self.pos_enc(self.embed(x))))
        return self.salida(self.decoder(self.pos_enc(self.desde_latente(z))))


def cargar_modelo(eq):
    r = ruta_eq(eq)
    ruta = next((p for p in [f'{r}/{NOMBRE_PT}', f'{r}/CAMBIOS/{NOMBRE_PT}'] if os.path.exists(p)), None)
    if ruta is None:
        raise SystemExit(f'Falta el modelo de {eq} ({NOMBRE_PT}): corre el 03b.')
    _c = os.path.join(os.path.dirname(ruta), f'autoencoder_W{W}_config.json')
    cfg = json.load(open(_c)) if os.path.exists(_c) else {'tipo': 'colapsado', 'latent_dim': 64}
    arq = {k: cfg[k] for k in ('d_model', 'nhead', 'num_layers', 'dim_ff', 'dropout') if k in cfg}
    if cfg['tipo'] == 'temporal':
        mod = AutoencoderTransformerTemporal(len(VARIABLES), W, latent_dim=cfg['latent_dim'], **arq)
    else:
        mod = AutoencoderTransformer(len(VARIABLES), W, **arq)
    mod.load_state_dict(torch.load(ruta, map_location=device))
    return mod.to(device).eval(), cfg


def errores(mod, X):
    """Error de cada variable por ventana (promedio en el tiempo)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            b = torch.from_numpy(X[i:i + 256]).to(device)
            out.append(((mod(b) - b) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


# =============================================================================
# 2. VENTANAS DE VALIDACIÓN Y PRUEBA DE CADA EQUIPO (como el paso 04)
# =============================================================================
def cargar_limpio(ruta):
    cab = pd.read_csv(ruta, nrows=0)
    col = 'timestamp' if 'timestamp' in cab.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col], low_memory=False)
    return df.set_index(col).sort_index().rename_axis('timestamp')


def ventanas_de(eq):
    r = ruta_eq(eq)
    df = cargar_limpio(f'{r}/{eq}_limpio.csv')
    faltan = [v for v in VARIABLES if v not in df.columns]
    if faltan:
        raise SystemExit(f'{eq}: faltan variables en el limpio: {faltan}')
    corte = json.load(open(f'{r}/{eq}_corte_split.json'))
    t_train, t_val = pd.Timestamp(corte['T_train']), pd.Timestamp(corte['T_val'])
    salto = df.index.to_series().diff().dt.total_seconds() > 10
    tramos = [g for _, g in df.groupby(salto.cumsum().to_numpy())]
    tramos = [t for t in tramos if len(t) >= W]
    X, ids = [], []
    for tid, t in enumerate(tramos):
        ini = t.index[0]
        cj = 'val' if t_train < ini <= t_val else ('test' if ini > t_val else None)
        if cj is None:
            continue
        arr = t[VARIABLES].to_numpy(dtype=np.float32)
        for vid, s in enumerate(range(0, len(arr) - W + 1, W)):
            X.append(arr[s:s + W])
            ids.append((tid, vid, cj))
    info = pd.DataFrame(ids, columns=LLAVES + ['conjunto'])
    err = pd.read_csv(f'{r}/{eq}_error_reconstruccion_W{W}.csv',
                      usecols=LLAVES + ['conjunto', 'n_reglas_activas', 'error_reconstruccion'])
    info = info.merge(err.rename(columns={'conjunto': 'conjunto_04'}), on=LLAVES, how='left', validate='one_to_one')
    if info.error_reconstruccion.isna().any() or (info.conjunto != info.conjunto_04).any():
        raise SystemExit(f'{eq}: las ventanas no coinciden con las del paso 04. ¿Mismo limpio y mismo corte?')
    info['alarma'] = (info.n_reglas_activas > 0).astype(int)
    norm = np.load(f'{r}/normalizador_W{W}.npz', allow_pickle=True)
    _v = list(norm['variables'])
    mu = pd.Series(norm['media'], index=_v)[VARIABLES].to_numpy(dtype=np.float32)
    sd = pd.Series(norm['std'], index=_v)[VARIABLES].to_numpy(dtype=np.float32)
    return np.stack(X), info, (mu, sd)


print('=' * 72)
print('PASO 04e — TRANSFERENCIA ENTRE EQUIPOS')
print('=' * 72)
print(f'Dispositivo: {device} | punto de parada: {PUNTO_PARADA}')
DATOS, NORM, MOD = {}, {}, {}
for eq in EQUIPOS:
    X, info, NORM[eq] = ventanas_de(eq)
    DATOS[eq] = (X, info)
    MOD[eq], cfg = cargar_modelo(eq)
    print(f'{eq}: validación {(info.conjunto == "val").sum():,} | prueba {(info.conjunto == "test").sum():,} '
          f'ventanas | modelo {cfg["tipo"]}, k = {cfg["latent_dim"]}')


# =============================================================================
# 3. CADA MODELO SOBRE CADA EQUIPO
# =============================================================================
filas, filas_v = [], []
for dst in EQUIPOS:
    X, info = DATOS[dst]
    y = info.alarma.to_numpy()
    va, te = (info.conjunto == 'val').to_numpy(), (info.conjunto == 'test').to_numpy()
    nv, nt = va & (y == 0), te & (y == 0)
    base_var = None
    for src in [dst] + [e for e in EQUIPOS if e != dst]:
        for modo in ([PROPIO] if src == dst else [DEL_EQUIPO, DEL_MODELO]):
            mu, sd = NORM[src if modo == DEL_MODELO else dst]
            ev = errores(MOD[src], ((X - mu) / sd).astype(np.float32))
            e = ev.mean(axis=1)
            if src == dst:
                ok = np.allclose(e, info.error_reconstruccion.to_numpy(), rtol=1e-3, atol=1e-6)
                print(f'{dst}: error del modelo propio igual al del paso 04: {"sí" if ok else "NO (revisar)"}')
                base_var = np.median(ev[nv], axis=0)
            umbral = np.percentile(e[nv], PERCENTIL)
            filas.append({'modelo_de': src, 'aplicado_a': dst, 'normalizador': modo,
                          'auc_val': auc(y[va], e[va]), 'auc_test': auc(y[te], e[te]),
                          'error_normal_val_mediano': float(np.median(e[nv])),
                          'error_normal_test_mediano': float(np.median(e[nt])) if nt.any() else np.nan,
                          'pct_normales_test_marcadas': 100 * float((e[nt] > umbral).mean()) if nt.any() else np.nan,
                          'recall_test': 100 * float((e[te & (y == 1)] > umbral).mean())
                          if (te & (y == 1)).any() else np.nan})
            if modo != DEL_MODELO:
                med = np.median(ev[nv], axis=0)
                for j, v in enumerate(VARIABLES):
                    filas_v.append({'modelo_de': src, 'aplicado_a': dst, 'variable': v,
                                    'error_normal_val_mediano': float(med[j]),
                                    'veces_el_propio': float(med[j] / base_var[j]) if base_var[j] > 0 else np.nan})
tr = pd.DataFrame(filas)
_prop = tr[tr.normalizador == PROPIO].set_index('aplicado_a').error_normal_val_mediano
tr['error_normal_veces_el_propio'] = np.where(tr.normalizador == DEL_MODELO, np.nan,
                                              tr.error_normal_val_mediano / tr.aplicado_a.map(_prop))
trv = pd.DataFrame(filas_v)
tr.round(4).to_csv(f'{RUTA_BASE}/04e_transferencia.csv', sep=';', decimal=',', index=False, encoding='utf-8-sig')
trv.round(4).to_csv(f'{RUTA_BASE}/04e_transferencia_variables.csv', sep=';', decimal=',', index=False,
                    encoding='utf-8-sig')


def matriz(col, modo):
    t = tr[(tr.normalizador == modo) | (tr.normalizador == PROPIO)]
    return t.pivot(index='modelo_de', columns='aplicado_a', values=col).reindex(index=EQUIPOS, columns=EQUIPOS)


print('\nFilas: modelo entrenado con; columnas: equipo evaluado. La diagonal es el modelo propio.')
for titulo, col, modo in [('AUC de validación, normalizador del equipo evaluado', 'auc_val', DEL_EQUIPO),
                          ('AUC de prueba, normalizador del equipo evaluado', 'auc_test', DEL_EQUIPO),
                          ('Error de las normales de validación, veces el del modelo propio',
                           'error_normal_veces_el_propio', DEL_EQUIPO),
                          ('% de normales de prueba marcadas (umbral recalibrado en el equipo evaluado)',
                           'pct_normales_test_marcadas', DEL_EQUIPO),
                          ('AUC de prueba, normalizador del modelo (se transfiere todo)', 'auc_test', DEL_MODELO)]:
    print(f'\n{titulo}:')
    _fmt = '{:.2f}' if 'auc' in col or 'veces' in col else '{:.1f}'
    print(matriz(col, modo).to_string(float_format=_fmt.format))

print('\nVariables que el modelo de otro equipo reconstruye peor (normales de validación, veces el propio):')
for (src, dst), t in trv[trv.modelo_de != trv.aplicado_a].groupby(['modelo_de', 'aplicado_a'], sort=False):
    top = t.sort_values('veces_el_propio', ascending=False).head(3)
    print(f'  modelo {src} en {dst}: ' + ', '.join(f'{r.variable} {r.veces_el_propio:.1f}×' for r in top.itertuples()))
print('\nLectura: si en cada columna los modelos de otros equipos quedan cerca del propio (AUC parecido y')
print('error de las normales cerca de 1×), los equipos se comportan igual y un modelo podría servir para')
print('todos. Si el propio es claramente mejor, cada equipo necesita su modelo; las variables de arriba')
print('dicen en qué se diferencian.')


# =============================================================================
# 4. FIGURA
# =============================================================================
paneles = [('AUC de prueba\n(normalizador del equipo evaluado)', matriz('auc_test', DEL_EQUIPO), (0.5, 1.0), '{:.2f}'),
           ('Error de las normales de validación,\nveces el del modelo propio',
            matriz('error_normal_veces_el_propio', DEL_EQUIPO), None, '{:.1f}×'),
           ('AUC de prueba\n(normalizador del modelo)', matriz('auc_test', DEL_MODELO), (0.5, 1.0), '{:.2f}')]
fig, ejes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (tit, mat, lim, fmt) in zip(ejes, paneles):
    v = mat.to_numpy(dtype=float)
    if lim is None:
        lim = (1.0, max(2.0, float(np.nanmax(v))))
    im = ax.imshow(v, cmap=AZULES, vmin=lim[0], vmax=lim[1])
    for i in range(len(EQUIPOS)):
        for j in range(len(EQUIPOS)):
            if np.isfinite(v[i, j]):
                claro = (v[i, j] - lim[0]) / (lim[1] - lim[0]) < 0.55
                ax.text(j, i, fmt.format(v[i, j]), ha='center', va='center', fontsize=11,
                        color='#0b0b0b' if claro else 'white', fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(len(EQUIPOS)))
    ax.set_xticklabels(EQUIPOS)
    ax.set_yticks(range(len(EQUIPOS)))
    ax.set_yticklabels(EQUIPOS)
    ax.set_xlabel('Equipo evaluado')
    ax.set_ylabel('Modelo entrenado con')
    ax.set_title(tit, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle('Transferencia entre equipos: la diagonal (en negrita) es el modelo propio')
fig.tight_layout()
fig.savefig(f'{RUTA_BASE}/04e_transferencia.png', dpi=150)
plt.close(fig)
print(f'\nGuardado en {RUTA_BASE}: 04e_transferencia.csv, 04e_transferencia_variables.csv, 04e_transferencia.png')
