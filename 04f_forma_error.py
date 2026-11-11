# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04f — FORMA DEL ERROR: ¿EL MSE SUBE POR DESFASE?
=============================================================================
Lo comentado en la reunión del 6 de noviembre: mirar el error de cada variable
en gráficos de líneas, porque el MSE compara instante a instante y castiga una
reconstrucción con la forma correcta pero corrida en el tiempo. El error de
cada variable se separa en nivel + amplitud + forma (suman exacto; el desfase
cae en la forma) y se recalcula permitiendo un desfase de hasta 1 min.
Entrada:  {EQUIPO}_limpio.csv y {EQUIPO}_corte_split.json (paso 01), normalizador_W60.npz,
          autoencoder_W60_normales[_sep].pt y su config (paso 03b), {EQUIPO}_error_reconstruccion_W60.csv (paso 04)
Salida:   04f_forma_error.csv, 04f_delta_componentes.csv, 04f_deteccion.csv, 04f_forma_error.png,
          04f_ejemplos.png
Se decide en validación; prueba solo se informa.
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

EQUIPO = os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# =============================================================================
# 0. CONSTANTES
# =============================================================================
W = 60
SEED = 42
PERCENTIL = 95
LAG = 6                         # desfase máximo: 6 muestras de 10 s = 1 min
SEG = 10                        # segundos por muestra
N_BOOT = 1000
BLOQUE = 36                     # bloques de 6 h para los intervalos de confianza
MIN_VENTANAS = 10
BAJA_PESA = 30                  # % que tiene que bajar el error para decir que el desfase pesa
LLAVES = ['tramo_id', 'ventana_id']
VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
UNIDAD = {'CM': '%', 'PI': '%', 'PDF': 'kPa', 'PEL': 'kPa', **{v: '°C' for v in ['T7', 'T8', 'T9', 'T1', 'T2', 'T5']},
          **{v: 'mm/s' for v in ['V1', 'V2', 'V3', 'V4']}}
PARTES = {'nivel': '#2a78d6', 'amplitud': '#eb6834', 'forma': '#1baf7a'}
ORIGINAL, RECONS, TINTA, SUAVE = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
GRIS, VIOLETA = '#b8b7b2', '#6250d6'

PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
if PUNTO_PARADA not in ('perdida', 'separacion'):
    raise SystemExit(f"PUNTO_PARADA tiene que ser 'perdida' o 'separacion', no {PUNTO_PARADA!r}.")
NOMBRE_PT = f'autoencoder_W{W}_normales_sep.pt' if PUNTO_PARADA == 'separacion' else f'autoencoder_W{W}_normales.pt'
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


def guardar(df, nombre):
    df.to_csv(salida(nombre), sep=';', decimal=',', index=False, encoding='utf-8-sig')


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


coma = lambda v, d=2: f'{v:.{d}f}'.replace('.', ',')


# =============================================================================
# 1. MODELO Y VENTANAS (los mismos de los pasos 04 y 04e)
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


def cargar_modelo():
    ruta = next((p for p in [f'{RUTA_EQ}/{NOMBRE_PT}', f'{RUTA_EQ}/CAMBIOS/{NOMBRE_PT}'] if os.path.exists(p)), None)
    if ruta is None:
        raise SystemExit(f'Falta el modelo ({NOMBRE_PT}): corre el 03b.')
    _c = os.path.join(os.path.dirname(ruta), f'autoencoder_W{W}_config.json')
    cfg = json.load(open(_c)) if os.path.exists(_c) else {'tipo': 'colapsado', 'latent_dim': 64}
    arq = {k: cfg[k] for k in ('d_model', 'nhead', 'num_layers', 'dim_ff', 'dropout') if k in cfg}
    if cfg['tipo'] == 'temporal':
        mod = AutoencoderTransformerTemporal(len(VARIABLES), W, latent_dim=cfg['latent_dim'], **arq)
    else:
        mod = AutoencoderTransformer(len(VARIABLES), W, **arq)
    mod.load_state_dict(torch.load(ruta, map_location=device))
    return mod.to(device).eval(), cfg


def cargar_limpio(ruta):
    cab = pd.read_csv(ruta, nrows=0)
    col = 'timestamp' if 'timestamp' in cab.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col], low_memory=False)
    return df.set_index(col).sort_index().rename_axis('timestamp')


def ventanas():
    df = cargar_limpio(f'{RUTA_EQ}/{EQUIPO}_limpio.csv')
    faltan = [v for v in VARIABLES if v not in df.columns]
    if faltan:
        raise SystemExit(f'Faltan variables en el limpio: {faltan}')
    corte = json.load(open(f'{RUTA_EQ}/{EQUIPO}_corte_split.json'))
    t_train, t_val = pd.Timestamp(corte['T_train']), pd.Timestamp(corte['T_val'])
    salto = df.index.to_series().diff().dt.total_seconds() > SEG
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
    err = pd.read_csv(f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv',
                      usecols=LLAVES + ['conjunto', 'n_reglas_activas', 'error_reconstruccion'])
    info = info.merge(err.rename(columns={'conjunto': 'conjunto_04'}), on=LLAVES, how='left', validate='one_to_one')
    if info.error_reconstruccion.isna().any() or (info.conjunto != info.conjunto_04).any():
        raise SystemExit('Las ventanas no coinciden con las del paso 04. ¿Mismo limpio y mismo corte?')
    info['alarma'] = (info.n_reglas_activas > 0).astype(int)
    norm = np.load(f'{RUTA_EQ}/normalizador_W{W}.npz', allow_pickle=True)
    _v = list(norm['variables'])
    mu = pd.Series(norm['media'], index=_v)[VARIABLES].to_numpy(dtype=np.float32)
    sd = pd.Series(norm['std'], index=_v)[VARIABLES].to_numpy(dtype=np.float32)
    return np.stack(X), info, mu, sd


print('=' * 72)
print(f'PASO 04f — FORMA DEL ERROR POR VARIABLE ({EQUIPO})')
print('=' * 72)
X, info, MU, SD = ventanas()
MOD, CFG = cargar_modelo()
Xn = ((X - MU) / SD).astype(np.float32)
with torch.no_grad():
    Rn = np.concatenate([MOD(torch.from_numpy(Xn[i:i + 256]).to(device)).cpu().numpy()
                         for i in range(0, len(Xn), 256)])
Xn, Rn = Xn.astype(np.float64), Rn.astype(np.float64)
_ok = np.allclose(((Rn - Xn) ** 2).mean(axis=(1, 2)), info.error_reconstruccion, rtol=1e-3, atol=1e-6)
y = info.alarma.to_numpy()
va, te = (info.conjunto == 'val').to_numpy(), (info.conjunto == 'test').to_numpy()
print(f'Dispositivo: {device} | modelo {CFG["tipo"]}, k = {CFG["latent_dim"]} | punto de parada: {PUNTO_PARADA}')
print(f'Ventanas: validación {va.sum():,} | prueba {te.sum():,} | error igual al del paso 04: {"sí" if _ok else "NO (revisar)"}')


# =============================================================================
# 2. MSE = NIVEL + AMPLITUD + FORMA, Y ERROR CON DESFASE DE HASTA 1 MIN
# =============================================================================
mx, my = Xn.mean(axis=1), Rn.mean(axis=1)
sx, sy = Xn.std(axis=1), Rn.std(axis=1)
cov = ((Xn - mx[:, None, :]) * (Rn - my[:, None, :])).mean(axis=1)
MSE = ((Xn - Rn) ** 2).mean(axis=1)                 # (ventanas, variables)
NIV = (mx - my) ** 2                                 # promedios distintos
AMP = (sx - sy) ** 2                                 # variabilidad distinta
FOR = np.clip(2 * (sx * sy - cov), 0, None)          # no se mueven juntas (aquí cae el desfase)
print(f'nivel + amplitud + forma = MSE: {"sí" if np.allclose(NIV + AMP + FOR, MSE, rtol=1e-6, atol=1e-9) else "NO"}')

_c = slice(LAG, W - LAG)
E = np.stack([((Xn[:, _c] - Rn[:, LAG + L:W - LAG + L]) ** 2).mean(axis=1) for L in range(-LAG, LAG + 1)])
MSE_C = E[LAG]                                       # sin desfase, en el tramo central
MSE_D = E.min(axis=0)                                # con el mejor desfase
DESF = (E.argmin(axis=0) - LAG) * SEG                # segundos; positivo = la reconstrucción va atrasada
del E
normal_val = va & (y == 0)
P95 = np.percentile(MSE[normal_val], PERCENTIL, axis=0)


# =============================================================================
# 3. ¿DE DÓNDE VIENE EL ERROR DE CADA VARIABLE?
# =============================================================================
print('\n' + '=' * 72)
print('3. ¿DE DÓNDE VIENE EL ERROR DE CADA VARIABLE? (% del MSE)')
print('=' * 72)
GRUPOS = {'normales': y == 0, 'anómalas': y == 1}
filas = []
for cj, c in [('val', va), ('test', te)]:
    for gn, g in GRUPOS.items():
        s = c & g
        if s.sum() < MIN_VENTANAS:
            continue
        for j, v in enumerate(VARIABLES):
            tot = MSE[s, j].sum()
            sobre = s & (MSE_D[:, j] <= 0.5 * MSE_C[:, j]) & (MSE_C[:, j] > 0)
            filas.append({'conjunto': cj, 'grupo': gn, 'variable': v, 'ventanas': int(s.sum()),
                          'mse_medio': MSE[s, j].mean(),
                          'pct_nivel': 100 * NIV[s, j].sum() / tot, 'pct_amplitud': 100 * AMP[s, j].sum() / tot,
                          'pct_forma': 100 * FOR[s, j].sum() / tot,
                          'baja_con_desfase_pct': 100 * (1 - MSE_D[s, j].sum() / MSE_C[s, j].sum()),
                          'pct_ventanas_sobre_todo_desfase': 100 * sobre.sum() / s.sum(),
                          'desfase_mediano_s': float(np.median(DESF[sobre, j])) if sobre.any() else np.nan,
                          'pct_atrasada': 100 * float((DESF[sobre, j] > 0).mean()) if sobre.any() else np.nan})
forma = pd.DataFrame(filas)
guardar(forma.round(4), '04f_forma_error.csv')

for cj in ('val', 'test'):
    print(f'\n{"Validación" if cj == "val" else "Prueba (solo se informa)"}, todas las variables juntas:')
    for gn, g in GRUPOS.items():
        s = (va if cj == 'val' else te) & g
        if s.sum() < MIN_VENTANAS:
            continue
        tot = MSE[s].sum()
        print(f'  {gn:<9} nivel {100 * NIV[s].sum() / tot:4.0f} % | amplitud {100 * AMP[s].sum() / tot:4.0f} % | '
              f'forma {100 * FOR[s].sum() / tot:4.0f} % | con desfase de hasta 1 min el error baja '
              f'{100 * (1 - MSE_D[s].sum() / MSE_C[s].sum()):.0f} %')
t = forma[(forma.conjunto == 'val') & (forma.grupo == 'anómalas')].set_index('variable')
n_ = forma[(forma.conjunto == 'val') & (forma.grupo == 'normales')].set_index('variable')
if len(t):
    print('\nValidación, ventanas anómalas, por variable:')
    tab = t[['mse_medio', 'pct_nivel', 'pct_amplitud', 'pct_forma', 'baja_con_desfase_pct']].copy()
    tab['baja_en_normales_pct'] = n_.baja_con_desfase_pct.reindex(tab.index)
    tab['ventanas_sobre_todo_desfase_pct'] = t.pct_ventanas_sobre_todo_desfase
    print(tab.round(1).to_string())
    pesa = [v for v in VARIABLES if v in tab.index and tab.loc[v, 'baja_con_desfase_pct'] >= BAJA_PESA
            and tab.loc[v, 'baja_con_desfase_pct'] > tab.loc[v, 'baja_en_normales_pct'] + 10]
    print('\nVariables donde el desfase pesa (el error de las anómalas baja 30 % o más y más que en las normales): '
          + (', '.join(f'{v} ({tab.loc[v, "baja_con_desfase_pct"]:.0f} %, desfase típico '
                       f'{t.loc[v, "desfase_mediano_s"]:+.0f} s)' for v in pesa) if pesa else 'ninguna.'))
print('nivel: la reconstrucción va más arriba o más abajo. amplitud: oscila más o menos. forma: no se')
print('mueven juntas (desfase, ruido o forma distinta). baja_con_desfase: cuánto baja el error si se corre')
print('la reconstrucción hasta 1 min; algo baja siempre, por eso se compara con las normales.')


# =============================================================================
# 4. EL Δ DEL 04c SEPARADO EN SUS PARTES, Y CON DESFASE TOLERADO
# =============================================================================
print('\n' + '=' * 72)
print('4. ¿EL Δ DE CADA VARIABLE VIENE DE NIVEL, AMPLITUD O FORMA?')
print('=' * 72)
filas_d = []
for cj, c in [('val', va), ('test', te)]:
    nn_, aa = c & (y == 0), c & (y == 1)
    if nn_.sum() < MIN_VENTANAS or aa.sum() < MIN_VENTANAS:
        continue
    dif = lambda A: A[aa].mean(axis=0) - A[nn_].mean(axis=0)
    d, dn, da, dfo, dc, dd = dif(MSE), dif(NIV), dif(AMP), dif(FOR), dif(MSE_C), dif(MSE_D)
    for j, v in enumerate(VARIABLES):
        filas_d.append({'conjunto': cj, 'variable': v, 'delta': d[j], 'pct_delta': 100 * d[j] / d.sum(),
                        'delta_nivel': dn[j], 'delta_amplitud': da[j], 'delta_forma': dfo[j],
                        'delta_con_desfase': dd[j], 'pct_delta_con_desfase': 100 * dd[j] / dd.sum(),
                        'delta_sin_desfase_central': dc[j]})
dcomp = pd.DataFrame(filas_d)
guardar(dcomp.round(6), '04f_delta_componentes.csv')
for cj in ('val', 'test'):
    t = dcomp[dcomp.conjunto == cj].sort_values('delta', ascending=False)
    if t.empty:
        continue
    print(f'\n{"Validación" if cj == "val" else "Prueba (solo se informa)"}: variables con más Δ y de qué parte viene')
    for r in t.head(5).itertuples():
        print(f'  {r.variable:<4} {r.pct_delta:5.0f} % del Δ | nivel {coma(r.delta_nivel, 3)} · amplitud '
              f'{coma(r.delta_amplitud, 3)} · forma {coma(r.delta_forma, 3)} | con desfase tolerado: '
              f'{r.pct_delta_con_desfase:.0f} % del Δ')
    top = list(t.variable.head(3))
    top_d = list(t.sort_values('delta_con_desfase', ascending=False).variable.head(3))
    print(f'  Las 3 con más Δ: {", ".join(top)} | tolerando desfase: {", ".join(top_d)} '
          f'({"no cambia" if top == top_d else "cambia"})')


# =============================================================================
# 5. ¿TOLERAR EL DESFASE SEPARA MEJOR? (se decide en validación)
# =============================================================================
print('\n' + '=' * 72)
print('5. ¿TOLERAR EL DESFASE SEPARA MEJOR LAS VENTANAS CON ALARMA?')
print('=' * 72)
PUNTAJES = {'error total (oficial)': MSE.mean(axis=1),
            'error sin desfase, tramo central': MSE_C.mean(axis=1),
            'error con desfase de hasta 1 min': MSE_D.mean(axis=1),
            'solo nivel y amplitud': (NIV + AMP).mean(axis=1),
            'solo forma': FOR.mean(axis=1)}
det = pd.DataFrame([{'puntaje': k, 'auc_val': auc(y[va], s[va]), 'auc_test': auc(y[te], s[te])}
                    for k, s in PUNTAJES.items()])
rng = np.random.default_rng(SEED)
_bloque = info.tramo_id.to_numpy().astype(np.int64) * 100000 + info.ventana_id.to_numpy() // BLOQUE


def ic_dif(a, b, sel):
    idx = np.flatnonzero(sel)
    yy, a, b = y[idx], a[idx], b[idx]
    grupos = pd.Series(np.arange(len(idx))).groupby(_bloque[idx]).indices
    claves = np.array(list(grupos))
    vals = []
    for _ in range(N_BOOT):
        m_ = np.concatenate([grupos[k] for k in rng.choice(claves, len(claves))])
        if 0 < yy[m_].sum() < len(m_):
            vals.append(auc(yy[m_], a[m_]) - auc(yy[m_], b[m_]))
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) > N_BOOT // 2 else (np.nan, np.nan)
    return auc(yy, a) - auc(yy, b), lo, hi


_a, _b = PUNTAJES['error con desfase de hasta 1 min'], PUNTAJES['error sin desfase, tramo central']
LINEAS = []
for cj, c in [('val', va), ('test', te)]:
    dif, lo, hi = ic_dif(_a, _b, c)
    LINEAS.append(f'con desfase − sin desfase ({"validación" if cj == "val" else "prueba"}): '
                  f'{dif:+.3f} [IC 95 % {lo:+.3f}, {hi:+.3f}]')
    det.loc[len(det)] = {'puntaje': LINEAS[-1], 'auc_val': np.nan, 'auc_test': np.nan}
    if cj == 'val':
        DIF_VAL = (dif, lo, hi)
guardar(det.round(4), '04f_deteccion.csv')
print(det.iloc[:len(PUNTAJES)].round(3).to_string(index=False))
print('\n'.join(LINEAS))
_d, _lo, _hi = DIF_VAL
if np.isnan(_lo):
    print('No hay ventanas suficientes para el intervalo en validación.')
elif _lo > 0:
    print('En validación, tolerar el desfase separa mejor: conviene usar un error que lo tolere.')
elif _hi < 0:
    print('En validación, tolerar el desfase separa peor: el MSE se queda.')
else:
    print('En validación, tolerar el desfase no cambia la separación: el MSE se queda.')
print('Diferencia de AUC con IC 95 % remuestreando bloques de 6 h; la comparación justa es contra el error')
print('sin desfase del mismo tramo central (los primeros y últimos 60 s no entran en ninguno de los dos).')


# =============================================================================
# 6. FIGURAS
# =============================================================================
t = forma[(forma.conjunto == 'val')]
fig, ejes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
an = t[t.grupo == 'anómalas'].set_index('variable').reindex(VARIABLES)
no = t[t.grupo == 'normales'].set_index('variable').reindex(VARIABLES)
pos = np.arange(len(VARIABLES))[::-1]
izq = np.zeros(len(VARIABLES))
for parte, col in PARTES.items():
    v = an[f'pct_{parte}'].fillna(0).to_numpy()
    ejes[0].barh(pos, v, left=izq, color=col, edgecolor='white', linewidth=1.5, height=0.7, label=parte)
    izq += v
ejes[0].set_yticks(pos)
ejes[0].set_yticklabels(VARIABLES)
ejes[0].set_xlim(0, 100)
ejes[0].set_xlabel('% del error (ventanas con alarma, validación)')
ejes[0].set_title('¿De dónde viene el error?', loc='left', fontsize=11, pad=34)
ejes[0].legend(frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=9, borderaxespad=0.2)
for k, (df_, col, et) in enumerate([(no, GRIS, 'ventanas normales'), (an, VIOLETA, 'ventanas con alarma')]):
    ejes[1].barh(pos + (0.18 if k == 0 else -0.18), df_.baja_con_desfase_pct.fillna(0), height=0.34, color=col,
                 edgecolor='white', linewidth=1, label=et)
ejes[1].axvline(BAJA_PESA, color=SUAVE, lw=1, ls='--')
ejes[1].text(BAJA_PESA, 0.01, f' {BAJA_PESA} %: el desfase pesa', transform=ejes[1].get_xaxis_transform(),
             color=SUAVE, fontsize=8, va='bottom')
ejes[1].set_xlabel('% que baja el error si se corre la reconstrucción hasta 1 min (validación)')
ejes[1].set_title('¿Cuánto pesa el desfase?', loc='left', fontsize=11, pad=34)
ejes[1].legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=9, borderaxespad=0.2)
for ax in ejes:
    ax.grid(alpha=0.25, axis='x')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
fig.suptitle(f'{EQUIPO}: el MSE de cada variable separado en nivel, amplitud y forma', x=0.01, ha='left', fontsize=12)
fig.tight_layout()
fig.savefig(salida('04f_forma_error.png'), dpi=150)
plt.close(fig)

# ejemplos en gráficos de líneas, en unidades del sensor
_dv = dcomp[dcomp.conjunto == 'val'].sort_values('delta', ascending=False)
COLS = list(_dv.variable.head(4)) if len(_dv) else VARIABLES[:4]
ERR = MSE.mean(axis=1)
umbral = np.percentile(ERR[normal_val], PERCENTIL)


def mediana_de(sel):
    idx = np.flatnonzero(sel)
    return idx[np.argsort(ERR[idx])[len(idx) // 2]] if len(idx) else None


_i = mediana_de(va & (y == 1) & (ERR > umbral))
filas_ej = [('Normal típica', mediana_de(normal_val)),
            ('Con alarma típica', _i if _i is not None else mediana_de(va & (y == 1)))]
minutos = np.arange(W) * SEG / 60
fig, ejes = plt.subplots(3, len(COLS), figsize=(4.2 * len(COLS), 9.6), squeeze=False)
for col, v in enumerate(COLS):
    j = VARIABLES.index(v)
    cand = np.flatnonzero(va & (MSE_C[:, j] > P95[j]))
    i_desf = cand[np.argmax(1 - MSE_D[cand, j] / MSE_C[cand, j])] if len(cand) else None
    for fila, (nombre, i) in enumerate(filas_ej + [('Donde más pesa el desfase', i_desf)]):
        ax = ejes[fila][col]
        if i is None:
            ax.axis('off')
            continue
        orig, rec = X[i, :, j], Rn[i, :, j] * SD[j] + MU[j]
        ax.plot(minutos, orig, color=ORIGINAL, lw=2, label='original')
        ax.plot(minutos, rec, color=RECONS, lw=2, label='reconstrucción')
        m = MSE[i, j]
        partes = f'nivel {100 * NIV[i, j] / m:.0f} % · amplitud {100 * AMP[i, j] / m:.0f} % · forma {100 * FOR[i, j] / m:.0f} %'
        extra = (f'\ncorrida {DESF[i, j]:+.0f} s: el error baja {100 * (1 - MSE_D[i, j] / MSE_C[i, j]):.0f} %'
                 if fila == 2 and MSE_C[i, j] > 0 else '')
        ax.set_title(f'{v} · MSE {coma(m)}', fontsize=10, loc='left', pad=30 if extra else 18)
        ax.text(0, 1.02, partes + extra, transform=ax.transAxes, fontsize=7.5, color=SUAVE, va='bottom')
        ax.set_ylabel(f'{nombre}\n({UNIDAD.get(v, "")})' if col == 0 else UNIDAD.get(v, ''), fontsize=9)
        ax.grid(alpha=0.25)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        if fila == 2:
            ax.set_xlabel('Minutos dentro de la ventana')
_h, _l = next((a.get_legend_handles_labels() for a in np.ravel(ejes) if a.get_legend_handles_labels()[0]), ([], []))
fig.legend(_h, _l, loc='upper right', ncol=2, frameon=False, fontsize=9)
fig.suptitle(f'{EQUIPO}: original y reconstrucción de las variables con más Δ (validación)', x=0.01, ha='left',
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(salida('04f_ejemplos.png'), dpi=150)
plt.close(fig)
print(f'\nGuardado en {RUTA_EQ}: 04f_forma_error.csv, 04f_delta_componentes.csv, 04f_deteccion.csv, '
      f'04f_forma_error.png, 04f_ejemplos.png')
