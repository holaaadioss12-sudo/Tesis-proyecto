# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04 — DETECCIÓN: ERROR DE RECONSTRUCCIÓN Y ERROR POR VARIABLE
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

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', device)

W = 60
ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
CANDIDATOS = {
    'AE_normales': [f'{RUTA_EQ}/autoencoder_W{W}_normales.pt',
                    f'{RUTA_EQ}/CAMBIOS/autoencoder_W{W}_normales.pt'],
    'AE_normales_sep': [f'{RUTA_EQ}/autoencoder_W{W}_normales_sep.pt',
                        f'{RUTA_EQ}/CAMBIOS/autoencoder_W{W}_normales_sep.pt'],
}
MODELO_PRINCIPAL = 'AE_normales'
import os as _os
MODELOS = {}
for _n, _rutas in CANDIDATOS.items():
    _hit = next((r for r in _rutas if _os.path.exists(r)), None)
    if _hit:
        MODELOS[_n] = _hit
    else:
        print(f'AVISO: {_n} no está en ninguna de estas rutas:')
        for _r in _rutas:
            print(f'        {_r}')
if MODELO_PRINCIPAL not in MODELOS:
    raise SystemExit(f'Falta el modelo principal ({MODELO_PRINCIPAL}). '
                     'Corre el 03b_autoencoder_entrenamiento.py antes que este.')
RUTA_MODELO = MODELOS[MODELO_PRINCIPAL]
print(f'Modelos encontrados: {len(MODELOS)} de {len(CANDIDATOS)}')
for _n, _r in MODELOS.items():
    print(f'  {_n:<18} {_r}')
RUTA_NORMALIZADOR = f'{RUTA_EQ}/normalizador_W{W}.npz'
ARCHIVO_SALIDA = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
RUTA_CONFIG = _os.path.join(_os.path.dirname(RUTA_MODELO), f'autoencoder_W{W}_config.json')

import sys as _sys
import json as _json
if RUTA_BASE not in _sys.path:
    _sys.path.append(RUTA_BASE)
from reglas_engine import POR_ID, IDS as IDS_REGLAS

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']


# =============================================================================
# 1. RECONSTRUIR EXACTAMENTE LAS VENTANAS DEL PASO 03b
# =============================================================================
print('=' * 70)
print(f'PASO 04 — DETECCIÓN (W = {W})')
print('=' * 70)

def cargar_limpio(ruta):
    """Acepta el CSV con la columna de tiempo llamada 'timestamp' o 'Time'."""
    cabecera = pd.read_csv(ruta, nrows=0)
    col_tiempo = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col_tiempo], low_memory=False)
    return df.set_index(col_tiempo).sort_index().rename_axis('timestamp')


df = cargar_limpio(ARCHIVO_LIMPIO)
VARIABLES = [v for v in VARIABLES if v in df.columns]


def segmentar_tramos_continuos(df, freq='10s', max_gap=1):
    idx = df.index.to_series()
    gaps = idx.diff() > pd.Timedelta(freq) * max_gap
    tramo_id = gaps.cumsum()
    df = df.copy()
    df['tramo_id'] = tramo_id
    return [g.drop(columns='tramo_id') for _, g in df.groupby('tramo_id')]


tramos = segmentar_tramos_continuos(df)
tramos = [t for t in tramos if len(t) >= W]

# ---- LA RE-NUMERACIÓN QUE NO SE PUEDE OLVIDAR --------------------------
tramos_con_id = [(i, t) for i, t in enumerate(tramos)]
print(f'Tramos útiles: {len(tramos_con_id):,}   <- esperado: 1.929')


# El mismo corte por fecha que usan los pasos 03 y 04.
import json as _json
_corte = _json.load(open(f'{RUTA_EQ}/{EQUIPO}_corte_split.json'))
_T_TRAIN, _T_VAL = pd.Timestamp(_corte['T_train']), pd.Timestamp(_corte['T_val'])
_ini5 = np.array([t[1].index[0] for t in tramos_con_id])
tramos_train = [t for t, i in zip(tramos_con_id, _ini5) if i <= _T_TRAIN]
tramos_val = [t for t, i in zip(tramos_con_id, _ini5)
              if _T_TRAIN < i <= _T_VAL]
tramos_test = [t for t, i in zip(tramos_con_id, _ini5) if i > _T_VAL]
print(f'Tramos: train={len(tramos_train)} | val={len(tramos_val)} | '
      f'test={len(tramos_test)}')
print(f'  corte unico: T_train = {_T_TRAIN} | T_val = {_T_VAL}')

norm = np.load(RUTA_NORMALIZADOR, allow_pickle=True)
vars_guardadas = list(norm['variables'])
params_norm = {
    'media': pd.Series(norm['media'], index=vars_guardadas),
    'std': pd.Series(norm['std'], index=vars_guardadas),
}
print(f'Normalizador cargado del paso 03b ({len(vars_guardadas)} variables)')
if vars_guardadas != VARIABLES:
    print('⚠ Las variables no coinciden con las del paso 03b. Revisa antes de seguir.')


def aplicar_normalizador(tid_tramo, variables, params):
    tid, tramo = tid_tramo
    tramo = tramo.copy()
    tramo[variables] = (tramo[variables] - params['media']) / params['std']
    return tid, tramo


tramos_val_n = [aplicar_normalizador(t, VARIABLES, params_norm) for t in tramos_val]
tramos_test_n = [aplicar_normalizador(t, VARIABLES, params_norm) for t in tramos_test]


def construir_ventanas_con_id(tramos, variables, W, stride):
    ventanas, ids = [], []
    for tid, t in tramos:
        arr = t[variables].to_numpy(dtype=np.float32)
        vid = 0
        for inicio in range(0, len(arr) - W + 1, stride):
            ventanas.append(arr[inicio:inicio + W])
            ids.append((tid, vid))
            vid += 1
    if not ventanas:
        return np.empty((0, W, len(variables)), dtype=np.float32), []
    return np.stack(ventanas), ids


# Sin traslape: una ventana por bloque, igual que en el paso 02
X_val, ids_val = construir_ventanas_con_id(tramos_val_n, VARIABLES, W, W)
X_test, ids_test = construir_ventanas_con_id(tramos_test_n, VARIABLES, W, W)
X_eval = np.concatenate([X_val, X_test], axis=0)
ids_eval = ids_val + ids_test
conjunto_eval = np.array(['val'] * len(X_val) + ['test'] * len(X_test))
print(f'Ventanas a evaluar: {len(X_eval):,}   '
      f'(val {len(X_val):,} | test {len(X_test):,})')


# =============================================================================
# 2. CARGAR EL MODELO Y CALCULAR EL ERROR POR VENTANA
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
        return self.encoder(self.pos_enc(self.embed(x))).mean(dim=1)

    def decode(self, latente):
        h = latente.unsqueeze(1).repeat(1, self.W, 1)
        return self.salida(self.decoder(self.pos_enc(h)))

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
        self.a_latente = nn.Linear(d_model, latent_dim)
        self.desde_latente = nn.Linear(latent_dim, d_model)
        capa_decoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(capa_decoder, num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def encode(self, x):
        return self.a_latente(self.encoder(self.pos_enc(self.embed(x))))

    def decode(self, latente):
        return self.salida(self.decoder(self.pos_enc(self.desde_latente(latente))))

    def forward(self, x):
        return self.decode(self.encode(x))


# el tipo de modelo lo dice la configuración que guardó el paso 03b
CONFIG_AE = (_json.load(open(RUTA_CONFIG)) if _os.path.exists(RUTA_CONFIG)
             else {'tipo': 'colapsado', 'latent_dim': 64})
if CONFIG_AE['tipo'] == 'temporal':
    modelo = AutoencoderTransformerTemporal(n_vars=len(VARIABLES), W=W,
                                            latent_dim=CONFIG_AE['latent_dim']).to(device)
else:
    modelo = AutoencoderTransformer(n_vars=len(VARIABLES), W=W).to(device)
print(f"Autoencoder: {CONFIG_AE['tipo']}, latent_dim = {CONFIG_AE['latent_dim']}"
      + ('' if _os.path.exists(RUTA_CONFIG) else '  (sin config del paso 03b: modelo anterior)'))
X_eval_t = torch.from_numpy(X_eval)
df_err = pd.DataFrame(ids_eval, columns=['tramo_id', 'ventana_id'])
df_err['conjunto'] = conjunto_eval
EVAR = [f'evar_{v}' for v in VARIABLES]

for nombre, ruta in MODELOS.items():
    modelo.load_state_dict(torch.load(ruta, map_location=device))
    modelo.eval()
    ev = []
    with torch.no_grad():
        for i in range(0, len(X_eval_t), 256):
            batch = X_eval_t[i:i + 256].to(device)
            ev.append(((modelo(batch) - batch) ** 2).mean(dim=1).cpu().numpy())
    ev = np.concatenate(ev)                     # (ventanas, 14): error de cada variable
    df_err[f'err_{nombre}'] = ev.mean(axis=1)   # error de la ventana: promedio de los 14
    if nombre == MODELO_PRINCIPAL:
        for _j, _c in enumerate(EVAR):
            df_err[_c] = ev[:, _j]
    print(f'  {nombre:<18} evaluado | error medio {ev.mean():.5f}')

if f'err_{MODELO_PRINCIPAL}' not in df_err.columns:
    raise SystemExit(f'Falta el modelo principal ({MODELO_PRINCIPAL}). '
                     'Corre el 03b_autoencoder_entrenamiento.py antes que este.')
df_err['error_reconstruccion'] = df_err[f'err_{MODELO_PRINCIPAL}']
print(f'\nColumna error_reconstruccion = {MODELO_PRINCIPAL}')


# =============================================================================
# 3. CRUZAR CON LAS ETIQUETAS (que el modelo nunca vio)
# =============================================================================
print('\n' + '=' * 70)
print('CRUZANDO CON LAS ETIQUETAS DEL MOTOR DE REGLAS')
print('=' * 70)

ventanas = pd.read_csv(ARCHIVO_VENTANAS)
cols_cruce = ['tramo_id', 'ventana_id', 'etiqueta_ventana', 'etiqueta_tfidf',
              'total_alarmas_ventana', 'total_eventos_ventana',
              'n_reglas_activas', 'banda_severidad', 'concentracion', 'persistencia']
cols_cruce = [c for c in cols_cruce if c in ventanas.columns]

m = df_err.merge(ventanas[cols_cruce], on=['tramo_id', 'ventana_id'], how='inner')
print(f'Ventanas evaluadas: {len(df_err):,} | cruzadas con éxito: {len(m):,}')

# --- La separación, por conjunto -------------------------------------------
if 'n_reglas_activas' in m.columns:
    print('\n' + '=' * 70)
    print('SEPARACIÓN NORMAL vs ANÓMALA, POR CONJUNTO')
    print('=' * 70)
    _es_norm = (m.n_reglas_activas == 0).to_numpy()
    _err = m.error_reconstruccion.to_numpy()
    print(f'{"conjunto":<10}{"n_norm":>8}{"n_anom":>8}{"err_norm":>11}'
          f'{"err_anom":>11}{"razón":>9}{"AUC":>9}')
    for _c in ('val', 'test', 'val+test'):
        _sel = (m.conjunto == _c).to_numpy() if _c != 'val+test' else np.ones(len(m), bool)
        _n, _a = _sel & _es_norm, _sel & ~_es_norm
        if not _n.any() or not _a.any():
            continue
        _en, _ea = float(np.median(_err[_n])), float(np.median(_err[_a]))
        _r = pd.Series(_err[_sel]).rank().to_numpy()
        _mn = _es_norm[_sel]
        _np_, _nn = int((~_mn).sum()), int(_mn.sum())
        _auc = (_r[~_mn].sum() - _np_ * (_np_ + 1) / 2) / (_np_ * _nn)
        _marca = '   <- la que se defiende' if _c == 'test' else (
                 '   <- aquí se eligió el modelo' if _c == 'val' else '')
        print(f'{_c:<10}{int(_n.sum()):>8,}{int(_a.sum()):>8,}{_en:>11.5f}'
              f'{_ea:>11.5f}{_ea/_en:>8.2f}x{_auc:>9.4f}{_marca}')
    print('\nLa brecha entre val y test mide cuánto sesgó la selección del')
    print('checkpoint. Si el checkpoint por separación cae mucho más que el de')
    print('pérdida, eso confirma que elegir maximizando la razón en validación')
    print('sobreajusta esa métrica, y es un resultado para declarar.')

if len(m) == 0:
    raise SystemExit('\n0 coincidencias. Casi seguro es la re-numeración de tramo_id. '
                     'Revisa que el paso 02 y este usen el mismo criterio.')
if len(m) < len(df_err) * 0.99:
    print('⚠ Se perdieron filas en el merge. Revisa los tramo_id / ventana_id.')

m['es_normal'] = m['etiqueta_ventana'] == 'normal'


# =============================================================================
# 4b. ERROR POR VARIABLE: QUÉ VARIABLES EXPLICAN QUE LA VENTANA SEA ANÓMALA
# =============================================================================
print('\n' + '=' * 70)
print('4b. ERROR POR VARIABLE')
print('=' * 70)
_sin_alarma = (m.n_reglas_activas == 0) if 'n_reglas_activas' in m.columns else m.es_normal
_quietas_val = ((m.conjunto == 'val') & _sin_alarma).to_numpy()
U_VAR = m.loc[_quietas_val, EVAR].quantile(0.95).to_numpy()      # umbral P95 por variable
_ev = m[EVAR].to_numpy()
_orden = np.argsort(-_ev, axis=1)
for _i in range(3):
    m[f'var_{_i + 1}'] = np.array(VARIABLES)[_orden[:, _i]]
m['aporte_var_1_pct'] = (100 * _ev[np.arange(len(m)), _orden[:, 0]] / _ev.sum(axis=1)).round(1)
m['variables_sobre_umbral'] = [', '.join(np.array(VARIABLES)[s]) for s in (_ev > U_VAR)]
pd.DataFrame({'variable': VARIABLES, 'umbral_P95': U_VAR}).to_csv(
    f'{RUTA_EQ}/04_umbral_por_variable.csv', sep=';', decimal=',', index=False, encoding='utf-8-sig')
print('Umbral por variable (P95 de las ventanas sin alarma de validación): 04_umbral_por_variable.csv')

# ¿la variable con más error es la de la regla que sonó? (contra el azar)
FAMILIAS = {
    'lubricacion': ['D', 'A5', 'E'],
    'corriente':   ['A', 'A1', 'A4'],
    'atollo':      ['B', 'A3'],
    'filtro':      ['C'],
    'temperatura': ['F', 'A7', 'A6', 'A10', 'A9', 'A8', 'I', 'A11', 'A13', 'A12',
                    'K', 'A15', 'A14', 'A16', 'L', 'A21', 'A22'],
    'vibracion':   ['O', 'A20', 'A19', 'A18', 'A17'],
}


def _variables_de(regla):
    r = POR_ID[regla]
    return [r[c] for c in ('var', 'var2', 'a', 'b') if c in r] + list(r.get('vars', []))


_rc = [r for r in IDS_REGLAS if r in ventanas.columns]
_act = (m[['tramo_id', 'ventana_id']]
        .merge(ventanas[['tramo_id', 'ventana_id'] + _rc], on=['tramo_id', 'ventana_id'], how='left')
        [_rc].fillna(0).to_numpy() > 0)
_r2v = np.array([[v in _variables_de(r) for v in VARIABLES] for r in _rc], dtype=int)
_vars_ok = (_act.astype(int) @ _r2v) > 0          # variables que miran las reglas que sonaron
_top = _orden[:, 0]
_dist0 = np.bincount(_top[_quietas_val], minlength=len(VARIABLES)) / max(int(_quietas_val.sum()), 1)
_fam = {f: (_act[:, [_rc.index(r) for r in rs if r in _rc]].any(axis=1)
            if any(r in _rc for r in rs) else np.zeros(len(m), bool)) for f, rs in FAMILIAS.items()}
_filas_at = []
for _cj in ('val', 'test'):
    _base = (m.conjunto == _cj).to_numpy() & _act.any(axis=1)
    _grupos = {'todas las anomalas': _base}
    for _f in FAMILIAS:
        _otras = np.column_stack([_fam[g] for g in FAMILIAS if g != _f]).any(axis=1)
        _grupos[f'solo {_f}'] = _base & _fam[_f] & ~_otras
    for _g, _s in _grupos.items():
        if _s.sum() == 0:
            continue
        _ok = _vars_ok[_s]
        _filas_at.append({'conjunto': _cj, 'grupo': _g, 'ventanas': int(_s.sum()),
                          'acierta_pct': round(100 * _ok[np.arange(_s.sum()), _top[_s]].mean(), 1),
                          'azar_pct': round(100 * (_ok * _dist0).sum(axis=1).mean(), 1)})
atribucion = pd.DataFrame(_filas_at)
print(atribucion.to_string(index=False))
atribucion.to_csv(f'{RUTA_EQ}/04_atribucion_variables.csv', sep=';', decimal=',', index=False,
                  encoding='utf-8-sig')
print('acierta_pct: la variable con más error es una de las que mira la regla que sonó.')
print('azar_pct: lo que acertaría si esa variable se repartiera como en las ventanas sin alarma.')


# =============================================================================
# 4. CRITERIO NORMAL VS. ANÓMALO
# =============================================================================
print('\n' + '=' * 70)
print('4. CRITERIO DEL PROFESOR: ¿el error es mayor en las ventanas malas?')
print('=' * 70)

resumen = m.groupby('es_normal')['error_reconstruccion'].agg(['count', 'mean', 'median', 'std'])
resumen.index = ['Anómala', 'Normal']
print(resumen.round(5).to_string())

err_normal = m.loc[m.es_normal, 'error_reconstruccion']
err_anomal = m.loc[~m.es_normal, 'error_reconstruccion']
razon_media = err_anomal.mean() / err_normal.mean()
razon_mediana = err_anomal.median() / err_normal.median()

print(f'\nRazón anómalo/normal en PROMEDIOS: {razon_media:.2f}x   <- referencia: 3,43x')
print(f'Razón anómalo/normal en MEDIANAS : {razon_mediana:.2f}x   <- referencia: 3,46x')
print('\nQue las dos razones sean parecidas es lo importante: significa que la')
print('separación NO la producen unos pocos casos extremos, sino que toda la')
print('distribución está corrida.')

print(f'\nVarianza recuperada — normales: {(1 - err_normal.mean()) * 100:.1f} % | '
      f'anómalas: {(1 - err_anomal.mean()) * 100:.1f} %')

# El umbral del percentil 90
p90 = err_normal.quantile(0.90)
capturadas = (err_anomal > p90).mean() * 100
falsas = (err_normal > p90).mean() * 100
print(f'\nUmbral en el percentil 90 de las normales = {p90:.5f}')
print(f'  captura el {capturadas:.1f} % de las ventanas anómalas')
print(f'  con {falsas:.1f} % de falsas alarmas')

corr = m['error_reconstruccion'].corr(m['total_alarmas_ventana'])
print(f'\nCorrelación error vs. total de alarmas: {corr:.3f}')
print('  (baja es lo esperado: el modelo detecta rareza, no cantidad)')


# =============================================================================
# 5. VALIDACIÓN DE LAS BANDAS DE SEVERIDAD
# =============================================================================
if 'banda_severidad' in m.columns:
    print('\n' + '=' * 70)
    print('5. VALIDACIÓN DE LAS BANDAS DE SEVERIDAD')
    print('=' * 70)

    g = m.groupby('banda_severidad').agg(
        ventanas=('error_reconstruccion', 'size'),
        err_mediano=('error_reconstruccion', 'median'),
        err_medio=('error_reconstruccion', 'mean'),
        alarmas_prom=('total_alarmas_ventana', 'mean'),
    ).round(5).sort_index()
    print(g.to_string())

    v = g.err_mediano.values
    monotono = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
    print(f'\n¿El error crece de forma monótona entre bandas? -> '
          f'{"SÍ ✓" if monotono else "NO ✗"}')
    if monotono:
        salto = g.err_medio.iloc[-1] / g.err_medio.iloc[0]
        print(f'De la banda normal a la grave el error promedio se multiplica por {salto:.1f}x')
        print('Un modelo sin etiquetas confirma el criterio de severidad por su cuenta.')

    # --- comparación con los índices alternativos ------------------------
    print('\n--- ¿Y los otros índices logran el mismo orden? ---')
    anom = m[m.total_alarmas_ventana > 0]
    candidatos = {}
    c2 = np.percentile(anom.total_alarmas_ventana, [33.3, 66.6])
    candidatos['total_alarmas'] = pd.cut(
        m.total_alarmas_ventana, [-1, 0, c2[0], c2[1], 1e9],
        labels=['0 normal', 'bajo', 'medio', 'alto'])
    if 'total_eventos_ventana' in m.columns:
        c3 = np.percentile(anom.total_eventos_ventana, [33.3, 66.6])
        candidatos['total_eventos'] = pd.cut(
            m.total_eventos_ventana, [-1, 0, max(c3[0], 0.5), max(c3[1], 1.5), 1e9],
            labels=['0 normal', 'bajo', 'medio', 'alto'], duplicates='drop')

    for nombre, serie in candidatos.items():
        med = m.groupby(serie, observed=True).error_reconstruccion.median()
        vv = med.values
        ok = all(vv[i] <= vv[i + 1] for i in range(len(vv) - 1))
        print(f'  {nombre:16s}: {" -> ".join(f"{x:.4f}" for x in vv)}   '
              f'{"ordena ✓" if ok else "NO ordena ✗"}')


m.to_csv(ARCHIVO_SALIDA, index=False)
print(f'\nGuardado: {ARCHIVO_SALIDA}')

plt.figure(figsize=(6, 5))
plt.boxplot([err_normal, err_anomal], tick_labels=['Normal', 'Anómala'], showfliers=False)
plt.ylabel('Error de reconstrucción (MSE)')
plt.title(f'Separación normal vs. anómala — W={W}  ({razon_media:.2f}x)')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{RUTA_EQ}/04_separacion_normal_anomalo.png', dpi=150)
plt.show()

if 'banda_severidad' in m.columns:
    orden = sorted(m.banda_severidad.unique())
    datos = [m.loc[m.banda_severidad == b, 'error_reconstruccion'] for b in orden]
    plt.figure(figsize=(8, 5))
    plt.boxplot(datos, tick_labels=orden, showfliers=False)
    plt.ylabel('Error de reconstrucción (MSE)')
    plt.xlabel('Banda de severidad (nº de reglas distintas activas)')
    plt.title('Validación de las bandas — el autoencoder nunca vio estas etiquetas')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{RUTA_EQ}/04_validacion_bandas.png', dpi=150)
    plt.show()

print('\n' + '=' * 70)
print('LISTO. Siguiente: 05a_nivel_N.py y 06_anticipacion.py')
print('=' * 70)
