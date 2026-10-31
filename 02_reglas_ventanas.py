# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 02 — REGLAS, VENTANAS Y ETIQUETADO   (W = 60, o sea 10 minutos)
=============================================================================
Entrada : CR010_limpio.csv                 (paso 01)
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
    RUTA_EQ = f'{RUTA_BASE}/CR010'
_os.makedirs(RUTA_EQ, exist_ok=True)
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import sys
sys.path.append(RUTA_BASE)

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from reglas_engine import evaluar_reglas

SEED = 42
np.random.seed(SEED)

W = 60
PASO_S = 10

ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
ARCHIVO_EVENTOS_REGLA = f'{RUTA_EQ}/{EQUIPO}_eventos_por_regla.csv'
ARCHIVO_WASSERSTEIN = f'{RUTA_EQ}/{EQUIPO}_wasserstein_bandas_W{W}.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']


# =============================================================================
# 0. CARGA
# =============================================================================
print('=' * 70)
print(f'PASO 02 — VENTANAS DE {W * 10 / 60:.0f} MINUTOS (W = {W})')
print('=' * 70)

def cargar_limpio(ruta):
    """Acepta el CSV con la columna de tiempo llamada 'timestamp' o 'Time'."""
    cabecera = pd.read_csv(ruta, nrows=0)
    col_tiempo = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col_tiempo], low_memory=False)
    return df.set_index(col_tiempo).sort_index().rename_axis('timestamp')


df = cargar_limpio(ARCHIVO_LIMPIO)
print(f'Dataset limpio: {df.shape[0]:,} filas × {df.shape[1]} columnas')

VARIABLES = [v for v in VARIABLES if v in df.columns]
print(f'Variables disponibles ({len(VARIABLES)}): {VARIABLES}')


# =============================================================================
# 1. SEGMENTAR EN TRAMOS CONTINUOS
# =============================================================================
print('\n' + '=' * 70)
print('1. SEGMENTANDO EN TRAMOS CONTINUOS')
print('=' * 70)


def segmentar_tramos_continuos(df, freq='10s', max_gap=1):
    idx = df.index.to_series()
    gaps = idx.diff() > pd.Timedelta(freq) * max_gap
    tramo_id = gaps.cumsum()
    df = df.copy()
    df['tramo_id'] = tramo_id
    return [g.drop(columns='tramo_id') for _, g in df.groupby('tramo_id')]


tramos = segmentar_tramos_continuos(df)
print(f'Tramos encontrados: {len(tramos):,}')

tramos = [t for t in tramos if len(t) >= W]
print(f'Tramos útiles (>= {W} registros): {len(tramos):,}   <- esperado: 1.929')


# =============================================================================
# UN SOLO CORTE DE ENTRENAMIENTO PARA TODO EL PIPELINE
# =============================================================================
import json as _json

ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'

_corte01 = _json.load(open(ARCHIVO_CORTE))
T_TRAIN = pd.Timestamp(_corte01['T_train'])
T_VAL = pd.Timestamp(_corte01['T_val'])
FRAC_TRAIN = _corte01.get('frac_train', 0.70)
FRAC_VAL = _corte01.get('frac_val', 0.15)

_largos = np.array([len(t) for t in tramos])
_ini_tr = np.array([t.index[0] for t in tramos])
_i_tr = int(np.flatnonzero(_ini_tr <= T_TRAIN)[-1])
_i_va = int(np.flatnonzero(_ini_tr <= T_VAL)[-1])

_n_tr = int(_largos[:_i_tr + 1].sum())
_n_va = int(_largos[_i_tr + 1:_i_va + 1].sum())
_n_te = int(_largos[_i_va + 1:].sum())
_n = int(_largos.sum())
print(f'\nCORTE UNICO (por muestras, en borde de tramo):')
print(f'  T_train = {T_TRAIN}   (fin del tramo {_i_tr})')
print(f'  T_val   = {T_VAL}   (fin del tramo {_i_va})')
print(f'  muestras: train {_n_tr:,} ({100*_n_tr/_n:.1f} %) | '
      f'val {_n_va:,} ({100*_n_va/_n:.1f} %) | '
      f'test {_n_te:,} ({100*_n_te/_n:.1f} %)')
print(f'  tramos:   train {_i_tr+1:,} | val {_i_va-_i_tr:,} | '
      f'test {len(tramos)-_i_va-1:,}')
print('  Si los porcentajes de MUESTRAS dan cerca de 70/15/15 pero los de')
print('  TRAMOS no, eso es exactamente lo que el corte viejo confundia.')

_corte01.update({'muestras_train': _n_tr, 'muestras_val': _n_va,
                 'muestras_test': _n_te,
                 'frac_train_efectiva': round(_n_tr / _n, 4),
                 'frac_val_efectiva': round(_n_va / _n, 4)})
_json.dump(_corte01, open(ARCHIVO_CORTE, 'w'), indent=2)
print(f'  Cifras efectivas agregadas a: {ARCHIVO_CORTE}')


def _inicio_de(t):
    """Primer timestamp de un tramo, venga suelto o como (id, tramo)."""
    return (t[1] if isinstance(t, tuple) else t).index[0]


def split_por_fecha(lista_tramos, t_train=None, t_val=None):
    """Parte una lista de tramos por los DOS instantes del corte unico."""
    t_train = T_TRAIN if t_train is None else t_train
    t_val = T_VAL if t_val is None else t_val
    tr = [t for t in lista_tramos if _inicio_de(t) <= t_train]
    va = [t for t in lista_tramos if t_train < _inicio_de(t) <= t_val]
    te = [t for t in lista_tramos if _inicio_de(t) > t_val]
    return tr, va, te


def split_df_por_fecha(d, col='timestamp_inicio', t_train=None, t_val=None):
    """Lo mismo para un DataFrame de ventanas, por su columna de tiempo."""
    t_train = T_TRAIN if t_train is None else t_train
    t_val = T_VAL if t_val is None else t_val
    ts = pd.to_datetime(d[col])
    return (d[ts <= t_train], d[(ts > t_train) & (ts <= t_val)], d[ts > t_val])


# =============================================================================
# 2. EVALUAR LAS REGLAS SOBRE CADA TRAMO
# =============================================================================
print('\n' + '=' * 70)
print('2. EVALUANDO REGLAS')
print('=' * 70)

tramos_con_reglas = []
ids_reglas = None
reglas_omitidas = None

for t in tramos:
    resultado = evaluar_reglas(t, verbose=False)
    if ids_reglas is None:
        ids_reglas = [c for c in resultado.columns
                      if c not in ('n_reglas_activadas', 'estado')]
        reglas_omitidas = resultado.attrs.get('reglas_omitidas', [])
    tramos_con_reglas.append(t.join(resultado))

print(f'Reglas evaluadas: {len(ids_reglas)}   <- esperado: 31')
print(f'Orden de las reglas: {ids_reglas}')
if reglas_omitidas:
    print(f'Reglas omitidas por falta de sensor: {reglas_omitidas}')
    print('  (A23 necesita el tag ZI266 / columna Conf, que no está en el dataset)')


print('\n' + '=' * 70)
print('2b. CONVIRTIENDO MUESTRAS A EVENTOS')
print('=' * 70)

registros_eventos = []
muestras_totales = 0

for tramo_idx, t in enumerate(tramos_con_reglas):
    n_ventanas_tramo = len(t) // W          # ventanas que sí se construyen
    limite = n_ventanas_tramo * W           # lo que sobra del tramo se ignora
    for r in ids_reglas:
        s = t[r].fillna(False).to_numpy().astype(bool)
        muestras_totales += int(s.sum())
        if not s.any():
            continue
        si = s.astype(np.int8)
        subidas = np.flatnonzero(np.diff(np.concatenate(([0], si))) == 1)
        bajadas = np.flatnonzero(np.diff(np.concatenate((si, [0]))) == -1)
        for ini, fin in zip(subidas, bajadas):
            if ini >= limite:               # evento en la cola descartada
                continue
            registros_eventos.append({
                'tramo_id': tramo_idx,
                'ventana_id': int(ini // W),   # la ventana donde ARRANCA
                'ventana_fin': int(min(fin, limite - 1) // W),  # donde TERMINA
                'regla': r,
                'paso_inicio': int(ini),
                'muestras': int(fin - ini + 1),
                'duracion_s': int((fin - ini + 1) * PASO_S),
                'borde_tramo': bool(ini == 0),
                'paso_en_ventana': int(ini % W),
            })

eventos = pd.DataFrame(registros_eventos)
print(f'Muestras en estado de alarma (lo que se contaba antes): {muestras_totales:,}')
print(f'Eventos de alarma (flancos de subida)                 : {len(eventos):,}')
if len(eventos):
    print(f'Razón muestras/eventos: {muestras_totales / len(eventos):.1f}')
    print(f'Eventos que arrancan en el borde de un tramo: '
          f'{int(eventos.borde_tramo.sum()):,} '
          f'({eventos.borde_tramo.mean() * 100:.1f} %)')

resumen_eventos = (eventos.groupby('regla')
                   .agg(eventos=('muestras', 'size'),
                        muestras_totales=('muestras', 'sum'),
                        dur_mediana_s=('duracion_s', 'median'),
                        dur_p95_s=('duracion_s', lambda x: x.quantile(0.95)),
                        dur_max_s=('duracion_s', 'max'))
                   .sort_values('muestras_totales', ascending=False))
resumen_eventos['muestras_por_evento'] = (
    resumen_eventos.muestras_totales / resumen_eventos.eventos).round(1)
resumen_eventos['pct_de_la_ventana'] = (
    100 * resumen_eventos.muestras_por_evento / W).round(1)

print('\n--- Muestras vs. eventos por regla ---')
print(resumen_eventos.to_string())

print('\n--- Distribución de la duración de los episodios (segundos) ---')
print(eventos.duracion_s.describe(
    percentiles=[.25, .5, .75, .9, .95, .99]).round(1).to_string())
print('\nMirá estos percentiles ANTES de fijar los umbrales de abajo: los '
      'valores por defecto son una elección mía, no de la literatura.')

UMBRAL_ESTADO_S = 180
UMBRAL_CHATTER_S = 60    # 1 min: episodio típico más corto -> transitoria

resumen_eventos['tipo'] = np.select(
    [resumen_eventos.dur_mediana_s >= UMBRAL_ESTADO_S,
     resumen_eventos.dur_mediana_s <= UMBRAL_CHATTER_S],
    ['estado', 'transitoria'],
    default='intermedia')

print(f'\n--- Poblaciones (umbrales: estado >= {UMBRAL_ESTADO_S} s, '
      f'transitoria <= {UMBRAL_CHATTER_S} s) ---')
print(resumen_eventos[['eventos', 'dur_mediana_s', 'muestras_por_evento',
                       'pct_de_la_ventana', 'tipo']].to_string())
print('\nConteo por tipo:')
print(resumen_eventos.tipo.value_counts().to_string())

resumen_eventos.to_csv(ARCHIVO_EVENTOS_REGLA, sep=';', decimal=',')
print(f'\nGuardado: {ARCHIVO_EVENTOS_REGLA}')

conteo_iniciados = (eventos.groupby(['tramo_id', 'ventana_id', 'regla'])
                    .size().unstack(fill_value=0)
                    .reindex(columns=ids_reglas, fill_value=0))

filas_activos = []
for tramo_id, v0, v1, regla in eventos[['tramo_id', 'ventana_id',
                                        'ventana_fin', 'regla']].itertuples(index=False):
    for v in range(v0, v1 + 1):
        filas_activos.append((tramo_id, v, regla))

conteo_activos = (pd.DataFrame(filas_activos,
                               columns=['tramo_id', 'ventana_id', 'regla'])
                  .groupby(['tramo_id', 'ventana_id', 'regla']).size()
                  .unstack(fill_value=0)
                  .reindex(columns=ids_reglas, fill_value=0))

print(f'\nVentanas donde ARRANCA al menos un evento : {conteo_iniciados.shape[0]:,}')
print(f'Ventanas CUBIERTAS por al menos un evento : {conteo_activos.shape[0]:,}')
print(f'  diferencia (ventanas en medio de un episodio largo): '
      f'{conteo_activos.shape[0] - conteo_iniciados.shape[0]:,}')
print(f'Episodios que abarcan más de una ventana  : '
      f'{int((eventos.ventana_fin > eventos.ventana_id).sum()):,} '
      f'({(eventos.ventana_fin > eventos.ventana_id).mean() * 100:.1f} %)')


# =============================================================================
# 3. CONTAR ACTIVACIONES POR VENTANA
# =============================================================================
print('\n' + '=' * 70)
print('3. CONSTRUYENDO VENTANAS')
print('=' * 70)


def construir_conteo_por_ventana(tramos_con_reglas, reglas_ids, W):
    resumenes = []
    for tramo_idx, t in enumerate(tramos_con_reglas):
        for v in range(len(t) // W):
            ventana = t.iloc[v * W:(v + 1) * W]
            conteo = ventana[reglas_ids].sum()
            fila = {
                'tramo_id': tramo_idx,
                'ventana_id': v,
                'timestamp_inicio': ventana.index[0],
                'timestamp_fin': ventana.index[-1],
            }
            fila.update(conteo.to_dict())
            fila['total_alarmas_ventana'] = int(conteo.sum())
            resumenes.append(fila)
    return pd.DataFrame(resumenes)


ventanas = construir_conteo_por_ventana(tramos_con_reglas, ids_reglas, W)
print(f'Ventanas construidas: {len(ventanas):,}   <- esperado: 19.615')


# =============================================================================
# 4. ETIQUETAR CADA VENTANA
# =============================================================================
print('\n' + '=' * 70)
print('4. ETIQUETANDO')
print('=' * 70)

CRITICIDAD_REGLAS = {}


def etiquetar_ventana(ventanas, reglas_ids, criterio='mayor_frecuencia', criticidad=None):
    df_out = ventanas.copy()
    criticidad = criticidad or CRITICIDAD_REGLAS

    def elegir(fila):
        conteos = fila[reglas_ids]
        if conteos.sum() == 0:
            return 'normal'
        maximo = conteos.max()
        candidatas = [r for r in reglas_ids if conteos[r] == maximo]
        if criterio == 'criticidad' and criticidad:
            con_crit = [c for c in candidatas if c in criticidad]
            if con_crit:
                return max(con_crit, key=lambda c: criticidad[c])
        return candidatas[0]

    df_out['etiqueta_ventana'] = df_out.apply(elegir, axis=1)
    return df_out


ventanas = etiquetar_ventana(ventanas, ids_reglas)

n_normal = int((ventanas.etiqueta_ventana == 'normal').sum())
n_anomala = len(ventanas) - n_normal
print(f'Normales: {n_normal:,} ({n_normal / len(ventanas) * 100:.1f} %)   <- esperado: 6.323')
print(f'Anómalas: {n_anomala:,} ({n_anomala / len(ventanas) * 100:.1f} %)   <- esperado: 13.292')
print('\nDistribución de etiquetas (top 10):')
print(ventanas.etiqueta_ventana.value_counts().head(10).to_string())

# --- Diagnóstico de la MODA ----------------------------------------------
print('\n--- Diagnóstico de la moda ---')
M_todas = ventanas[ids_reglas].to_numpy(float)
mask_anom = M_todas.sum(axis=1) > 0
M_anom = M_todas[mask_anom]

max_por_ventana = M_anom.max(axis=1)
n_con_el_maximo = (M_anom == max_por_ventana[:, None]).sum(axis=1)
dominancia = max_por_ventana / M_anom.sum(axis=1)

ventanas['moda_empate'] = False
ventanas['moda_dominancia'] = 0.0
ventanas.loc[mask_anom, 'moda_empate'] = n_con_el_maximo > 1
ventanas.loc[mask_anom, 'moda_dominancia'] = dominancia

print(f'  Moda única : {(n_con_el_maximo == 1).sum():,} ({(n_con_el_maximo == 1).mean() * 100:.1f} %)')
print(f'  Con EMPATE : {(n_con_el_maximo > 1).sum():,} ({(n_con_el_maximo > 1).mean() * 100:.1f} %)'
      '  <- el desempate lo decide el orden de columnas')
print(f'  Dominancia mediana de la moda: {np.median(dominancia) * 100:.1f} % de las alarmas')
print(f'  Ventanas donde la moda es < 50 % de las alarmas: '
      f'{(dominancia < 0.5).sum():,} ({(dominancia < 0.5).mean() * 100:.1f} %)')


print('\n' + '=' * 70)
print('4b. ETIQUETA POR TF·IDF SOBRE EVENTOS')
print('=' * 70)

# alinear las dos matrices de eventos con las ventanas del paso 3
idx_ventanas = pd.MultiIndex.from_frame(ventanas[['tramo_id', 'ventana_id']])
iniciados = conteo_iniciados.reindex(idx_ventanas, fill_value=0)
activos = conteo_activos.reindex(idx_ventanas, fill_value=0)
iniciados.index = ventanas.index
activos.index = ventanas.index

reglas_activas = [r for r in ids_reglas if activos[r].sum() > 0]
nunca = sorted(set(ids_reglas) - set(reglas_activas))
print(f'Reglas que se activan alguna vez: {len(reglas_activas)} de {len(ids_reglas)}')
if nunca:
    print(f'Reglas que NUNCA se activan ({len(nunca)}): {nunca}')
    print('  -> el espacio de características real tiene '
          f'{len(reglas_activas)} dimensiones, no {len(ids_reglas)}.')

N = len(ventanas)
idf = {r: float(np.log(N / (activos[r] > 0).sum())) for r in reglas_activas}
vec_idf = np.array([idf[r] for r in reglas_activas])[None, :]

print('\n--- Peso IDF por regla (mayor = más específica) ---')
print(pd.DataFrame({'ventanas_presente': [int((activos[r] > 0).sum()) for r in reglas_activas],
                    'pct_ventanas': [round(100 * (activos[r] > 0).mean(), 2) for r in reglas_activas],
                    'IDF': [round(idf[r], 3) for r in reglas_activas]},
                   index=reglas_activas).sort_values('IDF').to_string())


def tfidf(matriz):
    """TF x IDF fila por fila."""
    M = matriz[reglas_activas].to_numpy(float)
    tot = M.sum(axis=1, keepdims=True)
    TF = np.divide(M, tot, out=np.zeros_like(M), where=tot > 0)
    return TF * vec_idf, tot.ravel()


W_ini, tot_ini = tfidf(iniciados)
W_act, tot_act = tfidf(activos)

ventanas['total_eventos_ventana'] = tot_ini.astype(int)   # episodios que ARRANCAN
ventanas['total_activos_ventana'] = tot_act.astype(int)   # episodios que CUBREN

etq_ini = np.array(reglas_activas)[W_ini.argmax(axis=1)]
etq_act = np.array(reglas_activas)[W_act.argmax(axis=1)]
ventanas['etiqueta_tfidf'] = np.where(tot_ini > 0, etq_ini,
                             np.where(tot_act > 0, etq_act, 'normal'))
ventanas['origen_etiqueta'] = np.where(tot_ini > 0, 'inicia',
                               np.where(tot_act > 0, 'continua', 'sin_alarma'))

print('\n--- Origen de la etiqueta ---')
print(ventanas.origen_etiqueta.value_counts().to_string())

huerfanas = (ventanas.total_alarmas_ventana > 0) & (ventanas.total_eventos_ventana == 0)
sin_etiqueta = huerfanas & (ventanas.etiqueta_tfidf == 'normal')
print(f'\nVentanas con alarmas pero sin evento que arranque: {int(huerfanas.sum()):,} '
      f'({huerfanas.mean() * 100:.1f} %)')
print(f'  de ésas, RECUPERADAS por los eventos activos: '
      f'{int(huerfanas.sum() - sin_etiqueta.sum()):,}')
print(f'  de ésas, que quedan sin etiquetar            : {int(sin_etiqueta.sum()):,}')
if sin_etiqueta.sum() > 0:
    print('  ⚠ Si este número no es 0, hay episodios que empiezan fuera de las')
    print('    ventanas construidas (en la cola descartada del tramo anterior).')

print('\n--- Distribución de la etiqueta TF·IDF (top 10) ---')
print(ventanas.etiqueta_tfidf.value_counts().head(10).to_string())

con_alarma = ventanas.total_alarmas_ventana > 0
cambian = (ventanas.etiqueta_ventana != ventanas.etiqueta_tfidf) & con_alarma
print(f'\n--- Moda vs. TF·IDF, sobre las {int(con_alarma.sum()):,} ventanas con alarmas ---')
print(f'Cambian de etiqueta: {int(cambian.sum()):,} ({cambian.sum() / con_alarma.sum() * 100:.1f} %)')

print('\nMatriz de cambio (filas = moda, columnas = TF·IDF), top 8×8:')
top_moda = ventanas.loc[con_alarma, 'etiqueta_ventana'].value_counts().head(8).index
top_tfidf = ventanas.loc[con_alarma, 'etiqueta_tfidf'].value_counts().head(8).index
print(pd.crosstab(ventanas.loc[con_alarma, 'etiqueta_ventana'],
                  ventanas.loc[con_alarma, 'etiqueta_tfidf'])
      .reindex(index=top_moda, columns=top_tfidf, fill_value=0).to_string())

print('\n--- ¿Se rompió el sesgo por persistencia? ---')
print(f'{"regla":>5} {"presente":>9} {"gana moda":>10} {"gana TF·IDF":>12}')
for regla in resumen_eventos.index[:6]:
    presente = ventanas[regla] > 0
    if presente.sum() == 0:
        continue
    antes = (ventanas.loc[presente, 'etiqueta_ventana'] == regla).mean() * 100
    despues = (ventanas.loc[presente, 'etiqueta_tfidf'] == regla).mean() * 100
    print(f'{regla:>5} {int(presente.sum()):>9,} {antes:>9.1f} % {despues:>11.1f} %')

print(f'\n  Empates de la moda: {int(ventanas.moda_empate.sum()):,} '
      f'({ventanas.moda_empate.mean() * 100:.1f} %). '
      'Con TF·IDF continuo el argmax no empata.')


# =============================================================================
# 5. CONCENTRACIÓN — la métrica de severidad por ventana
# =============================================================================
print('\n' + '=' * 70)
print('5. CONCENTRACIÓN POR VENTANA')
print('=' * 70)


def calcular_concentracion(ventanas, reglas_ids):
    df_out = ventanas.copy()
    entropia_max = np.log(len(reglas_ids))

    def fila_c(fila):
        c = fila[reglas_ids].to_numpy(dtype=float)
        total = c.sum()
        if total == 0:
            return 0.0
        p = c[c > 0] / total
        entropia = -(p * np.log(p)).sum()
        return 1.0 - entropia / entropia_max

    df_out['concentracion'] = df_out.apply(fila_c, axis=1)
    return df_out


ventanas = calcular_concentracion(ventanas, ids_reglas)

anom = ventanas[ventanas.total_alarmas_ventana > 0]
print('Resumen sobre las ventanas anómalas:')
print(anom[['concentracion']].describe().round(4).to_string())

M = ventanas[ids_reglas].to_numpy(float)
tot = M.sum(axis=1)
act = tot > 0
pos_media = (M[act] * np.arange(len(ids_reglas))).sum(axis=1) / tot[act]
v = ventanas.loc[act, 'concentracion'].to_numpy()
print(f'\nCorrelación de concentracion con la POSICIÓN de las reglas: '
      f'{np.corrcoef(pos_media, v)[0, 1]:+.3f}')
print('  (la métrica anterior daba -0,907 acá: dependía del orden de la lista)')


# =============================================================================
# 5b. BANDAS DE SEVERIDAD — normal / leve / media / grave
# =============================================================================
print('\n' + '=' * 70)
print('5b. BANDAS DE SEVERIDAD')
print('=' * 70)

ventanas['n_reglas_activas'] = (ventanas[ids_reglas] > 0).sum(axis=1)
ventanas['persistencia'] = ventanas[ids_reglas].max(axis=1) / W


def asignar_banda(n):
    if n == 0:
        return '0_normal'
    if n == 1:
        return '1_leve'
    if n == 2:
        return '2_media'
    return '3_grave'


ventanas['banda_severidad'] = ventanas.n_reglas_activas.apply(asignar_banda)

print('Criterio: número de reglas DISTINTAS activas en la ventana\n')
resumen_bandas = ventanas.groupby('banda_severidad').agg(
    ventanas=('n_reglas_activas', 'size'),
    alarmas_promedio=('total_alarmas_ventana', 'mean'),
    persistencia_mediana=('persistencia', 'median'),
).round(2)
resumen_bandas['pct'] = (resumen_bandas.ventanas / len(ventanas) * 100).round(1)
print(resumen_bandas.to_string())
print('\nEsperado:  0_normal 6.323 | 1_leve 4.957 | 2_media 5.434 | 3_grave 2.901')

print('\nDistribución de reglas distintas por ventana anómala:')
print(ventanas.loc[ventanas.n_reglas_activas > 0, 'n_reglas_activas']
      .value_counts().sort_index().head(8).to_string())


print('\n' + '=' * 70)
print('5d. WASSERSTEIN POR BANDA (vs. dataset completo)')
print('=' * 70)


def calcular_wasserstein_por_banda(ventanas, columna_grupo='banda_severidad',
                                     columna_valor='total_alarmas_ventana'):
    distribucion_total = ventanas[columna_valor].to_numpy(dtype=float)
    filas = []
    for grupo in sorted(ventanas[columna_grupo].unique()):
        valores_grupo = ventanas.loc[ventanas[columna_grupo] == grupo, columna_valor].to_numpy(dtype=float)
        wemd = wasserstein_distance(valores_grupo, distribucion_total)
        filas.append({'banda_severidad': grupo, 'n_ventanas': len(valores_grupo),
                      'alarmas_promedio': valores_grupo.mean(), 'wasserstein_vs_total': wemd})
    return pd.DataFrame(filas)


wasserstein_bandas = calcular_wasserstein_por_banda(ventanas)
print(wasserstein_bandas.round(4).to_string(index=False))

media_global = ventanas['total_alarmas_ventana'].mean()
wasserstein_bandas['tipo'] = np.where(
    wasserstein_bandas['alarmas_promedio'] > media_global, 'anómala', 'no anómala')
print(f'\nPromedio global de alarmas por ventana: {media_global:.2f}')
print(wasserstein_bandas[['banda_severidad', 'alarmas_promedio', 'tipo',
                          'wasserstein_vs_total']].round(4).to_string(index=False))
for tipo, texto in [('anómala', 'la más GRAVE'), ('no anómala', 'la más ESTABLE')]:
    sub = wasserstein_bandas[wasserstein_bandas.tipo == tipo]
    if len(sub):
        top = sub.loc[sub.wasserstein_vs_total.idxmax(), 'banda_severidad']
        print(f'  Entre las {tipo}s, {texto} (mayor Wemd): {top}')

wasserstein_bandas.to_csv(ARCHIVO_WASSERSTEIN, index=False)
print(f'Guardado: {ARCHIVO_WASSERSTEIN}')

# --- El mismo cálculo sobre EVENTOS, para comparar -----------------------
print('\n--- Wemd por banda calculado sobre EVENTOS en vez de muestras ---')
wass_eventos = calcular_wasserstein_por_banda(
    ventanas, columna_valor='total_eventos_ventana')
media_global_ev = ventanas['total_eventos_ventana'].mean()
wass_eventos['tipo'] = np.where(
    wass_eventos['alarmas_promedio'] > media_global_ev, 'anómala', 'no anómala')
print(f'Promedio global de eventos por ventana: {media_global_ev:.2f}')
print(wass_eventos.round(4).to_string(index=False))
wass_eventos.to_csv(ARCHIVO_WASSERSTEIN.replace('.csv', '_eventos.csv'), index=False)


# =============================================================================
# 5c. DIAGNÓSTICO DEL CRITERIO DEL QUINTIL
# =============================================================================
print('\n' + '=' * 70)
print('5c. ¿QUÉ PASA SI APLICAMOS EL CRITERIO DEL PRIMER QUINTIL?')
print('=' * 70)

pct_sin_alarmas = (ventanas.total_alarmas_ventana == 0).mean() * 100
print(f'Ventanas que YA están en cero alarmas: {pct_sin_alarmas:.1f} %')
if pct_sin_alarmas > 20:
    print('  >> El primer quintil (20 %) cae completo DENTRO de lo que ya es')
    print('     normal, así que aplicado sobre todas las ventanas NO reclasifica nada.')

anomalas = ventanas[ventanas.total_alarmas_ventana > 0]
print('\nSi se aplica solo sobre las anómalas:')
for indice in ['concentracion', 'total_alarmas_ventana', 'total_eventos_ventana']:
    umbral = np.percentile(anomalas[indice], 20)
    bajo_umbral = anomalas[anomalas[indice] <= umbral]
    sobre_umbral = anomalas[anomalas[indice] > umbral]
    print(f'\n  {indice}:  umbral Q1 = {umbral:.2f}')
    print(f'    pasarían a normal: {len(bajo_umbral):,} ventanas, '
          f'con {bajo_umbral.total_alarmas_ventana.mean():.1f} alarmas promedio')
    print(f'    quedan anómalas  : {len(sobre_umbral):,} ventanas, '
          f'con {sobre_umbral.total_alarmas_ventana.mean():.1f} alarmas promedio')
    if bajo_umbral.total_alarmas_ventana.mean() > sobre_umbral.total_alarmas_ventana.mean():
        print('    ⚠ INVERTIDO: las que pasarían a normal tienen MÁS alarmas que el resto.')


# =============================================================================
# 5e. ÍNDICE W POR SENSOR, CORREGIDO POR MULTIPLICIDAD
# =============================================================================
print('\n' + '=' * 70)
print('5e. ÍNDICE W POR SENSOR')
print('=' * 70)

import sys as _sys
if RUTA_BASE not in _sys.path:
    _sys.path.append(RUTA_BASE)
from indice_w import IndiceW

ARCHIVO_W_SENSORES = f'{RUTA_EQ}/{EQUIPO}_w_sensores_W{W}.csv'

_X = np.stack([t[VARIABLES].to_numpy(dtype=np.float32)[v * W:(v + 1) * W]
               for t in tramos_con_reglas
               for v in range(len(t) // W)])
print(f'Ventanas para el índice W: {len(_X):,} '
      f'(deben ser las mismas {len(ventanas):,} del bloque 3)')
assert len(_X) == len(ventanas), 'el recorrido de ventanas no coincide'

_es_train = (pd.to_datetime(ventanas.timestamp_inicio) <= T_TRAIN).to_numpy()
_es_normal = (ventanas.n_reglas_activas == 0).to_numpy()
_fit = _es_train & _es_normal
print(f'Ventanas normales de entrenamiento para ajustar: {int(_fit.sum()):,}')
if _fit.sum() < 200:
    print('  AVISO: con tan pocas, los cuantiles de la nula van a ser ruidosos.')

_mu = _X[_es_train].reshape(-1, len(VARIABLES)).mean(axis=0)
_sd = _X[_es_train].reshape(-1, len(VARIABLES)).std(axis=0)
_sd = np.where(_sd > 0, _sd, 1.0)
_Xn = (_X - _mu) / _sd

_iw = IndiceW(VARIABLES, w=W).ajustar(_Xn[_fit])
_idx_w = _iw.transformar(_Xn)
_perfil = _iw.resumen_por_ventana(_idx_w)

for _c in _perfil.columns:
    ventanas[_c] = _perfil[_c].to_numpy()

_perfil.insert(0, 'ventana_id', ventanas.ventana_id.to_numpy())
_perfil.insert(0, 'tramo_id', ventanas.tramo_id.to_numpy())
_perfil.insert(2, 'timestamp_inicio', ventanas.timestamp_inicio.to_numpy())
_perfil.to_csv(ARCHIVO_W_SENSORES, index=False)
print(f'\nGuardado: {ARCHIVO_W_SENSORES}')

print('\n--- Tres escalas de severidad sobre las mismas ventanas ---')
print(pd.crosstab(ventanas.banda_w, ventanas.banda_severidad).to_string())
print('\n  n_reglas_activas -> cuántos subsistemas están comprometidos')
print('  criticidad_max   -> juicio experto (paso 02c)')
print('  banda_w          -> empírica, independiente de la tabla de criticidad')
print('El grado de acuerdo entre las tres es un RESULTADO, no un paso')
print('intermedio: es lo que compensa que la criticidad sea elicitación de un')
print('solo experto, no conversada con los operadores.')
print('\nY el sensor que alcanza el máximo (columna sensor_origen) contesta')
print('directamente lo que pidió el profesor: "faltaría determinar cuál es el')
print('que está originando la condición".')

del _X, _Xn


# =============================================================================
# 6. FEATURES DE SENSORES POR VENTANA
# =============================================================================
print('\n' + '=' * 70)
print('6. FEATURES DE SENSORES')
print('=' * 70)


def construir_features_sensores(tramos_con_reglas, variables, W):
    resumenes = []
    for tramo_idx, t in enumerate(tramos_con_reglas):
        for v in range(len(t) // W):
            ventana = t.iloc[v * W:(v + 1) * W]
            fila = {'tramo_id': tramo_idx, 'ventana_id': v}
            for var in variables:
                serie = ventana[var]
                fila[f'{var}_media'] = serie.mean()
                fila[f'{var}_std'] = serie.std()
                fila[f'{var}_min'] = serie.min()
                fila[f'{var}_max'] = serie.max()
                fila[f'{var}_delta'] = serie.iloc[-1] - serie.iloc[0]
            resumenes.append(fila)
    return pd.DataFrame(resumenes)


features = construir_features_sensores(tramos_con_reglas, VARIABLES, W)
n_features = len([c for c in features.columns if c not in ('tramo_id', 'ventana_id')])
print(f'Features construidos: {n_features}   <- esperado: 70 (14 variables × 5)')

dataset_features = features.merge(
    ventanas[['tramo_id', 'ventana_id', 'timestamp_inicio', 'timestamp_fin',
              'etiqueta_ventana', 'etiqueta_tfidf', 'total_alarmas_ventana',
              'total_eventos_ventana', 'n_reglas_activas',
              'banda_severidad', 'concentracion']],
    on=['tramo_id', 'ventana_id'], how='inner',
)
print(f'Filas tras el merge: {len(dataset_features):,}  '
      f'(debe coincidir con las {len(ventanas):,} ventanas)')

if len(dataset_features) != len(ventanas):
    print('⚠ ALERTA: el merge perdió filas. Revisa que tramo_id/ventana_id calcen.')


# =============================================================================
# 7. GUARDAR
# =============================================================================
print('\n' + '=' * 70)
print('7. GUARDANDO')
print('=' * 70)

ventanas.to_csv(ARCHIVO_VENTANAS, index=False)
print(f'{ARCHIVO_VENTANAS}')
print(f'   {ventanas.shape[0]:,} filas × {ventanas.shape[1]} columnas')

dataset_features.to_csv(ARCHIVO_FEATURES, index=False)
print(f'{ARCHIVO_FEATURES}')
print(f'   {dataset_features.shape[0]:,} filas × {dataset_features.shape[1]} columnas')

ARCHIVO_EVENTOS_VENTANA = f'{RUTA_EQ}/{EQUIPO}_eventos_por_ventana_W{W}.csv'
_ev_ventana = iniciados.copy()
_ev_ventana.insert(0, 'ventana_id', ventanas['ventana_id'].to_numpy())
_ev_ventana.insert(0, 'tramo_id', ventanas['tramo_id'].to_numpy())
_ev_ventana.to_csv(ARCHIVO_EVENTOS_VENTANA, index=False)
print(f'{ARCHIVO_EVENTOS_VENTANA}')
print(f'   {_ev_ventana.shape[0]:,} filas × {_ev_ventana.shape[1]} columnas')

ARCHIVO_ACTIVOS_VENTANA = f'{RUTA_EQ}/{EQUIPO}_activos_por_ventana_W{W}.csv'
_act_ventana = activos.copy()
_act_ventana.insert(0, 'ventana_id', ventanas['ventana_id'].to_numpy())
_act_ventana.insert(0, 'tramo_id', ventanas['tramo_id'].to_numpy())
_act_ventana.to_csv(ARCHIVO_ACTIVOS_VENTANA, index=False)
print(f'{ARCHIVO_ACTIVOS_VENTANA}')
print(f'   {_act_ventana.shape[0]:,} filas × {_act_ventana.shape[1]} columnas')

print('\n--- Primeras filas de las ventanas ---')
cols_muestra = ['tramo_id', 'ventana_id', 'total_alarmas_ventana', 'total_eventos_ventana',
                'n_reglas_activas', 'banda_severidad', 'etiqueta_ventana',
                'etiqueta_tfidf', 'concentracion']
print(ventanas[cols_muestra].head(10).to_string(index=False))

print('\n--- Comparación de las dos etiquetas ---')
cmp_etq = pd.DataFrame({
    'por moda (como antes)': ventanas.etiqueta_ventana.value_counts(),
    'por TF·IDF (papers)': ventanas.etiqueta_tfidf.value_counts(),
}).fillna(0).astype(int).sort_values('por moda (como antes)', ascending=False).head(8)
print(cmp_etq.to_string())
print('\nLa etiqueta oficial sigue siendo la moda. La de TF·IDF va al lado')
print('para poder mostrar cuánto cambia el diagnóstico al ponderar por')
print('especificidad en vez de por persistencia.')

print('\n' + '=' * 70)
print('LISTO. Siguiente: 03a_autoencoder_eleccion_k.py o 03b_autoencoder_entrenamiento.py')
print('=' * 70)
