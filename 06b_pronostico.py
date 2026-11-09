# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 06b — PRONÓSTICO: ¿VIENE UNA VENTANA GRAVE EN LA PRÓXIMA HORA?
=============================================================================
Con lo que se sabe al cierre de cada ventana (nivel de las reglas y su historia
de la última hora, error del autoencoder y z de las medias) se estima la
probabilidad de que haya una ventana grave en las próximas H ventanas del mismo
tramo. Regresión logística ajustada en la primera mitad de validación; el modelo
y el umbral de aviso se eligen en la segunda mitad; prueba solo se informa.
Línea base: persistencia (el nivel actual).
Entrada:  {EQUIPO}_error_reconstruccion_W60.csv (paso 04), {EQUIPO}_ventanas_W60.csv y
          {EQUIPO}_features_W60.csv (paso 02), 05a_nivel_ventanas.csv (paso 05a)
Salida:   06b_pronostico_metricas.csv, 06b_pronostico_comparaciones.csv, 06b_pronostico_aviso.csv,
          06b_pronostico_coeficientes.csv, 06b_pronostico_ventanas.csv, 06b_pronostico.png
"""
import os

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 0. CONSTANTES
# =============================================================================
W = 60
SEED = 42
PERCENTIL = 95
MIN_POR_VENTANA = 10
HORIZONTES = (1, 3, 6, 12)      # ventanas: 10 min, 30 min, 1 h y 2 h
H = 6                           # horizonte principal: la próxima hora
HISTORIA = 6                    # historia: la última hora
TOPE = 36                       # tope de los contadores de tiempo (6 h)
FPR_AVISO = 0.05                # avisos falsos tolerados, como el P95 del paso 04
N_BOOT = 1000
BLOQUE = 36                     # bloques de 6 h para el intervalo de confianza
MIN_CLASE = 20                  # ventanas mínimas por clase para ajustar o evaluar
C_REG = 0.1                     # regularización L2 fija (sin ajuste)
LLAVES = ['tramo_id', 'ventana_id']
BANDA = 'banda_N_empinada_natural'
NIVEL_NUM = {'0_normal': 0, '1_leve': 1, '2_media': 2, '3_grave': 3}
SUB = {'inicio': 'sin grave en la última hora', 'todas': 'todas las ventanas'}
COLORES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
TINTA, CRITICO = '#52514e', '#d03b3b'

PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
if PUNTO_PARADA not in ('perdida', 'separacion'):
    raise SystemExit(f"PUNTO_PARADA tiene que ser 'perdida' o 'separacion', no {PUNTO_PARADA!r}.")
AE = 'err_AE_normales_sep' if PUNTO_PARADA == 'separacion' else 'err_AE_normales'
ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
ARCHIVO_NIVEL = f'{RUTA_EQ}/05a_nivel_ventanas.csv'
for _f in [ARCHIVO_ERROR, ARCHIVO_VENTANAS, ARCHIVO_NIVEL]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste los pasos 02, 04 y 05a?')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


def guardar(df, nombre):
    df.to_csv(salida(nombre), sep=';', decimal=',', index=False, encoding='utf-8-sig')


def pasado(col, n, fun):
    """Resumen de las últimas n ventanas del tramo (incluye la actual)."""
    r = getattr(d.groupby('seg', sort=False)[col].rolling(n, min_periods=1), fun)()
    return r.reset_index(level=0, drop=True).reindex(d.index)


def desde_ultima(s):
    """Ventanas desde la última alarma del tramo, con tope."""
    out, ult = np.empty(len(s)), None
    for i, v in enumerate(s.to_numpy()):
        if v > 0:
            ult = i
        out[i] = TOPE if ult is None else min(i - ult, TOPE)
    return pd.Series(out, index=s.index)


def umbral_fpr(s_neg):
    """Umbral más bajo con a lo más FPR_AVISO de avisos falsos (sirve para puntajes discretos)."""
    t = np.inf
    for v in np.unique(s_neg)[::-1]:
        if (s_neg >= v).mean() > FPR_AVISO:
            break
        t = v
    return t


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def ap(y, s):
    return average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan


# =============================================================================
# 1. DATOS EN ORDEN DE TIEMPO
# =============================================================================
print('=' * 72)
print(f'PASO 06b — PRONÓSTICO DE VENTANAS GRAVES ({EQUIPO})')
print('=' * 72)
err = pd.read_csv(ARCHIVO_ERROR)
if AE not in err.columns or not np.allclose(err[AE], err.error_reconstruccion):
    raise SystemExit(f'El paso 04 se corrió con otro punto de parada: córrelo con PUNTO_PARADA={PUNTO_PARADA!r}.')
VARIABLES = [c[5:] for c in err.columns if c.startswith('evar_')]
EVAR = [f'evar_{v}' for v in VARIABLES]
d = err.loc[err.conjunto.isin(['val', 'test']),
            LLAVES + ['conjunto', 'error_reconstruccion', 'n_reglas_activas'] + EVAR]
d = d.merge(pd.read_csv(ARCHIVO_VENTANAS, usecols=LLAVES + ['timestamp_inicio'], parse_dates=['timestamp_inicio'])
            .drop_duplicates(LLAVES), on=LLAVES, how='left', validate='one_to_one')
if d.timestamp_inicio.isna().any():
    raise SystemExit('Hay ventanas del paso 04 sin fecha en el paso 02: ¿son de la misma corrida?')
_niv = pd.read_csv(ARCHIVO_NIVEL, nrows=0).columns
if BANDA not in _niv:
    raise SystemExit(f'El 05a_nivel_ventanas.csv no tiene la columna {BANDA}: vuelve a correr el 05a.')
d = d.merge(pd.read_csv(ARCHIVO_NIVEL, usecols=LLAVES + [BANDA]).drop_duplicates(LLAVES),
            on=LLAVES, how='left', validate='one_to_one')
_sin = int(d[BANDA].isna().sum())
if _sin:
    print(f'OJO: {_sin} ventanas sin nivel en el 05a; se toman como sin alarma.')
d['nivel'] = d[BANDA].map(NIVEL_NUM).fillna(0).astype(float)
MEDIAS = []
if os.path.exists(ARCHIVO_FEATURES):
    _m = [f'{v}_media' for v in VARIABLES]
    _f = pd.read_csv(ARCHIVO_FEATURES, usecols=lambda c: c in LLAVES + _m).drop_duplicates(LLAVES)
    MEDIAS = [c for c in _m if c in _f.columns]
    d = d.merge(_f, on=LLAVES, how='left', validate='one_to_one')

# tramos continuos dentro de cada conjunto
d = d.sort_values(LLAVES).reset_index(drop=True)
d['seg'] = ((d.tramo_id.diff() != 0) | (d.ventana_id.diff() != 1)
            | (d.conjunto != d.conjunto.shift())).cumsum()
d['alarma'] = (d.n_reglas_activas > 0).astype(float)
d['grave'] = (d.nivel == 3).astype(float)


# =============================================================================
# 2. LO QUE SE SABE AL CIERRE DE CADA VENTANA (solo mira hacia atrás)
# =============================================================================
# estado de las reglas
d['frac_alarma_1h'] = pasado('alarma', HISTORIA, 'mean')
d['nivel_max_1h'] = pasado('nivel', HISTORIA, 'max')
d['grave_1h'] = pasado('grave', HISTORIA, 'max')
d['log_desde_alarma'] = np.log1p(d.groupby('seg', sort=False).alarma.transform(desde_ultima))
d['log_minutos_tramo'] = np.log1p(np.minimum(d.ventana_id, TOPE) * MIN_POR_VENTANA)

# autoencoder
normal_val = (d.conjunto == 'val') & (d.alarma == 0)
if normal_val.sum() < MIN_CLASE:
    raise SystemExit('Hay muy pocas ventanas sin alarma en validación para fijar el nivel normal.')
_p95 = d.loc[normal_val, EVAR].quantile(PERCENTIL / 100).clip(lower=1e-12)
for v in VARIABLES:
    d[f'r_{v}'] = np.clip(np.log(np.maximum(d[f'evar_{v}'], 1e-12) / _p95[f'evar_{v}']), -5, 5)
d['log_error'] = np.log(np.maximum(d.error_reconstruccion, 1e-12))
d['log_error_max_30min'] = pasado('log_error', 3, 'max')
d['cambio_error_1h'] = d.log_error - pasado('log_error', HISTORIA, 'mean')

# z de las medias (línea base del paso 06)
if MEDIAS:
    _mu, _sd = d.loc[normal_val, MEDIAS].mean(), d.loc[normal_val, MEDIAS].std().replace(0, 1.0)
    _z = ((d[MEDIAS] - _mu) / _sd).abs()
    _faltan = int(_z.isna().any(axis=1).sum())
    if _faltan:
        print(f'OJO: {_faltan} ventanas sin medias en el paso 02; su z se toma como 0.')
    _z = _z.fillna(0)
    for c in MEDIAS:
        d[f'z_{c[:-6]}'] = np.log1p(_z[c])
    d['log_z_max'] = np.log1p(_z.max(axis=1))
    d['log_z_max_30min'] = pasado('log_z_max', 3, 'max')

GRUPOS = {'estado': ['nivel', 'alarma', 'frac_alarma_1h', 'nivel_max_1h', 'log_desde_alarma', 'log_minutos_tramo'],
          'autoencoder': ['log_error', 'log_error_max_30min', 'cambio_error_1h'] + [f'r_{v}' for v in VARIABLES]}
if MEDIAS:
    GRUPOS['z'] = ['log_z_max', 'log_z_max_30min'] + [f'z_{c[:-6]}' for c in MEDIAS]
MODELOS = {'persistencia (nivel actual)': None, 'estado': ['estado']}
if MEDIAS:
    MODELOS['z solo'] = ['z']
MODELOS['autoencoder solo'] = ['autoencoder']
if MEDIAS:
    MODELOS['estado + z'] = ['estado', 'z']
MODELOS['estado + autoencoder'] = ['estado', 'autoencoder']
if MEDIAS:
    MODELOS['estado + z + autoencoder'] = ['estado', 'z', 'autoencoder']

# objetivo: alguna ventana grave en las h siguientes del mismo tramo
_restantes = d.groupby('seg', sort=False).cumcount(ascending=False)
for h in HORIZONTES:
    fut = d.groupby('seg', sort=False).grave.transform(
        lambda s, h=h: s[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1))
    d[f'y_{h}'] = fut
    d[f'eval_{h}'] = fut.notna() & ((_restantes >= h) | (fut == 1))

# validación A (ajuste) y B (elección), con purga en el corte
_va = d.conjunto == 'val'
_corte = d.loc[_va, 'timestamp_inicio'].sort_values().iloc[int(_va.sum()) // 2]
d['parte'] = np.where(~_va, 'test', np.where(d.timestamp_inicio < _corte, 'val_a', 'val_b'))
_pos = d.groupby('seg', sort=False).cumcount()
_prim_b = pd.Series(np.where(d.parte == 'val_b', _pos, np.inf)).groupby(d.seg).transform('min')
d.loc[(d.parte == 'val_a') & (_pos >= _prim_b - max(HORIZONTES)), 'parte'] = 'purga'

_dur = d.groupby('seg').size() * MIN_POR_VENTANA
print(f'Ventanas: validación A {(d.parte == "val_a").sum():,} | validación B {(d.parte == "val_b").sum():,} '
      f'| prueba {(d.parte == "test").sum():,} | purgadas en el corte {(d.parte == "purga").sum():,}')
print(f'Corte de validación: {_corte:%Y-%m-%d %H:%M}')
print(f'Tramos continuos: {len(_dur):,} | duración mediana {_dur.median():.0f} min | '
      f'{100 * (_dur >= 120).mean():.0f} % duran 2 h o más')
_prev = d.groupby('seg', sort=False).grave_1h.shift(1)
d['inicio_grave'] = (d.grave == 1) & (_prev == 0)
for p in ('val_a', 'val_b', 'test'):
    s = d.parte == p
    print(f'  {p:<6} ventanas graves {100 * d.grave[s].mean():5.1f} % | inicios de grave '
          f'(sin grave en la hora previa): {int(d.inicio_grave[s].sum()):,}')
print(f'Detectores: estado de las reglas, autoencoder{", z de las medias" if MEDIAS else ""}')


# =============================================================================
# 3. AJUSTE EN VALIDACIÓN A Y EVALUACIÓN
# =============================================================================
print('\n' + '=' * 72)
print('3. ¿QUÉ TAN BIEN SE PRONOSTICA? (AUC: validación B / prueba; 0,5 = azar)')
print('=' * 72)
filas, PUNT, AJUSTE = [], {}, {}
_y_np = {h: d[f'y_{h}'].to_numpy() for h in HORIZONTES}
for h in HORIZONTES:
    y = _y_np[h]
    for sub in SUB:
        base = (d[f'eval_{h}'] & ((d.grave_1h == 0) if sub == 'inicio' else True)).to_numpy()
        fit = base & (d.parte == 'val_a').to_numpy()
        ya = y[fit].astype(int)
        if min(ya.sum(), len(ya) - ya.sum()) < MIN_CLASE:
            print(f'  {h * MIN_POR_VENTANA} min, {SUB[sub]}: pocas ventanas para ajustar '
                  f'({ya.sum()} con grave después, de {len(ya)}); se omite.')
            continue
        for nom, grupos in MODELOS.items():
            if grupos is None:
                s = d.nivel.to_numpy(dtype=float)
            else:
                cols = [c for g_ in grupos for c in GRUPOS[g_]]
                mod = make_pipeline(StandardScaler(), LogisticRegression(C=C_REG, max_iter=2000))
                mod.fit(d.loc[fit, cols].to_numpy(), ya)
                s = mod.predict_proba(d[cols].to_numpy())[:, 1]
                AJUSTE[(h, sub, nom)] = (mod, cols)
            PUNT[(h, sub, nom)] = s
            for parte in ('val_b', 'test'):
                sel = base & (d.parte == parte).to_numpy()
                yy = y[sel].astype(int)
                filas.append({'horizonte_min': h * MIN_POR_VENTANA, 'subconjunto': SUB[sub], 'modelo': nom,
                              'conjunto': parte, 'ventanas': int(sel.sum()), 'con_grave_despues': int(yy.sum()),
                              'pct_con_grave_despues': round(100 * yy.mean(), 1) if len(yy) else np.nan,
                              'suficiente': bool(min(yy.sum(), len(yy) - yy.sum()) >= MIN_CLASE),
                              'auc': auc(yy, s[sel]), 'ap': ap(yy, s[sel])})
met = pd.DataFrame(filas)
guardar(met.round(4), '06b_pronostico_metricas.csv')
if met.empty:
    raise SystemExit('No hay ventanas suficientes para ajustar ningún horizonte.')

_txt = lambda r: f'{r.auc:.2f}' if r.suficiente and pd.notna(r.auc) else '—'
met['_txt'] = met.apply(_txt, axis=1)
for sub in SUB:
    t = met[met.subconjunto == SUB[sub]]
    if t.empty:
        continue
    w = t.pivot_table(index=['modelo', 'horizonte_min'], columns='conjunto', values='_txt', aggfunc='first')
    w = (w['val_b'] + ' / ' + w['test']).unstack('horizonte_min').reindex([m for m in MODELOS if m in set(t.modelo)])
    w.columns = [f'{c} min' for c in w.columns]
    print(f'\n{SUB[sub].capitalize()} (validación B / prueba):')
    print(w.to_string())
    _b = t[(t.modelo == 'estado') & (t.conjunto == 'test')]
    print('Con grave después, en prueba: ' + ' | '.join(
        f'{r.horizonte_min} min {r.pct_con_grave_despues} %' for r in _b.itertuples()))
print('\npersistencia: el nivel actual como puntaje. estado: nivel, alarma, historia de la última hora y')
print('minutos desde el arranque del tramo. autoencoder: error total, su máximo de 30 min, su cambio')
print('en la hora y el error de cada variable sobre su P95. z: |z| de las medias de cada variable.')
print('"Sin grave en la última hora" es lo que interesa para avisar: deja fuera lo que ya está pasando.')

_cand = met[(met.horizonte_min == H * MIN_POR_VENTANA) & (met.subconjunto == SUB['inicio'])
            & (met.conjunto == 'val_b') & met.suficiente]
if _cand.empty or _cand.auc.isna().all():
    raise SystemExit('No hay ventanas suficientes para el horizonte principal (1 h): no se elige modelo.')
ELEGIDO = _cand.loc[_cand.auc.idxmax(), 'modelo']
print(f'\nElegido en validación B (próxima hora, {SUB["inicio"]}): {ELEGIDO} '
      f'(AUC {_cand.auc.max():.3f})')


# =============================================================================
# 4. ¿QUÉ AGREGA CADA FUENTE? (intervalo 95 % remuestreando bloques de 6 h)
# =============================================================================
print('\n' + '=' * 72)
print('4. ¿QUÉ AGREGA CADA FUENTE? (próxima hora, sin grave en la última hora)')
print('=' * 72)
rng = np.random.default_rng(SEED)
_bloque = (d.seg.astype(np.int64) * 100000 + _pos // BLOQUE).to_numpy()
_y = _y_np[H]
_base = (d[f'eval_{H}'] & (d.grave_1h == 0)).to_numpy()


def ic_dif(a, b, sel):
    idx = np.flatnonzero(sel)
    yy, a, b = _y[idx].astype(int), a[idx], b[idx]
    dif = auc(yy, a) - auc(yy, b)
    grupos = pd.Series(np.arange(len(idx))).groupby(_bloque[idx]).indices
    claves = np.array(list(grupos))
    vals = []
    for _ in range(N_BOOT):
        m_ = np.concatenate([grupos[c] for c in rng.choice(claves, len(claves))])
        if 0 < yy[m_].sum() < len(m_):
            vals.append(auc(yy[m_], a[m_]) - auc(yy[m_], b[m_]))
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) > N_BOOT // 2 else (np.nan, np.nan)
    return dif, lo, hi, len(claves)


_comp = [('estado + autoencoder', 'estado', '¿el autoencoder agrega a las reglas?'),
         ('estado + z', 'estado', '¿el z de las medias agrega a las reglas?'),
         ('estado + z + autoencoder', 'estado + z', '¿el autoencoder agrega al z?'),
         ('estado', 'persistencia (nivel actual)', '¿la historia agrega al nivel actual?'),
         (ELEGIDO, 'persistencia (nivel actual)', 'elegido contra persistencia')]
filas_c, _vistas = [], set()
for a_, b_, preg in _comp:
    if (a_, b_) in _vistas or a_ == b_ or (H, 'inicio', a_) not in PUNT or (H, 'inicio', b_) not in PUNT:
        continue
    _vistas.add((a_, b_))
    for parte in ('val_b', 'test'):
        sel = _base & (d.parte == parte).to_numpy()
        dif, lo, hi, nb = ic_dif(PUNT[(H, 'inicio', a_)], PUNT[(H, 'inicio', b_)], sel)
        filas_c.append({'pregunta': preg, 'comparacion': f'{a_} − {b_}', 'conjunto': parte,
                        'dif_auc': round(dif, 3), 'ic_inf': round(lo, 3), 'ic_sup': round(hi, 3), 'bloques': nb})
comp = pd.DataFrame(filas_c)
guardar(comp, '06b_pronostico_comparaciones.csv')
print(comp.to_string(index=False))
print('Si el intervalo no cruza 0, la diferencia es clara. Se decide con validación B; prueba se informa.')


# =============================================================================
# 5. AVISO CON 5 % DE AVISOS FALSOS (umbral fijado en validación B)
# =============================================================================
print('\n' + '=' * 72)
print(f'5. AVISO: UMBRAL CON {100 * FPR_AVISO:.0f} % DE AVISOS FALSOS EN VALIDACIÓN B (próxima hora)')
print('=' * 72)
_neg_b = _base & (d.parte == 'val_b').to_numpy() & (_y == 0)
_pos_t = _base & (d.parte == 'test').to_numpy() & (_y == 1)
_neg_t = _base & (d.parte == 'test').to_numpy() & (_y == 0)
_ini_t = np.flatnonzero((d.inicio_grave & (d.parte == 'test')).to_numpy())
_cens = int(((d.grave == 1) & _prev.isna() & (d.parte == 'test')).sum())
_dias = (d.parte == 'test').sum() * MIN_POR_VENTANA / 1440
_ref_t = _neg_t & (d.grave == 0).to_numpy() & (_pos.to_numpy() > 0)     # momentos tranquilos sin grave después
_pos_np = _pos.to_numpy()
filas_a, MARCA = [], {}
for nom in MODELOS:
    if (H, 'inicio', nom) not in PUNT:
        continue
    s = PUNT[(H, 'inicio', nom)]
    umbral = float(umbral_fpr(s[_neg_b]))
    marca = s >= umbral
    _sh = pd.Series(marca.astype(float)).groupby(d.seg).shift(1)
    _hora = (_sh.groupby(d.seg).rolling(HISTORIA, min_periods=1).max()
             .reset_index(level=0, drop=True).reindex(d.index).fillna(0).to_numpy() > 0)
    MARCA[nom] = (marca, umbral)
    adelantos = []
    for i in _ini_t:
        k = min(_pos_np[i], HISTORIA)
        previas = marca[i - k:i]
        if previas.any():
            adelantos.append((k - int(np.argmax(previas))) * MIN_POR_VENTANA)
    acierto = int(marca[_pos_t].sum())
    filas_a.append({'modelo': nom, 'umbral': round(umbral, 4),
                    'avisos_falsos_pct_val_b': round(100 * marca[_neg_b].mean(), 1),
                    'avisos_falsos_pct_test': round(100 * marca[_neg_t].mean(), 1),
                    'ventanas_aviso_falso_por_dia': round(marca[_neg_t].sum() / _dias, 1),
                    'sensibilidad_pct_test': round(100 * marca[_pos_t].mean(), 1),
                    'precision_pct_test': round(100 * acierto / max(acierto + marca[_neg_t].sum(), 1), 1),
                    'inicios_grave_test': len(_ini_t), 'inicios_censurados_test': _cens,
                    'inicios_avisados_pct': round(100 * len(adelantos) / max(len(_ini_t), 1), 1),
                    'avisados_por_azar_pct': round(100 * _hora[_ref_t].mean(), 1) if _ref_t.any() else np.nan,
                    'adelanto_mediano_min': float(np.median(adelantos)) if adelantos else np.nan})
aviso = pd.DataFrame(filas_a)
guardar(aviso, '06b_pronostico_aviso.csv')
print(aviso.drop(columns=['umbral', 'inicios_grave_test', 'inicios_censurados_test']).to_string(index=False))
print(f'Inicios de grave en prueba: {len(_ini_t):,} (sin grave en la hora previa) | además {_cens:,} empiezan en la '
      f'primera ventana de un tramo y nadie puede avisarlos.')
print('inicios_avisados_pct: inicios con al menos un aviso en la hora previa; avisados_por_azar_pct: lo mismo')
print('antes de momentos tranquilos que no terminan en grave. adelanto: minutos entre el primer aviso de esa')
print('hora y el inicio. ventanas_aviso_falso_por_dia: ventanas avisadas sin grave en la hora siguiente.')


# =============================================================================
# 6. ¿QUÉ PESA EN EL MODELO COMPLETO? (coeficientes estandarizados)
# =============================================================================
print('\n' + '=' * 72)
print('6. ¿QUÉ PESA EN EL MODELO COMPLETO? (próxima hora, coeficientes estandarizados)')
print('=' * 72)
_completo = next(m for m in ['estado + z + autoencoder', 'estado + autoencoder'] if (H, 'inicio', m) in AJUSTE)
_mod, _cols = AJUSTE[(H, 'inicio', _completo)]
coef = pd.DataFrame({'variable': _cols, 'coeficiente': _mod[-1].coef_[0]})
coef['grupo'] = [next(g for g, cs in GRUPOS.items() if c in cs) for c in _cols]
coef = coef.reindex(coef.coeficiente.abs().sort_values(ascending=False).index)
coef['modelo'] = _completo
guardar(coef.round(4), '06b_pronostico_coeficientes.csv')
print(f'Modelo: {_completo}. Positivo: sube la probabilidad de grave en la próxima hora.')
print(coef.head(10)[['grupo', 'variable', 'coeficiente']].round(3).to_string(index=False))


# =============================================================================
# 7. FIGURA
# =============================================================================
_fig_mod = ELEGIDO if ELEGIDO in MARCA and MODELOS[ELEGIDO] is not None else _completo
fig, ejes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={'width_ratios': [1, 2]})
a1 = ejes[0]
_lineas = [m for m in ['persistencia (nivel actual)', 'estado', 'estado + z', 'estado + autoencoder',
                       'estado + z + autoencoder'] if m in MODELOS]
for col, nom in zip(COLORES, _lineas):
    t = met[(met.modelo == nom) & (met.subconjunto == SUB['inicio']) & (met.conjunto == 'test') & met.suficiente]
    a1.plot(t.horizonte_min, t.auc, marker='o', ms=5, lw=2, color=col, label=nom)
a1.axhline(0.5, color=TINTA, lw=1, ls='--')
a1.text(HORIZONTES[-1] * MIN_POR_VENTANA, 0.505, 'azar', color=TINTA, fontsize=8, ha='right', va='bottom')
a1.set_xticks([h * MIN_POR_VENTANA for h in HORIZONTES])
a1.set_xlabel('Horizonte (min)')
a1.set_ylabel('AUC en prueba')
a1.set_title('¿Viene grave? (sin grave en la última hora)', fontsize=10)
a1.grid(alpha=0.25)
a1.legend(fontsize=8, frameon=False)

a2 = ejes[1]
_t = d[d.parte == 'test']
_n_ini = _t.groupby('seg').inicio_grave.sum()
if len(_n_ini) and _n_ini.max() > 0:
    _s = _n_ini.idxmax()
    _r = _t[_t.seg == _s]
    _p0 = max(int(np.argmax(_r.inicio_grave.to_numpy())) - BLOQUE, 0)
    _r = _r.iloc[_p0:_p0 + 3 * 144]
    _idx = _r.index.to_numpy()
    marca, umbral = MARCA[_fig_mod]
    prob = PUNT[(H, 'inicio', _fig_mod)][_idx]
    a2.fill_between(_r.timestamp_inicio, 0, 1, where=_r.grave.to_numpy() == 1, step='post', color=CRITICO,
                    alpha=0.18, lw=0, transform=a2.get_xaxis_transform(), label='ventana grave')
    a2.plot(_r.timestamp_inicio, np.where(_r.grave_1h.to_numpy() == 0, prob, np.nan), color=COLORES[0], lw=2,
            drawstyle='steps-post', label=f'probabilidad de grave en la próxima hora ({_fig_mod}), '
                                         'solo sin grave en la última hora')
    _tope = float(np.nanmax(prob))
    if np.isfinite(umbral):
        a2.axhline(umbral, color=TINTA, lw=1, ls='--', label=f'umbral de aviso ({100 * FPR_AVISO:.0f} % de avisos falsos)')
        _tope = max(_tope, umbral)
    _ini = _r[_r.inicio_grave]
    a2.scatter(_ini.timestamp_inicio, np.full(len(_ini), 0.95), marker='v', s=40, color=CRITICO,
               transform=a2.get_xaxis_transform(), label='inicio de grave')
    a2.set_ylim(0, min(1.0, 1.2 * _tope) if _tope > 0 else 1.0)
    a2.set_ylabel('Probabilidad')
    a2.set_title('Ejemplo en prueba: tramo con más inicios de grave', fontsize=10, pad=44)
    a2.grid(alpha=0.25)
    a2.legend(fontsize=8, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    _loc = mdates.AutoDateLocator()
    a2.xaxis.set_major_locator(_loc)
    a2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(_loc))
else:
    a2.axis('off')
fig.suptitle(f'{EQUIPO}: pronóstico de ventanas graves')
fig.tight_layout()
fig.savefig(salida('06b_pronostico.png'), dpi=150)
plt.close(fig)


# =============================================================================
# 8. GUARDAR
# =============================================================================
_marca, _umbral = MARCA[_fig_mod]
ven = d[LLAVES + ['timestamp_inicio', 'conjunto', 'parte', 'nivel', 'grave', 'grave_1h', f'y_{H}', f'eval_{H}',
                  'inicio_grave']].rename(columns={'grave_1h': 'grave_ultima_hora', f'y_{H}': 'grave_proxima_hora',
                                                   f'eval_{H}': 'evaluable'})
ven['modelo'] = _fig_mod
ven['probabilidad'] = PUNT[(H, 'inicio', _fig_mod)].round(4)
ven['aviso'] = _marca
ven['punto_parada'] = PUNTO_PARADA
guardar(ven, '06b_pronostico_ventanas.csv')
print(f'\nProbabilidad guardada y graficada: {_fig_mod} (próxima hora, umbral fijado en validación B).')
print(f'\nGuardado en {RUTA_EQ}: 06b_pronostico_metricas.csv, 06b_pronostico_comparaciones.csv, '
      f'06b_pronostico_aviso.csv, 06b_pronostico_coeficientes.csv, 06b_pronostico_ventanas.csv, 06b_pronostico.png')
