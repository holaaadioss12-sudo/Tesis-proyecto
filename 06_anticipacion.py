# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 06 — ¿EL AUTOENCODER AVISA ANTES QUE LAS REGLAS? (CON ALARMAS NUEVAS Y REACTIVACIONES)
=============================================================================
"""

import os
import sys
import json

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

if RUTA_BASE not in sys.path:
    sys.path.append(RUTA_BASE)

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and _os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'     # CR010 en su subcarpeta, como en los demás pasos
_os.makedirs(RUTA_EQ, exist_ok=True)
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import criticidad

SEED = 42
W = 60
MIN_POR_VENTANA = W * 10 / 60          # 10 minutos

ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
SALIDA_PRECURSORES = f'{RUTA_EQ}/06_precursores.csv'
SALIDA_EPISODIOS = f'{RUTA_EQ}/06_episodios.csv'
SALIDA_RESUMEN = f'{RUTA_EQ}/06_resumen.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

HORIZONTES = [1, 2, 3]

PERCENTIL_UMBRAL = 95

N_BOOT = 1000

N_MIN_MARCADAS = 30

# Alarma nueva o reactivación (sección 6c): un inicio es NUEVO si antes tuvo al
# menos CALMA_MIN minutos sin ninguna alarma en su tramo, y es una REACTIVACIÓN
# si una alarma del mismo tramo se apagó hace menos. 60 min es la misma hora de
# historia que el 06b usa como persistencia; 30 y 120 min se informan para ver
# si la conclusión depende del corte.
CALMA_MIN = 60
CALMAS_SENSIBILIDAD = [30, 60, 120]

PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
if PUNTO_PARADA not in ('perdida', 'separacion'):
    raise SystemExit(f"PUNTO_PARADA tiene que ser 'perdida' o 'separacion', no {PUNTO_PARADA!r}.")
DETECTOR_PRINCIPAL = 'AE_normales_sep' if PUNTO_PARADA == 'separacion' else 'AE_normales'

PERMITIR_SPLIT_DESVIADO = False


# =============================================================================
# 1. CARGA Y VERIFICACIONES
# =============================================================================
print('=' * 72)
print('PASO 06 — ANTICIPACIÓN: ¿EL DETECTOR AVISA ANTES QUE LAS REGLAS?')
print('=' * 72)

corte = json.load(open(ARCHIVO_CORTE))
T_TRAIN, T_VAL = pd.Timestamp(corte['T_train']), pd.Timestamp(corte['T_val'])
_f_tr = corte.get('frac_train_efectiva')
print(f'Corte: T_train = {T_TRAIN} | T_val = {T_VAL} | '
      f'frac_train_efectiva = {_f_tr}')
if _f_tr is None or abs(_f_tr - corte.get('frac_train', 0.70)) > 0.05:
    print('  EL CORTE NO ESTÁ DONDE DEBE. Corré 01 -> 05 parchados primero.')
    if not PERMITIR_SPLIT_DESVIADO:
        raise SystemExit('Corte desviado.')

ventanas = pd.read_csv(ARCHIVO_VENTANAS, parse_dates=['timestamp_inicio'])
err = pd.read_csv(ARCHIVO_ERROR)
ids_reglas = [c for c in ventanas.columns if c in criticidad.CRITICIDAD]

cols_err = [c for c in ('err_AE_normales_sep', 'err_AE_normales')
            if c in err.columns]
if not cols_err:
    raise SystemExit(f'El paso 04 no trae columnas err_AE_*: {list(err.columns)}')
if 'error_reconstruccion' in err.columns and not np.allclose(err[f'err_{DETECTOR_PRINCIPAL}'], err.error_reconstruccion):
    raise SystemExit(f'El paso 04 se corrió con otro punto de parada: córrelo con PUNTO_PARADA={PUNTO_PARADA!r}.')

LLAVES = ['tramo_id', 'ventana_id']
medias = [f'{v}_media' for v in VARIABLES]
features = pd.read_csv(ARCHIVO_FEATURES, usecols=LLAVES + medias)

tramos_eval = set(err.tramo_id)
d = (ventanas[ventanas.tramo_id.isin(tramos_eval)]
     [LLAVES + ['timestamp_inicio', 'n_reglas_activas', 'w_max'] + ids_reglas]
     .merge(err[LLAVES + ['conjunto'] + cols_err], on=LLAVES, how='left',
            validate='one_to_one')
     .merge(features, on=LLAVES, how='left', validate='one_to_one')
     .sort_values(LLAVES).reset_index(drop=True))
sin_error = int(d[cols_err[0]].isna().sum())
assert sin_error == 0, (f'{sin_error} ventanas de tramos de val/test sin error '
                        f'de reconstrucción: el paso 04 no las evaluó todas.')

_salto = d.groupby('tramo_id').ventana_id.diff().dropna()
assert (_salto == 1).all(), 'ventana_id no es contiguo dentro de algún tramo'


def historia_alarmas(v):
    """Para cada ventana, con todo el registro de su tramo (entrenamiento incluido):
    racha_quieta = cuántas ventanas sin alarma seguidas terminan en ella (0 si tiene
    alarma) y alarma_antes = si antes de esa racha hubo una alarma en el mismo tramo
    (False si la racha llega al inicio del tramo: la historia no se conoce)."""
    v = v[LLAVES + ['n_reglas_activas']].sort_values(LLAVES)
    al = (v.n_reglas_activas > 0).to_numpy()
    tr, vid = v.tramo_id.to_numpy(), v.ventana_id.to_numpy()
    racha = np.zeros(len(v), dtype=int)
    previa = np.zeros(len(v), dtype=bool)
    for i in range(len(v)):
        if al[i]:
            continue
        if i == 0 or tr[i] != tr[i - 1] or vid[i] != vid[i - 1] + 1:
            racha[i] = 1
        elif al[i - 1]:
            racha[i], previa[i] = 1, True
        else:
            racha[i], previa[i] = racha[i - 1] + 1, previa[i - 1]
    return pd.DataFrame({'tramo_id': v.tramo_id.to_numpy(), 'ventana_id': vid,
                         'racha_quieta': racha, 'alarma_antes': previa})


d = d.merge(historia_alarmas(ventanas), on=LLAVES, how='left', validate='one_to_one')
if CALMA_MIN % MIN_POR_VENTANA or any(c % MIN_POR_VENTANA for c in CALMAS_SENSIBILIDAD):
    raise SystemExit(f'CALMA_MIN y CALMAS_SENSIBILIDAD tienen que ser múltiplos de '
                     f'{MIN_POR_VENTANA:.0f} min (una ventana).')
H_CALMA = int(CALMA_MIN // MIN_POR_VENTANA)


def clase_inicio(racha, previa, h):
    """Tipo de un inicio según la ventana quieta que lo precede: 'nueva' si hubo al
    menos h ventanas sin alarma, 'reactivacion' si una alarma del tramo se apagó
    hace menos, 'sin_historia' si el tramo empezó hace menos (no se puede saber)."""
    racha = np.asarray(racha, dtype=int)
    previa = np.asarray(previa, dtype=bool)
    return np.where(racha >= h, 'nueva',
                    np.where(previa, 'reactivacion', 'sin_historia'))


# el mismo criterio, visto desde una ventana quieta: ¿viene de horas de calma o
# de una alarma que se acaba de apagar?
CONTEXTO_DE = {'nueva': 'calma', 'reactivacion': 'alarma_reciente',
               'sin_historia': 'sin_historia'}

d['alarma'] = d.n_reglas_activas > 0
d['quieta'] = ~d.alarma
print(f'Ventanas de val + test: {len(d):,} en {d.tramo_id.nunique():,} tramos')
print(f'  val {int((d.conjunto == "val").sum()):,} | '
      f'test {int((d.conjunto == "test").sum()):,} | '
      f'quietas (ninguna regla activa) {int(d.quieta.sum()):,}')


# =============================================================================
# 2. LOS PUNTAJES DE LOS CUATRO DETECTORES
# =============================================================================
print('\n' + '=' * 72)
print('2. DETECTORES')
print('=' * 72)

_ref = d[(d.conjunto == 'val') & d.quieta]
_mu, _sd = _ref[medias].mean(), _ref[medias].std().replace(0, 1.0)
d['z_medias'] = ((d[medias] - _mu) / _sd).abs().max(axis=1)

DETECTORES = {c.replace('err_', ''): c for c in cols_err}
DETECTORES['indice_W'] = 'w_max'
DETECTORES['z_medias'] = 'z_medias'
_ref = d[(d.conjunto == 'val') & d.quieta]   # re-tomado: ya trae z_medias

umbrales = {}
for nom, col in DETECTORES.items():
    u = float(np.percentile(_ref[col], PERCENTIL_UMBRAL))
    umbrales[nom] = u
    d[f'marca_{nom}'] = d[col] > u
    _te_q = d[(d.conjunto == 'test') & d.quieta]
    print(f'  {nom:<16} umbral P{PERCENTIL_UMBRAL} = {u:.5f} | marca el '
          f'{100 * (_te_q[col] > u).mean():5.1f} % de las quietas de prueba')
print('\nTodos calibrados para marcar ~5 % de las quietas de VALIDACIÓN. Si en')
print('prueba marcan mucho más, es deriva entre períodos: se reporta, y por eso')
print('lo que sigue compara detectores entre sí y contra el azar, no contra un')
print('5 % ideal.')

print('\n' + '=' * 72)
print('2b. DERIVA DE CALIBRACIÓN ENTRE VALIDACIÓN Y PRUEBA (ventanas quietas)')
print('=' * 72)
_filas_deriva = []
for nom, col in DETECTORES.items():
    _qv = d[(d.conjunto == 'val') & d.quieta][col]
    _qt = d[(d.conjunto == 'test') & d.quieta][col]
    _filas_deriva.append({
        'detector': nom, 'umbral_p95_val': round(umbrales[nom], 5),
        'pct_quietas_marcadas_val': round(100 * (_qv > umbrales[nom]).mean(), 1),
        'pct_quietas_marcadas_test': round(100 * (_qt > umbrales[nom]).mean(), 1),
        'mediana_quietas_val': round(float(_qv.median()), 5),
        'mediana_quietas_test': round(float(_qt.median()), 5),
        'razon_medianas_test_val': round(float(_qt.median() / _qv.median()), 2)
                                   if _qv.median() else np.nan,
    })
deriva = pd.DataFrame(_filas_deriva)
print(deriva.to_string(index=False))
print('razón > 1: las ventanas sin ninguna alarma tienen puntajes más altos en')
print('prueba que en validación. Eso es deriva del proceso, no más anomalías.')


# =============================================================================
# 3. QUÉ VIENE DESPUÉS DE CADA VENTANA
# =============================================================================
g = d.groupby('tramo_id')
for k in HORIZONTES:
    fut = pd.concat([g.alarma.shift(-j) for j in range(1, k + 1)], axis=1)
    d[f'futura_{k}'] = np.where(fut.isna().any(axis=1), np.nan,
                                fut.fillna(False).astype(bool).any(axis=1))

te = d[d.conjunto == 'test'].copy()
tramos_te = te.tramo_id.to_numpy()
rng = np.random.default_rng(SEED)


def bootstrap_tramos(df, estadistico, n=N_BOOT):
    grupos = df.groupby('tramo_id').indices
    claves = np.array(list(grupos))
    vals = []
    for _ in range(n):
        sel = np.concatenate([grupos[c] for c in rng.choice(claves, len(claves))])
        v = estadistico(df.iloc[sel])
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


# =============================================================================
# 4. PRECURSORES — ¿LAS MARCAS SOBRE VENTANAS QUIETAS PRECEDEN ALARMAS?
# =============================================================================
print('\n' + '=' * 72)
print('4. PRECURSORES (sobre las ventanas QUIETAS de PRUEBA)')
print('=' * 72)
print('P(alarma en las próximas k ventanas | marcada) contra | no marcada.')
print('Riesgo relativo (RR) > 1 con el IC por encima de 1 = la marca anticipa.\n')

filas_prec = []
for nom in DETECTORES:
    for k in HORIZONTES:
        q = te[te.quieta & te[f'futura_{k}'].notna()]
        m = q[f'marca_{nom}']
        y = q[f'futura_{k}'].astype(bool)

        def rr(x, nom=nom, k=k):
            mm, yy = x[f'marca_{nom}'], x[f'futura_{k}'].astype(bool)
            p1 = yy[mm].mean() if mm.any() else np.nan
            p0 = yy[~mm].mean() if (~mm).any() else np.nan
            return p1 / p0 if p0 and np.isfinite(p0) and p0 > 0 else np.nan

        p1 = float(y[m].mean()) if m.any() else np.nan
        p0 = float(y[~m].mean()) if (~m).any() else np.nan
        lo, hi = bootstrap_tramos(q, rr)
        _suf = int(m.sum()) >= N_MIN_MARCADAS
        if not _suf:
            p1, lo, hi = np.nan, np.nan, np.nan
        filas_prec.append({
            'detector': nom, 'horizonte_min': int(k * MIN_POR_VENTANA),
            'quietas': len(q), 'marcadas': int(m.sum()),
            'p_alarma_si_marcada': round(p1, 4),
            'p_alarma_si_no_marcada': round(p0, 4),
            'riesgo_relativo': round(p1 / p0, 3) if p0 else np.nan,
            'rr_ic_inf': round(lo, 3), 'rr_ic_sup': round(hi, 3),
            'n_suficiente': _suf,
        })
prec = pd.DataFrame(filas_prec)
print(prec.to_string(index=False))
_insuf = prec[~prec.n_suficiente]
if len(_insuf):
    print(f'\nn insuficiente (< {N_MIN_MARCADAS} quietas marcadas): RR e IC no se '
          f'reportan en estas filas:')
    for _, _r in _insuf.iterrows():
        print(f'  {_r.detector:<16} {_r.horizonte_min:>3} min: '
              f'{_r.marcadas} marcadas de {_r.quietas}')


# =============================================================================
# 5. AUC DE ANTICIPACIÓN — sin umbral
# =============================================================================
print('\n' + '=' * 72)
print('5. AUC DE ANTICIPACIÓN (próxima ventana, quietas de prueba)')
print('=' * 72)

q1 = te[te.quieta & te.futura_1.notna()].copy()
q1['y'] = q1.futura_1.astype(bool)
print(f'Ventanas quietas con siguiente observable: {len(q1):,} | '
      f'seguidas de alarma: {int(q1.y.sum()):,} ({100 * q1.y.mean():.1f} %)')
auc = {}
for nom, col in DETECTORES.items():
    def _auc(x, col=col):
        return (roc_auc_score(x.y, x[col])
                if x.y.nunique() == 2 else np.nan)
    a = _auc(q1)
    lo, hi = bootstrap_tramos(q1, _auc)
    auc[nom] = (a, lo, hi)
    print(f'  {nom:<16} AUC = {a:.3f}   IC 95 % [{lo:.3f}, {hi:.3f}]'
          + ('   <- no se distingue del azar' if lo <= 0.5 else ''))


# =============================================================================
# 6. TIEMPO DE ANTICIPACIÓN, EPISODIO POR EPISODIO
# =============================================================================
print('\n' + '=' * 72)
print('6. TIEMPO DE ANTICIPACIÓN (inicios de episodio en prueba)')
print('=' * 72)

te['prev_quieta'] = te.groupby('tramo_id').quieta.shift(1)
inicio = te.alarma & (te.prev_quieta == True)                       # noqa: E712
censurados = int((te.alarma & (te.ventana_id == 0)).sum())
print(f'Inicios de episodio observables: {int(inicio.sum()):,} | censurados '
      f'(alarma ya en la ventana 0 del tramo): {censurados:,}')

# la regla más grave que se ENCIENDE en la ventana de inicio
_act = te[ids_reglas].to_numpy() > 0
_prev_act = te.groupby('tramo_id')[ids_reglas].shift(1).fillna(0).to_numpy() > 0
_nuevas = _act & ~_prev_act
_crit = np.array([criticidad.CRITICIDAD[r] for r in ids_reglas])
_comp = np.array([criticidad.COMPONENTE[r] for r in ids_reglas])

eps = []
for i in np.flatnonzero(inicio.to_numpy()):
    fila = te.iloc[i]
    nuevas = np.flatnonzero(_nuevas[i]) if _nuevas[i].any() else np.flatnonzero(_act[i])
    peor = nuevas[_crit[nuevas].argmin()]
    reg = {'tramo_id': int(fila.tramo_id), 'ventana_id': int(fila.ventana_id),
           'timestamp_inicio': fila.timestamp_inicio,
           'reglas_nuevas': ','.join(np.array(ids_reglas)[nuevas]),
           'regla_mas_grave': ids_reglas[peor],
           'criticidad': int(_crit[peor]), 'componente': _comp[peor],
           'trip': bool(set(np.array(ids_reglas)[nuevas]) & set(criticidad.REGLAS_TRIP)),
           'quietas_previas': 0}
    # con toda la historia del tramo (06c): la ventana quieta justo antes del inicio
    _prev = te.iloc[i - 1]
    reg['quietas_antes'] = int(_prev.racha_quieta)
    reg['alarma_antes'] = bool(_prev.alarma_antes)
    # cuántas ventanas quietas hay justo antes, y cuántas marcadas seguidas
    j = i - 1
    while (j >= 0 and te.iloc[j].tramo_id == fila.tramo_id
           and bool(te.iloc[j].quieta)):
        reg['quietas_previas'] += 1
        j -= 1
    for nom in DETECTORES:
        n = 0
        j = i - 1
        while (n < reg['quietas_previas']
               and bool(te.iloc[j][f'marca_{nom}'])):
            n += 1
            j -= 1
        reg[f'anticip_{nom}_min'] = n * MIN_POR_VENTANA
    eps.append(reg)
eps = pd.DataFrame(eps)
if len(eps):
    eps['min_desde_alarma_previa'] = np.where(
        eps.alarma_antes, eps.quietas_antes * MIN_POR_VENTANA, np.nan)
    eps['tipo_inicio'] = clase_inicio(eps.quietas_antes, eps.alarma_antes, H_CALMA)

filas_res = []
if len(eps):
    print(f'\nQuietas previas disponibles (cuánto "antes" se puede ver): mediana '
          f'{eps.quietas_previas.median():.0f} ventanas | '
          f'{100 * (eps.quietas_previas >= 3).mean():.0f} % de los inicios '
          f'tiene 3 o más.')
    print(f'Inicios con una regla TRIP (L, A8, A11): {int(eps.trip.sum())}')

    _total_inicios = len(eps) + censurados
    _pct_avisables = 100 * len(eps) / _total_inicios if _total_inicios else np.nan
    print(f'\nTecho estructural: de {_total_inicios:,} inicios de alarma en prueba, '
          f'{len(eps):,} ({_pct_avisables:.1f} %) tienen al menos una ventana')
    print('quieta antes en su tramo. El resto no lo puede avisar ningún detector.')
    print('\nPor detector:')
    for nom in DETECTORES:
        a = eps[f'anticip_{nom}_min']
        avisados = a > 0
        lo, hi = bootstrap_tramos(eps, lambda x, nom=nom:
                                  (x[f'anticip_{nom}_min'] > 0).mean())
        _ctrl = te[te.quieta & (te.futura_1 == 0)]
        azar = float(_ctrl[f'marca_{nom}'].mean())
        a_av = a[avisados]
        filas_res.append({
            'detector': nom, 'inicios': len(eps),
            'pct_avisados': round(100 * avisados.mean(), 1),
            'ic_inf': round(100 * lo, 1), 'ic_sup': round(100 * hi, 1),
            'pct_esperado_por_azar': round(100 * azar, 1),
            'anticip_mediana_min_si_aviso': (float(a_av.median())
                                             if len(a_av) else np.nan),
            'anticip_p90_min_si_aviso': (float(a_av.quantile(0.9))
                                         if len(a_av) else np.nan),
            'auc_anticipacion': round(auc[nom][0], 3),
            'auc_ic_inf': round(auc[nom][1], 3),
            'auc_ic_sup': round(auc[nom][2], 3),
            'inicios_censurados': censurados,
            'pct_inicios_avisables': round(_pct_avisables, 1),
            'pct_avisados_sobre_todos': round(100 * avisados.sum()
                                              / _total_inicios, 1),
            'pct_avisos_en_techo_adelanto': round(100 * (
                a[avisados] >= eps.quietas_previas[avisados] * MIN_POR_VENTANA
            ).mean(), 1) if avisados.any() else np.nan,
        })
        print(f'  {nom:<16} avisó antes del {100 * avisados.mean():5.1f} % de los '
              f'inicios [IC {100 * lo:.1f}-{100 * hi:.1f}] | por azar '
              f'{100 * azar:.1f} % | adelanto mediano '
              f'{a_av.median() if len(a_av) else float("nan"):.0f} min')
        print(f'  {"":<16} = {100 * avisados.sum() / _total_inicios:.1f} % de '
              f'TODOS los inicios (techo {_pct_avisables:.1f} %) | en el '
              f'{filas_res[-1]["pct_avisos_en_techo_adelanto"]} % de los avisos el '
              f'adelanto ya es todo lo que el tramo deja ver')

    print('\nPor criticidad de la regla que se enciende (detector principal):')
    _col = f'anticip_{DETECTOR_PRINCIPAL}_min'
    def _por_grupo(col_g):
        t = eps.groupby(col_g).agg(
            inicios=(_col, 'size'),
            pct_avisados=(_col, lambda s: round(100 * (s > 0).mean(), 1)),
            adelanto_mediano_min=(_col, lambda s: s[s > 0].median()))
        t['nota'] = np.where(t.inicios < N_MIN_MARCADAS, 'n insuficiente', '')
        return t.sort_values('inicios', ascending=False).to_string()
    print(_por_grupo('criticidad'))
    print('\nPor componente:')
    print(_por_grupo('componente'))
else:
    print('No hay inicios de episodio observables en prueba.')

resumen = pd.DataFrame(filas_res)


# =============================================================================
# 6b.LÍNEA BASE DE LA DETECCIÓN (no de la anticipación)
# =============================================================================
print('\n' + '=' * 72)
print('6b. AUC DE DETECCIÓN EN PRUEBA (alarma vs quieta) — línea base')
print('=' * 72)
td = te.copy()
td['y'] = td.alarma
print(f'Ventanas de prueba: {len(td):,} | con alarma: {int(td.y.sum()):,}')
auc_det = {}
for nom, col in DETECTORES.items():
    def _aucd(x, col=col):
        return (roc_auc_score(x.y, x[col])
                if x.y.nunique() == 2 else np.nan)
    a = _aucd(td)
    lo, hi = bootstrap_tramos(td, _aucd)
    auc_det[nom] = (a, lo, hi)
    print(f'  {nom:<16} AUC = {a:.3f}   IC 95 % [{lo:.3f}, {hi:.3f}]')

def _dif(x):
    if x.y.nunique() < 2:
        return np.nan
    return (roc_auc_score(x.y, x[DETECTORES[DETECTOR_PRINCIPAL]])
            - roc_auc_score(x.y, x['z_medias']))
d_det = _dif(td)
d_lo, d_hi = bootstrap_tramos(td, _dif)
print(f'\n  {DETECTOR_PRINCIPAL} - z_medias = {d_det:+.3f}   '
      f'IC 95 % [{d_lo:+.3f}, {d_hi:+.3f}]')
if d_lo > 0:
    print('  -> El autoencoder DETECTA mejor que la línea base trivial.')
elif d_hi < 0:
    print('  -> z_medias detecta mejor que el autoencoder: se reporta así.')
else:
    print('  -> No hay diferencia demostrable en detección con la línea base.')
if len(resumen):
    resumen['auc_deteccion'] = resumen.detector.map(
        lambda n: round(auc_det[n][0], 3))
    resumen['auc_det_ic_inf'] = resumen.detector.map(
        lambda n: round(auc_det[n][1], 3))
    resumen['auc_det_ic_sup'] = resumen.detector.map(
        lambda n: round(auc_det[n][2], 3))

_difs = {DETECTOR_PRINCIPAL: (d_det, d_lo, d_hi)}
for nom in [n for n in DETECTORES if n.startswith('AE_') and n != DETECTOR_PRINCIPAL]:
    def _dif_n(x, nom=nom):
        if x.y.nunique() < 2:
            return np.nan
        return (roc_auc_score(x.y, x[DETECTORES[nom]])
                - roc_auc_score(x.y, x['z_medias']))
    _lo, _hi = bootstrap_tramos(td, _dif_n)
    _difs[nom] = (_dif_n(td), _lo, _hi)
    print(f'  {nom} - z_medias = {_difs[nom][0]:+.3f}   '
          f'IC 95 % [{_lo:+.3f}, {_hi:+.3f}]')
if len(resumen):
    for _k, _j in (('dif_auc_det_vs_z', 0), ('dif_ic_inf', 1), ('dif_ic_sup', 2)):
        resumen[_k] = resumen.detector.map(
            lambda n, j=_j: round(_difs[n][j], 3) if n in _difs else np.nan)


# =============================================================================
# 6c. ALARMAS NUEVAS Y REACTIVACIONES
# =============================================================================
print('\n' + '=' * 72)
print(f'6c. ALARMAS NUEVAS Y REACTIVACIONES (calma de {CALMA_MIN} min, prueba)')
print('=' * 72)
print('Una alarma que se apagó hace 10 min y vuelve no es una alarma nueva: la')
print('ventana quieta de en medio todavía arrastra la perturbación, y marcarla no')
print('es anticipar. Cada inicio se clasifica por lo que pasó antes en su tramo:')
print(f'  nueva         al menos {CALMA_MIN} min sin ninguna alarma antes del inicio')
print(f'  reactivacion  una alarma del mismo tramo se apagó hace menos de {CALMA_MIN} min')
print(f'  sin_historia  el tramo empezó hace menos de {CALMA_MIN} min: no se sabe')
print('El azar de cada tipo sale de ventanas quietas a la misma distancia de la')
print('alarma anterior que NO fueron seguidas por una alarma.')

TIPOS = ['nueva', 'reactivacion', 'sin_historia']


def _clave_distancia(racha, previa, h):
    """'calma' si hubo h o más ventanas sin alarma; si no, la distancia exacta a la
    alarma anterior, con r (hubo alarma antes) o s (sin historia) adelante."""
    racha = np.asarray(racha, dtype=int)
    previa = np.asarray(previa, dtype=bool)
    return np.where(racha >= h, 'calma',
                    np.char.add(np.where(previa, 'r', 's'), racha.astype(str)))


def _azar_por_inicio(nom, h):
    """Para cada inicio, la probabilidad de que la ventana quieta anterior quede marcada
    si la marca no supiera nada de la alarma que viene: la tasa de marcas en ventanas
    quietas de prueba NO seguidas por alarma y a la misma distancia de la alarma
    anterior (si no hay ninguna a esa distancia, la tasa de su contexto)."""
    c = te[te.quieta & (te.futura_1 == 0)]
    marca = pd.Series(c[f'marca_{nom}'].to_numpy())
    fina = marca.groupby(_clave_distancia(c.racha_quieta, c.alarma_antes, h)).mean()
    gruesa = marca.groupby(pd.Series(clase_inicio(c.racha_quieta, c.alarma_antes, h))
                           .map(CONTEXTO_DE).to_numpy()).mean()
    p = pd.Series(_clave_distancia(eps.quietas_antes, eps.alarma_antes, h)).map(fina)
    ctx = pd.Series(clase_inicio(eps.quietas_antes, eps.alarma_antes, h)).map(CONTEXTO_DE)
    return p.fillna(ctx.map(gruesa)).to_numpy(dtype=float)


def _riesgo(q, nom, k):
    """Riesgo relativo de alarma en las próximas k ventanas, marcada contra no marcada."""
    m = q[f'marca_{nom}'].astype(bool)
    y = q[f'futura_{k}'].astype(bool)
    p1 = float(y[m].mean()) if m.any() else np.nan
    p0 = float(y[~m].mean()) if (~m).any() else np.nan

    def rr(x):
        mm, yy = x[f'marca_{nom}'].astype(bool), x[f'futura_{k}'].astype(bool)
        a = yy[mm].mean() if mm.any() else np.nan
        b = yy[~mm].mean() if (~mm).any() else np.nan
        return a / b if np.isfinite(b) and b > 0 else np.nan

    suf = int(m.sum()) >= N_MIN_MARCADAS
    lo, hi = bootstrap_tramos(q, rr) if suf else (np.nan, np.nan)
    return {'quietas': len(q), 'marcadas': int(m.sum()),
            'p_alarma_si_marcada': round(p1, 4) if suf else np.nan,
            'p_alarma_si_no_marcada': round(p0, 4),
            'riesgo_relativo': round(p1 / p0, 3) if suf and p0 > 0 else np.nan,
            'rr_ic_inf': round(lo, 3), 'rr_ic_sup': round(hi, 3),
            'n_suficiente': suf}


def _auc_anticipacion(q, col):
    """AUC del puntaje para separar las quietas seguidas por alarma de las que no."""
    def f(x):
        return roc_auc_score(x.y, x[col]) if x.y.nunique() == 2 else np.nan
    a = f(q)
    if not np.isfinite(a):
        return np.nan, np.nan, np.nan
    lo, hi = bootstrap_tramos(q, f)
    return a, lo, hi


filas_tipo, filas_ctx = [], []
for calma in CALMAS_SENSIBILIDAD:
    h = int(calma // MIN_POR_VENTANA)
    principal = calma == CALMA_MIN

    # --- inicios: ¿la ventana quieta anterior quedó marcada? -------------
    if len(eps):
        tipo = clase_inicio(eps.quietas_antes, eps.alarma_antes, h)
        for nom in DETECTORES:
            col = f'anticip_{nom}_min'
            azar_i = _azar_por_inicio(nom, h)
            for t in TIPOS:
                sel = tipo == t
                fila = {'calma_min': calma, 'principal': principal,
                        'tipo_inicio': t, 'detector': nom, 'inicios': int(sel.sum()),
                        'pct_de_los_inicios': round(100 * sel.mean(), 1)}
                if sel.any():
                    e = eps.loc[sel, ['tramo_id', col]]
                    av = e[col] > 0
                    lo, hi = bootstrap_tramos(e, lambda x, col=col: (x[col] > 0).mean())
                    fila.update({
                        'pct_avisados': round(100 * av.mean(), 1),
                        'ic_inf': round(100 * lo, 1), 'ic_sup': round(100 * hi, 1),
                        'pct_esperado_por_azar': round(100 * float(np.nanmean(azar_i[sel])), 1),
                        'adelanto_mediano_min': (float(e.loc[av, col].median())
                                                 if av.any() else np.nan),
                        'nota': 'n insuficiente' if sel.sum() < N_MIN_MARCADAS else ''})
                filas_tipo.append(fila)

    # --- ventanas quietas: riesgo relativo y AUC según el contexto -------
    ctx = pd.Series(clase_inicio(te.racha_quieta, te.alarma_antes, h),
                    index=te.index).map(CONTEXTO_DE)
    for c in ('calma', 'alarma_reciente'):
        for nom, col_det in DETECTORES.items():
            for k in (HORIZONTES if principal else [1]):
                q = te.loc[te.quieta & te[f'futura_{k}'].notna() & (ctx == c),
                           ['tramo_id', f'marca_{nom}', f'futura_{k}', col_det]]
                fila = {'calma_min': calma, 'principal': principal, 'contexto': c,
                        'detector': nom, 'horizonte_min': int(k * MIN_POR_VENTANA)}
                if len(q):
                    fila.update(_riesgo(q, nom, k))
                    if k == 1:
                        a, lo, hi = _auc_anticipacion(
                            q.assign(y=q.futura_1.astype(bool)), col_det)
                        fila.update({'auc_anticipacion': round(a, 3),
                                     'auc_ic_inf': round(lo, 3), 'auc_ic_sup': round(hi, 3)})
                else:
                    fila.update({'quietas': 0, 'marcadas': 0, 'n_suficiente': False})
                filas_ctx.append(fila)

tipos_ini = pd.DataFrame(filas_tipo)
prec_ctx = pd.DataFrame(filas_ctx)

if len(eps):
    _n = tipos_ini[tipos_ini.principal & (tipos_ini.detector == DETECTOR_PRINCIPAL)]
    print(f'\nInicios de prueba por tipo: ' + ' | '.join(
        f'{r.tipo_inicio} {r.inicios} ({r.pct_de_los_inicios:.0f} %)' for _, r in _n.iterrows()))
    _r = eps.loc[eps.tipo_inicio == 'reactivacion', 'min_desde_alarma_previa']
    if len(_r):
        print(f'Reactivaciones: la alarma anterior se había apagado hace una mediana de '
              f'{_r.median():.0f} min; el {100 * (_r <= 10).mean():.0f} % volvió en 10 min '
              f'o menos y el {100 * (_r <= 30).mean():.0f} % en 30 min o menos.')
    _a = eps.loc[eps.alarma_antes, 'min_desde_alarma_previa']
    if len(_a):
        print(f'Inicios con alguna alarma antes en su tramo, a cualquier distancia: '
              f'{len(_a)} de {len(eps)} ({100 * len(_a) / len(eps):.0f} %), mediana '
              f'{_a.median():.0f} min desde que se apagó.')

    print('\n¿Avisó antes del inicio? (la ventana quieta anterior quedó marcada)')
    print(f'  {"detector":<16} {"tipo":<13} {"inicios":>7} {"avisó":>7}  {"IC 95 %":<13} '
          f'{"azar":>6} {"adelanto":>9}')
    for _, r in tipos_ini[tipos_ini.principal & (tipos_ini.inicios > 0)].iterrows():
        print(f'  {r.detector:<16} {r.tipo_inicio:<13} {r.inicios:>7} {r.pct_avisados:>6.1f}%  '
              f'[{r.ic_inf:4.1f}-{r.ic_sup:4.1f}]  {r.pct_esperado_por_azar:>5.1f}% '
              f'{r.adelanto_mediano_min:>5.0f} min  {r.nota}')
else:
    print('No hay inicios de episodio observables en prueba.')

print(f'\nRiesgo relativo y AUC de anticipación según el contexto de la ventana quieta:')
print(f'calma = al menos {CALMA_MIN} min sin alarma; alarma_reciente = se apagó una hace menos.')
_cols = ['contexto', 'detector', 'horizonte_min', 'quietas', 'marcadas', 'riesgo_relativo',
         'rr_ic_inf', 'rr_ic_sup', 'auc_anticipacion', 'auc_ic_inf', 'auc_ic_sup']
print(prec_ctx[prec_ctx.principal].reindex(columns=_cols).to_string(index=False))

print(f'\nSensibilidad al corte de calma ({DETECTOR_PRINCIPAL}, 10 min):')
for calma in CALMAS_SENSIBILIDAD:
    _t = tipos_ini[(tipos_ini.calma_min == calma) & (tipos_ini.detector == DETECTOR_PRINCIPAL)
                   & (tipos_ini.tipo_inicio == 'nueva')]
    _c = prec_ctx[(prec_ctx.calma_min == calma) & (prec_ctx.detector == DETECTOR_PRINCIPAL)
                  & (prec_ctx.contexto == 'calma') & (prec_ctx.horizonte_min == 10)]
    _txt = f'  {calma:>4} min: '
    if len(_t) and _t.iloc[0].inicios:
        _t = _t.iloc[0]
        _txt += (f'{_t.inicios} inicios nuevos, avisó {_t.pct_avisados:.1f} % '
                 f'(azar {_t.pct_esperado_por_azar:.1f} %)')
    else:
        _txt += 'sin inicios nuevos'
    if len(_c):
        _c = _c.iloc[0]
        _txt += (f' | RR en calma {_c.riesgo_relativo:.2f} [{_c.rr_ic_inf:.2f}, {_c.rr_ic_sup:.2f}]'
                 if _c.n_suficiente else f' | RR en calma: n insuficiente ({_c.marcadas} marcadas)')
        if np.isfinite(_c.get('auc_anticipacion', np.nan)):
            _txt += (f' | AUC {_c.auc_anticipacion:.3f} '
                     f'[{_c.auc_ic_inf:.3f}, {_c.auc_ic_sup:.3f}]')
    print(_txt)


# =============================================================================
# 7. GUARDAR Y LEER
# =============================================================================
print('\n' + '=' * 72)
print('7. GUARDANDO')
print('=' * 72)
for _df, _f in ((prec, SALIDA_PRECURSORES), (eps, SALIDA_EPISODIOS),
                (resumen, SALIDA_RESUMEN),
                (deriva, f'{RUTA_EQ}/06_deriva.csv'),
                (tipos_ini, f'{RUTA_EQ}/06_inicios_por_tipo.csv'),
                (prec_ctx, f'{RUTA_EQ}/06_precursores_contexto.csv')):
    _df.to_csv(_f, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'Guardado: {_f}')

print('\n' + '=' * 72)
print(f'RESUMEN — detector principal: {DETECTOR_PRINCIPAL}')
print('=' * 72)
if len(resumen):
    r = resumen.set_index('detector')
    p = r.loc[DETECTOR_PRINCIPAL]
    pr = prec[(prec.detector == DETECTOR_PRINCIPAL) & (prec.horizonte_min == 10)]
    if len(pr) and not pr.iloc[0].n_suficiente:
        print(f'RR a 10 min: n insuficiente ({pr.iloc[0].marcadas} quietas '
              f'marcadas, mínimo {N_MIN_MARCADAS}).')
    elif len(pr):
        pr = pr.iloc[0]
        print(f'Una ventana quieta marcada tiene {pr.riesgo_relativo:.2f} veces '
              f'más probabilidad de ser seguida por una alarma en 10 min '
              f'(IC {pr.rr_ic_inf:.2f}-{pr.rr_ic_sup:.2f}).')
    print(f'Avisó antes del {p.pct_avisados} % de los inicios de alarma '
          f'(IC {p.ic_inf}-{p.ic_sup}); por azar serían el '
          f'{p.pct_esperado_por_azar} %.')
    print(f'AUC de anticipación: {p.auc_anticipacion} '
          f'(IC {p.auc_ic_inf}-{p.auc_ic_sup}).')
    print('\nContra los otros detectores (AUC de anticipación):')
    for nom, fila in r.sort_values('auc_anticipacion', ascending=False).iterrows():
        print(f'  {nom:<16} {fila.auc_anticipacion:.3f} '
              f'[{fila.auc_ic_inf:.3f}, {fila.auc_ic_sup:.3f}]')
    mejor_base = r.drop(index=[n for n in r.index if n.startswith('AE_')],
                        errors='ignore')
    if len(mejor_base):
        b = mejor_base.auc_anticipacion.idxmax()
        if p.auc_ic_inf > mejor_base.loc[b, 'auc_ic_sup']:
            print(f'\n-> El autoencoder anticipa MÁS que {b}: los intervalos no '
                  f'se tocan. El deep learning aporta anticipación.')
        elif p.auc_ic_sup < mejor_base.loc[b, 'auc_ic_inf']:
            print(f'\n-> {b} anticipa MÁS que el autoencoder. Se reporta así: el')
            print('   deep learning detecta, pero no es lo que mejor anticipa.')
        else:
            print(f'\n-> El autoencoder y {b} se solapan: no hay evidencia de que')
            print('   el deep learning anticipe más que la línea base. También es')
            print('   un resultado, y acota lo que se afirma.')
    if p.auc_ic_inf > 0.5:
        print('\nLa anticipación existe y no es azar: el objetivo de PRONOSTICAR')
        print('síntomas queda respaldado, con el horizonte que permiten los tramos.')
        print('Ojo: esto mezcla alarmas nuevas y reactivaciones; lo que sigue las separa.')
    else:
        print('\nEl AUC de anticipación no se separa de 0,5: con estos datos el')
        print('autoencoder DETECTA pero no ANTICIPA. Hay que decirlo así.')

if len(tipos_ini):
    _t = (tipos_ini[tipos_ini.principal & (tipos_ini.detector == DETECTOR_PRINCIPAL)]
          .set_index('tipo_inicio'))
    _c = (prec_ctx[prec_ctx.principal & (prec_ctx.detector == DETECTOR_PRINCIPAL)
                   & (prec_ctx.horizonte_min == 10)].set_index('contexto'))
    print(f'\nALARMAS NUEVAS (al menos {CALMA_MIN} min sin alarma antes): '
          f'{int(_t.loc["nueva", "inicios"])} de {len(eps)} inicios '
          f'({_t.loc["nueva", "pct_de_los_inicios"]:.0f} %).')
    if _t.loc['nueva', 'inicios']:
        print(f'  Avisó antes del {_t.loc["nueva", "pct_avisados"]} % '
              f'(IC {_t.loc["nueva", "ic_inf"]}-{_t.loc["nueva", "ic_sup"]}); por azar '
              f'{_t.loc["nueva", "pct_esperado_por_azar"]} %.')
    if _t.loc['reactivacion', 'inicios']:
        print(f'REACTIVACIONES: {int(_t.loc["reactivacion", "inicios"])} inicios; avisó '
              f'antes del {_t.loc["reactivacion", "pct_avisados"]} %; por azar '
              f'{_t.loc["reactivacion", "pct_esperado_por_azar"]} %.')

    _ok_rr = _ok_auc = False
    if 'calma' in _c.index and _c.loc['calma', 'quietas']:
        r = _c.loc['calma']
        if r.n_suficiente:
            _ok_rr = r.rr_ic_inf > 1
            print(f'En calma, una ventana marcada tiene {r.riesgo_relativo:.2f} veces más '
                  f'probabilidad de ser seguida por una alarma en 10 min '
                  f'(IC {r.rr_ic_inf:.2f}-{r.rr_ic_sup:.2f}).')
        else:
            print(f'En calma: n insuficiente para el riesgo relativo ({r.marcadas} quietas '
                  f'marcadas, mínimo {N_MIN_MARCADAS}).')
        if np.isfinite(r.get('auc_anticipacion', np.nan)):
            _ok_auc = r.auc_ic_inf > 0.5
            print(f'AUC de anticipación en calma: {r.auc_anticipacion:.3f} '
                  f'(IC {r.auc_ic_inf:.3f}-{r.auc_ic_sup:.3f}).')
        _z = prec_ctx[prec_ctx.principal & (prec_ctx.detector == 'z_medias')
                      & (prec_ctx.contexto == 'calma') & (prec_ctx.horizonte_min == 10)]
        if len(_z) and np.isfinite(_z.iloc[0].get('auc_anticipacion', np.nan)):
            _z = _z.iloc[0]
            print(f'  z de las medias, en calma: AUC {_z.auc_anticipacion:.3f} '
                  f'(IC {_z.auc_ic_inf:.3f}-{_z.auc_ic_sup:.3f}).')

    print('\nLECTURA:')
    if _ok_rr or _ok_auc:
        print('-> También en calma, lejos de una alarma anterior, el puntaje sube antes')
        print('   de una alarma nueva: la anticipación no es solo la perturbación que')
        print('   sigue después de que la regla se apagó. El horizonte es el de arriba.')
    else:
        print('-> Lejos de una alarma anterior la marca no se separa del azar: la')
        print('   anticipación del total viene de las reactivaciones, es decir, el modelo')
        print('   ve que la perturbación sigue cuando la regla ya se apagó. Eso sirve para')
        print('   no dar por cerrada una alarma, pero no es anticipar alarmas nuevas.')

print('\n' + '=' * 72)
print('LISTO.')
print('=' * 72)
