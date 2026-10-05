# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 06 — ¿EL AUTOENCODER AVISA ANTES QUE LAS REGLAS?
=============================================================================
Entradas: CR010_ventanas_W60.csv              (paso 02: reglas por ventana, w_max)
          CR010_features_W60.csv              (paso 02: para la línea base)
          CR010_error_reconstruccion_W60.csv  (paso 05: error por ventana)
          CR010_corte_split.json              (paso 01)
          criticidad.py                       (misma carpeta de Drive)

Salidas:  06_precursores.csv          tasa de alarma futura, marcadas vs. no
          06_episodios.csv            un renglón por inicio de alarma en prueba
          06_resumen.csv              una fila por detector

-----------------------------------------------------------------------------
LA PREGUNTA, Y POR QUÉ HACE FALTA ESTE PASO
-----------------------------------------------------------------------------
El objetivo de la tesis dice PRONOSTICAR síntomas de falla. Lo que mostraron
los pasos 04, 05 y la Fase B es DETECCIÓN SIMULTÁNEA: el autoencoder marca las
ventanas en que las reglas ya están activas. Eso es necesario, pero no es
pronóstico. Lo que se quiere demostrar es lo que el handoff llama el objetivo
de fondo: avisar cuando el comportamiento empieza a desviarse, AUNQUE NINGÚN
UMBRAL SE HAYA CRUZADO TODAVÍA.

El material para medirlo ya existe. El filtro del autoencoder marca ventanas en
que no hay ninguna regla activa -- en prueba, con P95, el 37 % de las
normales -- y hasta ahora se las contaba como falsas alarmas. Este paso
pregunta si esas "falsas alarmas" vienen ANTES de una alarma real más seguido
de lo que vendría por azar. Si es así, no son falsas: son avisos anticipados.

No entrena nada. Cruza el error de reconstrucción con los episodios de alarma.

-----------------------------------------------------------------------------
TRES MEDICIONES
-----------------------------------------------------------------------------
(1) PRECURSORES. Entre las ventanas SIN NINGUNA regla activa ("quietas"):
    ¿qué fracción tiene una alarma en las próximas k ventanas del mismo tramo,
    si el detector la marcó, contra si no la marcó? Riesgo relativo, con IC
    por bootstrap de tramos.

(2) AUC DE ANTICIPACIÓN. Sobre las mismas ventanas quietas, el puntaje del
    detector como predictor de "hay alarma en la próxima ventana". No depende
    de ningún umbral: 0,5 es azar.

(3) TIEMPO DE ANTICIPACIÓN. Para cada inicio de un episodio de alarma, cuántas
    ventanas quietas inmediatamente anteriores ya estaban marcadas, sin
    interrupción. Eso, por 10 minutos, es cuánto antes avisó.

-----------------------------------------------------------------------------
CONTRA QUÉ SE COMPARA
-----------------------------------------------------------------------------
Que el autoencoder anticipe no basta: hay que mostrar que anticipa más que algo
más simple. Se miden cuatro detectores, todos calibrados IGUAL -- umbral en el
percentil 95 del puntaje sobre las ventanas quietas de VALIDACIÓN -- y todos
evaluados sobre PRUEBA:

    AE_normales_sep   el autoencoder, checkpoint por separación (el del 05)
    AE_normales       el autoencoder, checkpoint por pérdida mínima
    indice_W          w_max del paso 02 -- Wasserstein, sin deep learning
    z_medias          el máximo |z| de las medias de la ventana -- lo trivial

Si el autoencoder no le gana a z_medias, el deep learning no está aportando
anticipación, y eso se reporta tal cual. Medir los dos checkpoints sirve además
para la decisión pendiente sobre MODELO_AE (el 04e).

-----------------------------------------------------------------------------
LÍMITES QUE VAN DECLARADOS
-----------------------------------------------------------------------------
- Sin bitácora de mantenimiento, lo que se anticipa son ALARMAS del sistema,
  no fallas confirmadas. Es consistente con cómo la tesis define "síntoma".
- Los tramos duran ~1,7 h en promedio: el horizonte medible es de decenas de
  minutos. Un aviso con horas de anticipación no se puede ver en estos datos.
- Un episodio que ya está activo en la primera ventana de un tramo no tiene un
  "antes" observable: queda fuera del análisis de anticipación y se cuenta.

Corre en CPU, en segundos.
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

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import criticidad

SEED = 42
W = 60
MIN_POR_VENTANA = W * 10 / 60          # 10 minutos

ARCHIVO_VENTANAS = f'{RUTA_BASE}/CR010_ventanas_W{W}.csv'
ARCHIVO_FEATURES = f'{RUTA_BASE}/CR010_features_W{W}.csv'
ARCHIVO_ERROR = f'{RUTA_BASE}/CR010_error_reconstruccion_W{W}.csv'
ARCHIVO_CORTE = f'{RUTA_BASE}/CR010_corte_split.json'
SALIDA_PRECURSORES = f'{RUTA_BASE}/06_precursores.csv'
SALIDA_EPISODIOS = f'{RUTA_BASE}/06_episodios.csv'
SALIDA_RESUMEN = f'{RUTA_BASE}/06_resumen.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

# Horizontes en ventanas: 1, 2 y 3 son 10, 20 y 30 minutos. Más allá, con
# tramos de ~10 ventanas, casi todas las ventanas quedan censuradas.
HORIZONTES = [1, 2, 3]

# Todos los detectores se calibran igual: percentil 95 del puntaje en las
# ventanas quietas de validación. Así todos tienen ~5 % de marcas en lo quieto
# de validación, y la comparación es justa.
PERCENTIL_UMBRAL = 95

N_BOOT = 1000

# El detector cuyo resumen va al cierre. Es el mismo checkpoint del paso 05 y
# de la Fase B; los otros se miden igual, al lado.
DETECTOR_PRINCIPAL = 'AE_normales_sep'

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
    raise SystemExit(f'El paso 05 no trae columnas err_AE_*: {list(err.columns)}')

LLAVES = ['tramo_id', 'ventana_id']
medias = [f'{v}_media' for v in VARIABLES]
features = pd.read_csv(ARCHIVO_FEATURES, usecols=LLAVES + medias)

# Sólo los tramos de validación y prueba tienen error de reconstrucción. Los
# tramos no cruzan el corte (PARCHE 2), así que cada tramo está entero de un
# lado: se toman los tramos que aparecen en el archivo de error, completos.
tramos_eval = set(err.tramo_id)
d = (ventanas[ventanas.tramo_id.isin(tramos_eval)]
     [LLAVES + ['timestamp_inicio', 'n_reglas_activas', 'w_max'] + ids_reglas]
     .merge(err[LLAVES + ['conjunto'] + cols_err], on=LLAVES, how='left',
            validate='one_to_one')
     .merge(features, on=LLAVES, how='left', validate='one_to_one')
     .sort_values(LLAVES).reset_index(drop=True))
sin_error = int(d[cols_err[0]].isna().sum())
assert sin_error == 0, (f'{sin_error} ventanas de tramos de val/test sin error '
                        f'de reconstrucción: el paso 05 no las evaluó todas.')

# ventana_id tiene que ser contigua dentro de cada tramo: el "antes" y el
# "después" de una ventana se buscan por índice, no por fecha.
_salto = d.groupby('tramo_id').ventana_id.diff().dropna()
assert (_salto == 1).all(), 'ventana_id no es contiguo dentro de algún tramo'

d['alarma'] = d.n_reglas_activas > 0
d['quieta'] = ~d.alarma
print(f'Ventanas de val + test: {len(d):,} en {d.tramo_id.nunique():,} tramos')
print(f'  val {int((d.conjunto == "val").sum()):,} | '
      f'test {int((d.conjunto == "test").sum()):,} | '
      f'quietas (ninguna regla activa) {int(d.quieta.sum()):,}')


# =============================================================================
# 2. LOS PUNTAJES DE LOS CUATRO DETECTORES
# =============================================================================
# z_medias: la línea base trivial. Cada sensor se estandariza con la media y
# la desviación de sus medias en las ventanas QUIETAS de VALIDACIÓN, y el
# puntaje de la ventana es el máximo |z| sobre los 14. No sabe de forma, ni
# de dinámica, ni de correlación entre sensores: sólo "algún sensor está
# lejos de su nivel normal".
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
print('prueba marcan mucho más, es la deriva entre períodos que ya mostró la')
print('Fase B: se reporta, y por eso lo que sigue compara detectores entre sí')
print('y contra el azar, no contra un 5 % ideal.')


# =============================================================================
# 3. QUÉ VIENE DESPUÉS DE CADA VENTANA
# =============================================================================
# futura_k: hay alarma en alguna de las k ventanas siguientes DEL MISMO TRAMO.
# Si el tramo se acaba antes de k ventanas, el resultado no se conoce y la
# ventana queda censurada para ese horizonte (NaN), no se cuenta como "no".
g = d.groupby('tramo_id')
for k in HORIZONTES:
    fut = pd.concat([g.alarma.shift(-j) for j in range(1, k + 1)], axis=1)
    d[f'futura_{k}'] = np.where(fut.isna().any(axis=1), np.nan,
                                fut.fillna(False).astype(bool).any(axis=1))

te = d[d.conjunto == 'test'].copy()
tramos_te = te.tramo_id.to_numpy()
rng = np.random.default_rng(SEED)


def bootstrap_tramos(df, estadistico, n=N_BOOT):
    """IC 95 % remuestreando TRAMOS enteros: las ventanas de un mismo tramo
    están correlacionadas, y remuestrearlas sueltas achicaría el intervalo."""
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
        filas_prec.append({
            'detector': nom, 'horizonte_min': int(k * MIN_POR_VENTANA),
            'quietas': len(q), 'marcadas': int(m.sum()),
            'p_alarma_si_marcada': round(p1, 4),
            'p_alarma_si_no_marcada': round(p0, 4),
            'riesgo_relativo': round(p1 / p0, 3) if p0 else np.nan,
            'rr_ic_inf': round(lo, 3), 'rr_ic_sup': round(hi, 3),
        })
prec = pd.DataFrame(filas_prec)
print(prec.to_string(index=False))


# =============================================================================
# 5. AUC DE ANTICIPACIÓN — sin umbral
# =============================================================================
# Sobre las ventanas quietas de prueba: ¿el puntaje ordena las que van a tener
# alarma en la próxima ventana por encima de las que no? Es la misma pregunta
# que (4) pero sin depender del P95: un detector puede tener mal calibrado el
# umbral y aun así ordenar bien.
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
# Inicio de episodio: ventana con alarma cuya ventana anterior, en el mismo
# tramo, está quieta. Si la alarma ya está en la ventana 0 del tramo, el
# inicio real puede haber sido antes del hueco de datos: queda fuera
# (censurado a la izquierda) y se cuenta.
#
# Anticipación: cuántas ventanas quietas INMEDIATAMENTE anteriores al inicio
# estaban marcadas sin interrupción. 0 = la ventana justo anterior no estaba
# marcada: no hubo aviso. Por 10 minutos es el adelanto.
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

filas_res = []
if len(eps):
    print(f'\nQuietas previas disponibles (cuánto "antes" se puede ver): mediana '
          f'{eps.quietas_previas.median():.0f} ventanas | '
          f'{100 * (eps.quietas_previas >= 3).mean():.0f} % de los inicios '
          f'tiene 3 o más.')
    print(f'Inicios con una regla TRIP (L, A8, A11): {int(eps.trip.sum())}')
    print('\nPor detector:')
    for nom in DETECTORES:
        a = eps[f'anticip_{nom}_min']
        avisados = a > 0
        lo, hi = bootstrap_tramos(eps, lambda x, nom=nom:
                                  (x[f'anticip_{nom}_min'] > 0).mean())
        # Lo que se esperaría por azar: que la ventana justo anterior esté
        # marcada con la misma frecuencia con que el detector marca CUALQUIER
        # ventana quieta de prueba que NO precede a una alarma.
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
        })
        print(f'  {nom:<16} avisó antes del {100 * avisados.mean():5.1f} % de los '
              f'inicios [IC {100 * lo:.1f}-{100 * hi:.1f}] | por azar '
              f'{100 * azar:.1f} % | adelanto mediano '
              f'{a_av.median() if len(a_av) else float("nan"):.0f} min')

    print('\nPor criticidad de la regla que se enciende (detector principal):')
    _col = f'anticip_{DETECTOR_PRINCIPAL}_min'
    print(eps.groupby('criticidad').agg(
        inicios=(_col, 'size'),
        pct_avisados=(_col, lambda s: round(100 * (s > 0).mean(), 1)),
        adelanto_mediano_min=(_col, lambda s: s[s > 0].median()))
        .to_string())
    print('\nPor componente:')
    print(eps.groupby('componente').agg(
        inicios=(_col, 'size'),
        pct_avisados=(_col, lambda s: round(100 * (s > 0).mean(), 1)),
        adelanto_mediano_min=(_col, lambda s: s[s > 0].median()))
        .sort_values('inicios', ascending=False).to_string())
else:
    print('No hay inicios de episodio observables en prueba.')

resumen = pd.DataFrame(filas_res)


# =============================================================================
# 6b. PARCHE 1 — LÍNEA BASE DE LA DETECCIÓN (no de la anticipación)
# =============================================================================
# El paso 05 reporta AUC = 0,83 en prueba para separar ventanas con alarma de
# ventanas quietas, pero sin línea base: no dice si un detector trivial
# hace lo mismo. Aquí se mide esa misma AUC (todas las ventanas de prueba,
# alarma vs quieta) para los cuatro detectores, y la DIFERENCIA pareada
# AE - z_medias remuestreando los mismos tramos para ambos: con eso no basta
# con mirar si los IC se tocan.
# Va después de la sección 6 a propósito: bootstrap_tramos comparte el
# generador aleatorio, y ponerla antes cambiaría los IC ya reportados.
# Advertencia para la lectura: z_medias parte con ventaja, porque las reglas
# se disparan por niveles que se ven en las medias (misma lógica que la
# restricción 2 de la Fase B). Si empata, no significa que el AE sobre.
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


# =============================================================================
# 7. GUARDAR Y LEER
# =============================================================================
print('\n' + '=' * 72)
print('7. GUARDANDO')
print('=' * 72)
for _df, _f in ((prec, SALIDA_PRECURSORES), (eps, SALIDA_EPISODIOS),
                (resumen, SALIDA_RESUMEN)):
    _df.to_csv(_f, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'Guardado: {_f}')

print('\n' + '=' * 72)
print(f'RESUMEN — detector principal: {DETECTOR_PRINCIPAL}')
print('=' * 72)
if len(resumen):
    r = resumen.set_index('detector')
    p = r.loc[DETECTOR_PRINCIPAL]
    pr = prec[(prec.detector == DETECTOR_PRINCIPAL) & (prec.horizonte_min == 10)]
    if len(pr):
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
    else:
        print('\nEl AUC de anticipación no se separa de 0,5: con estos datos el')
        print('autoencoder DETECTA pero no ANTICIPA. Hay que decirlo así.')

print('\n' + '=' * 72)
print('LISTO.')
print('=' * 72)
