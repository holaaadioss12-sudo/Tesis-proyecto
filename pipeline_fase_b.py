# -*- coding: utf-8 -*-
"""
=============================================================================
FASE B — CLASIFICACIÓN DE LAS VENTANAS QUE EL AUTOENCODER MARCA COMO ANÓMALAS
=============================================================================
Entradas: CR010_ventanas_W60.csv              (paso 02: reglas, banda_w, w_*)
          CR010_features_W60.csv              (paso 02: 70 estadísticos)
          CR010_error_reconstruccion_W60.csv  (paso 05: error por ventana)
          CR010_corte_split.json              (paso 01: el corte único)
          criticidad.py, diagnosticos.py      (misma carpeta de Drive)
Opcionales, sólo para cotejar:
          CR010_ventanas_criticidad_W60.csv   (paso 02b)
          05b_criterios_umbral.csv            (paso 05b)

Salidas:  B1_resultados.csv
          B2_resultados.csv
          B2_variable_causante.csv

-----------------------------------------------------------------------------
QUÉ ES LA FASE B — la definición del profesor, en dos reuniones
-----------------------------------------------------------------------------
25-sep: un Random Forest entrenado SOLO sobre las ventanas que el autoencoder
        marca como no normales, que las clasifica en leve / media / grave según
        la distancia de Wasserstein. El autoencoder queda como filtro.
1-oct:  Random Forest o XGBoost sobre las anómalas, para asociarlas a un tipo
        de falla y determinar qué variable origina la condición.

Son dos clasificadores:
  B1 — gravedad                        (banda_w, y la escala del experto)
  B2 — tipo de falla + variable causante

-----------------------------------------------------------------------------
LAS DOS TRAMPAS QUE ESTE SCRIPT EXISTE PARA NO PISAR
-----------------------------------------------------------------------------
(1) banda_w es un corte DETERMINISTA sobre w_max: sale de
    np.digitize(indice.max(axis=1), cortes_agregado) en
    indice_w.resumen_por_ventana. Un modelo que vea w_max, w_medio, cualquier
    w_<sensor> o las columnas de nivel predice banda_w al ~100 % y no
    significa nada. Esas columnas están en PROHIBIDAS_B1 y un assert revienta
    si alguna llega a una matriz de entrenamiento.

(2) El componente se deriva de qué REGLA está activa, y las reglas son
    umbrales sobre los máximos y mínimos de la ventana. Es el razonamiento del
    PARCHE 3 del paso 03: un árbol que corta T1_max en 50 reproduce la regla K
    exacta. Por eso B2 se entrena con SIN_EXTREMOS -- media, desviación y
    delta, ninguna de las cuales es un cruce de umbral -- y el conjunto
    completo se reporta al lado sólo como contraste.

-----------------------------------------------------------------------------
EL SPLIT DE ESTA FASE, Y POR QUÉ NO ES 70 / 15 / 15
-----------------------------------------------------------------------------
El paso 05 calcula el error de reconstrucción sólo sobre validación y prueba:
las ventanas de entrenamiento no lo tienen, y si lo tuvieran estaría sesgado
a la baja, porque el autoencoder las vio. Sin error no hay filtro, y sin
filtro una ventana no puede entrar a la Fase B tal como la definió el
profesor. Entonces:

    umbral del filtro   se ajusta con las NORMALES de VALIDACIÓN
    clasificadores B    se entrenan con las anómalas de VALIDACIÓN
    todo lo reportado   se mide UNA vez sobre PRUEBA, que no se tocó

No hay ajuste de hiperparámetros: no queda un tercer conjunto donde hacerlo
sin tocar prueba. Las configuraciones son fijas y razonables, y van
declaradas así.

REQUIERE: scikit-learn y xgboost (vienen en Colab). Corre en CPU.
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
    # Fuera de Colab (por ejemplo, para probarlo con datos sintéticos) la
    # carpeta se pasa por variable de entorno.
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

if RUTA_BASE not in sys.path:
    sys.path.append(RUTA_BASE)

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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             confusion_matrix, cohen_kappa_score,
                             recall_score)
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

import criticidad
from diagnosticos import degeneracion_criticidad

try:
    from xgboost import XGBClassifier
    HAY_XGB = True
except ImportError:
    HAY_XGB = False

SEED = 42
np.random.seed(SEED)

W = 60
PASO_S = 10

ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
ARCHIVO_CRIT_02B = f'{RUTA_EQ}/{EQUIPO}_ventanas_criticidad_W{W}.csv'
ARCHIVO_05B = f'{RUTA_EQ}/05b_criterios_umbral.csv'
SALIDA_B1 = f'{RUTA_EQ}/B1_resultados.csv'
SALIDA_B2 = f'{RUTA_EQ}/B2_resultados.csv'
SALIDA_CAUSANTE = f'{RUTA_EQ}/B2_variable_causante.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']


# =============================================================================
# 0. CONSTANTES QUE DECIDEN — todas acá arriba, ninguna enterrada en el código
# =============================================================================

# QUÉ CHECKPOINT DEL AUTOENCODER HACE DE FILTRO.
# El paso 05 guarda las dos columnas, err_AE_normales (mínima pérdida en las
# normales de validación) y err_AE_normales_sep (máxima razón anómala/normal
# en validación), y hoy usa la segunda. Está en discusión: el 04d midió que
# el error en normales predice el AUC con rho = -1,000 y la separación no lo
# predice (rho = -0,25, p = 0,52), y el 04e es el que lo decide sobre prueba.
# Cambiar el filtro es cambiar esta línea y nada más.
MODELO_AE = 'AE_normales_sep'

# Criterios del umbral del filtro. Los dos que nombró el profesor, y la
# variante robusta por MAD, que entra sólo si la asimetría del error de las
# normales pasa de 1 (mismo criterio que el 05b).
CRITERIO_PRINCIPAL = 'percentil 95'
ASIMETRIA_PARA_MAD = 1.0

# Una clase con menos ventanas que esto en ENTRENAMIENTO no tiene métricas que
# signifiquen algo. En B2 se agrupa en 'otros' antes de medir nada.
MIN_CLASE = 30

# El chequeo del split. El JSON tiene que traer un corte cerca del 70 %. Ya se
# entrenó una vez con el 18 % sin que nada avisara (ver paso 01, CORTE
# DEFINITIVO), y la Fase B heredaría ese error entero. Si el corte está
# desviado el script se detiene; esta constante existe sólo para poder correr
# una prueba a sabiendas, nunca para los números de la memoria.
PERMITIR_SPLIT_DESVIADO = False

# RESTRICCIÓN 1 -- el índice W y todo lo que se deriva de él. banda_w es
# np.digitize sobre w_max: cualquiera de estas en la entrada lo filtra.
PROHIBIDAS_B1 = [
    'w_CM', 'w_PI', 'w_PDF', 'w_PEL', 'w_T7', 'w_T8', 'w_T9', 'w_T1', 'w_T2',
    'w_T5', 'w_V1', 'w_V2', 'w_V3', 'w_V4',
    'w_max', 'w_medio', 'nivel_ventana', 'nivel_sensor_max',
    'n_sensores_sobre_P80', 'banda_w', 'sensor_origen',
]

# Lo que sale de las reglas o de las etiquetas tampoco puede entrar: es la
# misma lista NO_FEATURE del paso 03, más los objetivos del paso 03 y las
# columnas que agrega esta fase.
NO_FEATURE = ['tramo_id', 'ventana_id', 'timestamp_inicio', 'timestamp_fin',
              'etiqueta_ventana', 'etiqueta_tfidf', 'banda_severidad',
              'total_alarmas_ventana', 'total_eventos_ventana',
              'n_reglas_activas', 'concentracion']

ETIQUETAS_BANDA_W = ['1_leve', '2_media', '3_grave']      # 0_normal no entra


def verificar_matriz(columnas, donde):
    """El assert de las dos restricciones. Revienta, no avisa."""
    columnas = list(columnas)
    fuga_w = sorted(set(columnas) & set(PROHIBIDAS_B1))
    assert not fuga_w, (f'[{donde}] columnas del índice W en la entrada: '
                        f'{fuga_w}. banda_w es un corte sobre w_max: con '
                        f'esto el modelo acierta por construcción.')
    reglas = set(criticidad.CRITICIDAD)
    fuga_r = sorted(set(columnas) & (reglas | set(NO_FEATURE)))
    assert not fuga_r, (f'[{donde}] columnas de reglas o etiquetas en la '
                        f'entrada: {fuga_r}. Las etiquetas salen de ahí.')
    etq = [c for c in columnas if c.startswith(('etiqueta', 'criticidad',
                                                'componente', 'banda'))]
    assert not etq, f'[{donde}] columnas de etiqueta en la entrada: {etq}'


# =============================================================================
# 1. CARGA, Y EL CORTE ÚNICO DEL JSON
# =============================================================================
print('=' * 72)
print(f'FASE B — CLASIFICACIÓN DE LAS ANÓMALAS (W = {W})')
print('=' * 72)
print(f'Filtro: checkpoint {MODELO_AE} | criterio principal: {CRITERIO_PRINCIPAL}')
print(f'XGBoost disponible: {"sí" if HAY_XGB else "NO -- se corre sólo RF"}')

corte = json.load(open(ARCHIVO_CORTE))
T_TRAIN, T_VAL = pd.Timestamp(corte['T_train']), pd.Timestamp(corte['T_val'])
print(f'\nCorte único: T_train = {T_TRAIN} | T_val = {T_VAL}')
_f_tr = corte.get('frac_train_efectiva')
print(f'  frac_train_efectiva = {_f_tr}   ajustado_en: {corte.get("ajustado_en")}')
if _f_tr is None or abs(_f_tr - corte.get('frac_train', 0.70)) > 0.05:
    print('\n' + '!' * 72)
    print('EL CORTE DEL JSON NO ESTÁ DONDE DEBE (o no dice dónde está).')
    print(f'frac_train_efectiva = {_f_tr}, objetivo {corte.get("frac_train")}.')
    print('Con ese corte la Fase A entrenó con una fracción equivocada del')
    print('registro, y la Fase B heredaría el error entero sin que nada lo')
    print('delate. Corré 01 -> 05 con el pipeline parchado antes de seguir.')
    print('!' * 72)
    if not PERMITIR_SPLIT_DESVIADO:
        raise SystemExit('Corte desviado, ver arriba.')
    print('PERMITIR_SPLIT_DESVIADO = True: se sigue SÓLO como prueba.')


def split_df_por_fecha(d, col='timestamp_inicio', t_train=None, t_val=None):
    """COPIA LITERAL de split_df_por_fecha del paso 02 (PARCHE 2).

    No se importa porque el paso 02 es una celda de Colab, no un módulo: un
    import ejecutaría el paso entero. Se copia sin cambios, salvo que los dos
    instantes se pasan explícitos en vez de leerse de variables globales.
    """
    t_train = T_TRAIN if t_train is None else t_train
    t_val = T_VAL if t_val is None else t_val
    ts = pd.to_datetime(d[col])
    return (d[ts <= t_train], d[(ts > t_train) & (ts <= t_val)], d[ts > t_val])


ventanas = pd.read_csv(ARCHIVO_VENTANAS, parse_dates=['timestamp_inicio',
                                                      'timestamp_fin'])
features = pd.read_csv(ARCHIVO_FEATURES)
err = pd.read_csv(ARCHIVO_ERROR)
print(f'\nVentanas: {len(ventanas):,} | features: {len(features):,} | '
      f'con error de reconstrucción: {len(err):,}')

COL_ERR = f'err_{MODELO_AE}'
if COL_ERR not in err.columns:
    raise SystemExit(f'El archivo del paso 05 no trae {COL_ERR}. Columnas: '
                     f'{list(err.columns)}')

ids_reglas = [c for c in ventanas.columns if c in criticidad.CRITICIDAD]
print(f'Reglas en el archivo de ventanas: {len(ids_reglas)}   <- esperado: 31')


# =============================================================================
# 2. PASO 0 — ¿LA ESCALA EXPERTA TIENE ALGO QUE CLASIFICAR?
# =============================================================================
# diagnosticos.degeneracion_criticidad() dice en su docstring "hay que correr
# esto ANTES de construir la Fase B". Se corre acá, sobre la criticidad que
# calcula criticidad.anotar_ventanas() -- la misma tabla del profesor.
#
# POR QUÉ NO SE LEE CR010_ventanas_criticidad_W60.csv. El 02b toma el
# componente de la columna 'componente' de criticidad_reglas.csv, cuyos
# valores son 'Sistema', 'Motor', 'Alimentacion/motor'..., NO la taxonomía de
# criticidad.COMPONENTE (motor, lubricacion, estructura...). 'Sistema', en
# particular, junta toda la lubricación bajo un nombre que no dice nada. Acá
# la etiqueta se calcula con criticidad.COMPONENTE, y si el archivo del 02b
# existe se coteja la criticidad contra él.
# =============================================================================
print('\n' + '=' * 72)
print('2. PASO 0 — DEGENERACIÓN DE LA ESCALA EXPERTA')
print('=' * 72)

criticidad.test_equivalencia(verbose=False)
v_anot = criticidad.anotar_ventanas(ventanas, ids_reglas)
print()
degeneracion_criticidad(v_anot, col='criticidad_min', min_por_clase=MIN_CLASE)

# --- el componente, con el desempate del 02b -----------------------------
# anotar_ventanas() elige la regla más grave y, si empatan, la primera en el
# orden de columnas -- que es arbitrario. Con D y A4 (las dos de criticidad 2,
# lubricación y motor) activas en más de un tercio de las ventanas cada una,
# ese empate decide la etiqueta de miles de ventanas. Se desempata como el
# 02b: entre las más graves, la que tiene más muestras en alarma dentro de la
# ventana; recién después, el orden de columnas.
_M = ventanas[ids_reglas].to_numpy(dtype=float)
_peso = np.array([criticidad.PESO[r] for r in ids_reglas], dtype=float)
_activa = _M > 0
_peso_act = np.where(_activa, _peso[None, :], -1)
_peso_top = _peso_act.max(axis=1)
_es_top = _activa & (_peso_act == _peso_top[:, None])
_clave = np.where(_es_top, _M, -1.0)                 # muestras, sólo en el top
_dom = _clave.argmax(axis=1)
_hay = _activa.any(axis=1)
_comp_col = np.array([criticidad.COMPONENTE[r] for r in ids_reglas])
v_anot['regla_dominante_b'] = np.where(_hay, np.array(ids_reglas)[_dom], 'ninguna')
v_anot['componente_b'] = np.where(_hay, _comp_col[_dom], 'normal')

# Cuán ambigua es la etiqueta: ventanas donde las reglas del nivel más grave
# apuntan a más de un componente. Ahí el componente lo decide el desempate,
# no la física, y ese porcentaje acota lo que B2 puede afirmar.
_n_comp_top = np.array([len(set(_comp_col[fila])) for fila in _es_top])
_ambiguas = _hay & (_n_comp_top > 1)
print(f'\nComponente (taxonomía criticidad.COMPONENTE, desempate por muestras):')
print(f'  ventanas con alguna regla: {int(_hay.sum()):,}')
print(f'  ambiguas -- el nivel más grave toca >1 componente: '
      f'{int(_ambiguas.sum()):,} ({100*_ambiguas.sum()/max(_hay.sum(), 1):.1f} %)')
print(f'  coincide con el desempate por orden de columnas: '
      f'{100*(v_anot.componente_b == v_anot.componente)[_hay].mean():.1f} %')
print('  El porcentaje de ambiguas va a la memoria: es la fracción de las')
print('  etiquetas de B2 que dependen de una regla de desempate y no de qué')
print('  componente está comprometido.')

if os.path.exists(ARCHIVO_CRIT_02B):
    _c02b = pd.read_csv(ARCHIVO_CRIT_02B,
                        usecols=['tramo_id', 'ventana_id', 'criticidad_max'])
    _x = v_anot[['tramo_id', 'ventana_id', 'criticidad_min']].merge(
        _c02b, on=['tramo_id', 'ventana_id'])
    _iguales = (_x.criticidad_min.replace(4, 0) == _x.criticidad_max).mean()
    print(f'\nCotejo con el 02b: criticidad igual en {100*_iguales:.2f} % de '
          f'{len(_x):,} ventanas (4 de criticidad.py = 0 del 02b).')
else:
    print(f'\n(No está {os.path.basename(ARCHIVO_CRIT_02B)}: no hay cotejo con el 02b.)')


# =============================================================================
# 3. EL UNIVERSO: VENTANAS CON ERROR, FEATURES, Y LAS VERIFICACIONES DURAS
# =============================================================================
print('\n' + '=' * 72)
print('3. UNIVERSO DE LA FASE B')
print('=' * 72)

LLAVES = ['tramo_id', 'ventana_id']
cols_v = (LLAVES + ['timestamp_inicio', 'timestamp_fin', 'banda_w',
                    'sensor_origen', 'banda_severidad', 'n_reglas_activas',
                    'criticidad_min', 'banda_criticidad', 'componente_b',
                    'regla_dominante_b'])
cols_v = [c for c in cols_v if c in v_anot.columns]

# --- las features, con la construcción del paso 03 ------------------------
# FEATURES y FEATURES_SIN_EXTREMOS se arman con EXACTAMENTE la misma expresión
# del paso 03 (bloques 1 y 3, PARCHE 3). No se importan por la misma razón que
# split_df_por_fecha: el paso 03 es una celda.
FEATURES = [c for c in features.columns if c not in NO_FEATURE]
FEATURES_SIN_EXTREMOS = [f for f in FEATURES
                         if not f.endswith(('_max', '_min'))]
assert len(FEATURES) == 70, f'{len(FEATURES)} features, esperaba 70 (14 x 5)'
assert len(FEATURES_SIN_EXTREMOS) == 42, \
    f'{len(FEATURES_SIN_EXTREMOS)} sin extremos, esperaba 42 (14 x 3)'

d = (err[LLAVES + ['conjunto', COL_ERR]]
     .merge(v_anot[cols_v], on=LLAVES, how='inner', validate='one_to_one')
     .merge(features[LLAVES + FEATURES], on=LLAVES, how='inner',
            validate='one_to_one'))
d = d.rename(columns={COL_ERR: 'error_ae'})
assert len(d) == len(err), (f'El cruce perdió ventanas: {len(err):,} con error '
                            f'y {len(d):,} cruzadas. Revisá tramo_id/ventana_id.')

# --- el split, por fecha, y su coherencia con el paso 05 --------------------
d_tr, d_va, d_te = split_df_por_fecha(d)
print(f'Por fecha: train {len(d_tr):,} | val {len(d_va):,} | test {len(d_te):,}')
assert len(d_tr) == 0, ('Hay ventanas de entrenamiento con error de '
                        'reconstrucción: el paso 05 no debería calcularlo ahí.')
assert (d_va.conjunto == 'val').all() and (d_te.conjunto == 'test').all(), \
    'La columna conjunto del paso 05 no coincide con el corte del JSON.'
print('  Coincide con la columna "conjunto" del paso 05 ventana por ventana.')

# --- sin solape: stride = W ----------------------------------------------------
# Dentro de cada tramo, ventana_id consecutivas tienen que estar exactamente a
# W muestras una de otra, y cada ventana termina antes de que empiece la
# siguiente. Si alguna se solapara, la misma muestra estaría en dos ventanas y
# una podría quedar en entrenamiento y la otra en prueba.
_o = d.sort_values(LLAVES)
_mismo = _o.tramo_id.eq(_o.tramo_id.shift())
_paso = (_o.timestamp_inicio - _o.timestamp_inicio.shift())[_mismo]
_salto_id = (_o.ventana_id - _o.ventana_id.shift())[_mismo]
_paso_1 = _paso[_salto_id == 1]
assert (_paso_1 == pd.Timedelta(seconds=W * PASO_S)).all(), \
    'Ventanas consecutivas no están a W muestras: hay solape o hueco.'
assert (_o.timestamp_fin.shift()[_mismo] < _o.timestamp_inicio[_mismo]).all(), \
    'Hay una ventana que empieza antes de que termine la anterior.'
print(f'  Sin solape: {len(_paso_1):,} pares consecutivos a exactamente '
      f'{W * PASO_S} s.')

CONJUNTOS_B1 = {
    'SOLO_SENSORES': FEATURES,
    'SIN_EXTREMOS': FEATURES_SIN_EXTREMOS,
    'MAS_ERROR': FEATURES + ['error_ae'],
}
CONJUNTOS_B2 = {
    'SIN_EXTREMOS': FEATURES_SIN_EXTREMOS,      # el que se defiende
    'SOLO_SENSORES': FEATURES,                  # sólo como contraste
}
for _n, _c in {**CONJUNTOS_B1, **{f'B2_{k}': v for k, v in CONJUNTOS_B2.items()}}.items():
    verificar_matriz(_c, _n)
print('  Las listas de features pasan el assert de PROHIBIDAS_B1 y de reglas.')


# =============================================================================
# 4. EL FILTRO: QUÉ VENTANAS MARCA EL AUTOENCODER
# =============================================================================
# Los umbrales se ajustan con las ventanas NORMALES de VALIDACIÓN (sin ninguna
# regla activa), con las fórmulas del 05b. El 05b los calcula sobre val + test
# mezclados; acá sólo con val, para que prueba quede limpia. Si el 05b está en
# Drive se imprime al lado para que la diferencia quede a la vista.
# =============================================================================
print('\n' + '=' * 72)
print('4. EL FILTRO DEL AUTOENCODER')
print('=' * 72)

_nor_va = d_va.loc[d_va.n_reglas_activas == 0, 'error_ae']
_media, _desv = _nor_va.mean(), _nor_va.std()
_mediana = _nor_va.median()
_mad = 1.4826 * (_nor_va - _mediana).abs().median()
_asim = _nor_va.skew()
print(f'Error en las {len(_nor_va):,} normales de validación: media {_media:.5f} | '
      f'mediana {_mediana:.5f} | sd {_desv:.5f} | asimetría {_asim:.2f}')

UMBRALES = {'percentil 95': float(np.percentile(_nor_va, 95)),
            'media + 3 sigma': float(_media + 3 * _desv)}
if _asim > ASIMETRIA_PARA_MAD:
    UMBRALES['mediana + 3 MAD'] = float(_mediana + 3 * _mad)
    print(f'  Asimetría > {ASIMETRIA_PARA_MAD}: se agrega la variante robusta.')
assert CRITERIO_PRINCIPAL in UMBRALES

if os.path.exists(ARCHIVO_05B):
    _t05 = pd.read_csv(ARCHIVO_05B, sep=';', decimal=',')
    _p95 = _t05.loc[_t05.criterio == 'percentil 95', 'umbral']
    if len(_p95):
        print(f'  05b (val + test mezclados): P95 = {float(_p95.iloc[0]):.5f} | '
              f'acá (sólo val): {UMBRALES["percentil 95"]:.5f}')


def graves_perdidas(dd, marcada):
    """Cuántas de las graves NO pasan el filtro, por cada escala de gravedad.

    Una grave que el filtro no marca nunca llega a la Fase B, y ese error no
    se recupera. Todo el desempeño de B se lee condicionado a esto.
    """
    out = {}
    for nombre, es_grave in [('banda_w_3_grave', dd.banda_w == '3_grave'),
                             ('criticidad_1', dd.criticidad_min == 1),
                             ('banda_severidad_3_grave',
                              dd.banda_severidad == '3_grave')]:
        n = int(es_grave.sum())
        perd = int((es_grave & ~marcada).sum())
        out[f'graves_{nombre}'] = n
        out[f'pct_perdidas_{nombre}'] = round(100 * perd / max(n, 1), 1)
    return out


filtro = []
for crit, u in UMBRALES.items():
    m_va, m_te = d_va.error_ae > u, d_te.error_ae > u
    fila = {'criterio': crit, 'umbral': round(u, 6),
            'percentil_equiv_val': round(100 * (_nor_va <= u).mean(), 1),
            'marcadas_val': int(m_va.sum()), 'marcadas_test': int(m_te.sum()),
            'falsas_alarmas_test_pct': round(
                100 * (m_te & (d_te.n_reglas_activas == 0)).sum()
                / max((d_te.n_reglas_activas == 0).sum(), 1), 1)}
    fila.update(graves_perdidas(d_te, m_te))
    filtro.append(fila)
filtro = pd.DataFrame(filtro)
print('\nPor criterio, medido sobre PRUEBA:')
print(filtro.to_string(index=False))
print('\npct_perdidas_* es la fracción de las graves de cada escala que el')
print('filtro deja afuera. Va al lado de cada métrica de B: un F1 alto sobre')
print('lo que pasó el filtro no dice nada de lo que el filtro no dejó pasar.')


# =============================================================================
# 5. DOS ESCALAS DE GRAVEDAD: LA EMPÍRICA Y LA DEL EXPERTO
# =============================================================================
# banda_w sale de los datos (Wasserstein sobre los sensores). criticidad sale
# del juicio de una sola persona, no contrastado con operadores. El acuerdo
# entre las dos es lo que compensa esa limitación; el desacuerdo también es un
# resultado. comparar_escalas() de criticidad.py hace la matriz contra
# banda_severidad; acá se agrega la que pidió el profesor, contra banda_w.
# =============================================================================
print('\n' + '=' * 72)
print('5. ESCALA EMPÍRICA (banda_w) CONTRA ESCALA EXPERTA (criticidad)')
print('=' * 72)
print('Sobre todas las ventanas -- banda_severidad vs banda_criticidad '
      '(criticidad.comparar_escalas):')
print(criticidad.comparar_escalas(v_anot).to_string())
print('\nbanda_w vs banda_criticidad:')
print(pd.crosstab(v_anot.banda_w, v_anot.banda_criticidad).to_string())
_ambas = v_anot[(v_anot.banda_w != '0_normal') & (v_anot.criticidad_min < 4)]
if len(_ambas) > 30:
    _kw = cohen_kappa_score(
        _ambas.banda_w.map({'1_leve': 0, '2_media': 1, '3_grave': 2}),
        _ambas.criticidad_min.map({3: 0, 2: 1, 1: 2}), weights='linear')
    print(f'\nKappa ponderado (lineal) sobre las {len(_ambas):,} ventanas '
          f'anómalas en las dos escalas: {_kw:.3f}')
    print('  0 = acuerdo de azar, 1 = acuerdo perfecto. Va a la memoria tal cual.')


# =============================================================================
# HERRAMIENTAS COMUNES A B1 Y B2
# =============================================================================
def metricas(y, p, clases, orden=None):
    """F1 macro y exactitud balanceada; para escalas ordenadas, error ordinal.

    No se reporta la exactitud sola: con clases desbalanceadas, predecir
    siempre la mayoritaria da una exactitud alta y no sabe nada.
    """
    out = {'f1_macro': f1_score(y, p, labels=clases, average='macro',
                                zero_division=0),
           'bal_acc': balanced_accuracy_score(y, p)}
    if orden is not None:
        pos = {c: i for i, c in enumerate(orden)}
        out['err_ordinal'] = float(np.mean(np.abs(
            np.array([pos[x] for x in p]) - np.array([pos[x] for x in y]))))
    return out


def baseline_umbral(e_tr, y_tr, e_te, orden=None):
    """Un solo umbral sobre el error de reconstrucción, sin modelo.

    Ordinal (B1): más error = más grave. Los cortes se ponen en los cuantiles
    del error de entrenamiento que reproducen las proporciones de clase de
    entrenamiento -- con dos clases, es un solo umbral. Nominal (B2): el error
    se parte en tantos tramos como clases y cada tramo predice la clase más
    frecuente que tuvo en entrenamiento.
    """
    e_tr, e_te, y_tr = np.asarray(e_tr), np.asarray(e_te), np.asarray(y_tr)
    if orden is not None:
        props = np.array([(y_tr == c).mean() for c in orden])
        cortes = np.quantile(e_tr, np.cumsum(props)[:-1])
        return np.array(orden)[np.digitize(e_te, cortes)]
    k = len(np.unique(y_tr))
    cortes = np.unique(np.quantile(e_tr, np.linspace(0, 1, k + 1)[1:-1]))
    b_tr, b_te = np.digitize(e_tr, cortes), np.digitize(e_te, cortes)
    moda_global = pd.Series(y_tr).mode().iloc[0]
    mapa = {b: pd.Series(y_tr[b_tr == b]).mode().iloc[0]
            for b in np.unique(b_tr)}
    return np.array([mapa.get(b, moda_global) for b in b_te])


def fusionar_escasas(y_tr, orden):
    """Junta cada clase con menos de MIN_CLASE ventanas en entrenamiento con
    su vecina en la escala, la más poblada de las dos. Repite hasta que no
    quede ninguna escasa o queden dos clases.

    Una clase con dos ventanas de entrenamiento no se aprende, y su F1 entra
    con el mismo peso que las otras en el F1 macro: lo hunde o lo infla por
    azar. Se junta con la vecina -- no se borra -- para que toda ventana
    marcada siga recibiendo una gravedad, y el nombre de la clase fusionada
    dice qué se juntó.
    """
    grupos = [[c] for c in orden]
    while len(grupos) > 2:
        n = [int(np.isin(y_tr, g).sum()) for g in grupos]
        escasas = [i for i, k in enumerate(n) if k < MIN_CLASE]
        if not escasas:
            break
        i = escasas[0]
        vec = [j for j in (i - 1, i + 1) if 0 <= j < len(grupos)]
        j = max(vec, key=lambda k: n[k])
        a, b = sorted((i, j))
        grupos[a:b + 1] = [grupos[a] + grupos[b]]
    mapa = {c: '+'.join(g) for g in grupos for c in g}
    return mapa, ['+'.join(g) for g in grupos]


class XGBConEtiquetas:
    """XGBoost que recibe y devuelve las etiquetas de texto, como el RF.

    XGBoost exige etiquetas enteras 0..k-1, todas presentes en entrenamiento,
    y no acepta class_weight. Esto codifica con las clases que SÍ están en
    entrenamiento, pesa las muestras como class_weight='balanced' del RF, y
    decodifica al predecir. Así correr_tarea() y permutation_importance() lo
    usan igual que al RF, sin un caso especial para cada uno.
    """
    _estimator_type = 'classifier'

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        cod = {c: i for i, c in enumerate(self.classes_)}
        self.modelo_ = XGBClassifier(**self.params).fit(
            X, np.array([cod[c] for c in y]),
            sample_weight=compute_sample_weight('balanced', y))
        return self

    def predict(self, X):
        return self.classes_[self.modelo_.predict(X).astype(int)]


def modelos():
    """Configuraciones FIJAS. No se ajustan: no hay dónde sin tocar prueba."""
    out = {'RF': RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                        class_weight='balanced',
                                        random_state=SEED, n_jobs=-1)}
    if HAY_XGB:
        out['XGB'] = XGBConEtiquetas(n_estimators=300, max_depth=4,
                                     learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, random_state=SEED,
                                     n_jobs=-1, eval_metric='mlogloss')
    return out


def correr_tarea(tarea, criterio, X_por_conjunto, y_tr, y_te, e_tr, e_te,
                 clases, orden=None, imprimir_cm=True):
    """Baselines + cada modelo sobre cada conjunto de features. Una fila por
    combinación, todas medidas sobre prueba."""
    filas, preds = [], {}
    base = {'n_train': len(y_tr), 'n_test': len(y_te), 'n_clases': len(clases),
            'tarea': tarea, 'criterio_umbral': criterio}

    # --- las tres líneas base obligatorias -------------------------------------
    _Xd = np.zeros((len(y_tr), 1))
    for nombre, pred in [
            ('base_mayoritaria',
             DummyClassifier(strategy='most_frequent').fit(_Xd, y_tr)
             .predict(np.zeros((len(y_te), 1)))),
            ('base_estratificada',
             DummyClassifier(strategy='stratified', random_state=SEED)
             .fit(_Xd, y_tr).predict(np.zeros((len(y_te), 1)))),
            ('base_umbral_error', baseline_umbral(e_tr, y_tr, e_te, orden))]:
        filas.append({**base, 'features': '-', 'n_features': 0,
                      'modelo': nombre, **metricas(y_te, pred, clases, orden)})
        preds[('-', nombre)] = pred

    # --- los modelos -------------------------------------------------------------
    for conj, (X_tr, X_te) in X_por_conjunto.items():
        verificar_matriz(X_tr.columns, f'{tarea}/{conj}')
        for nom_m, m in modelos().items():
            pred = m.fit(X_tr, y_tr).predict(X_te)
            filas.append({**base, 'features': conj, 'n_features': X_tr.shape[1],
                          'modelo': nom_m, **metricas(y_te, pred, clases, orden)})
            preds[(conj, nom_m)] = pred

    t = pd.DataFrame(filas)
    cols = ['features', 'modelo', 'f1_macro', 'bal_acc'] + \
        (['err_ordinal'] if orden is not None else [])
    print(t[cols].round(4).to_string(index=False))

    if imprimir_cm:
        mej = t[t.modelo.isin(['RF', 'XGB'])].sort_values('f1_macro').iloc[-1]
        print(f'\n  Matriz de confusión del mejor ({mej.features} / {mej.modelo}),'
              f' filas = real, columnas = predicho:')
        cm = confusion_matrix(y_te, preds[(mej.features, mej.modelo)],
                              labels=clases)
        print(pd.DataFrame(cm, index=clases, columns=clases).to_string())
    return t, preds


def ic_diferencia(y, p_mod, p_base, grupos, clases, n_boot=1000):
    """IC 95 % de F1(modelo) - F1(baseline), con bootstrap POR TRAMO.

    Remuestrear ventanas sueltas subestimaría la incertidumbre: las ventanas
    de un mismo tramo están correlacionadas. Se remuestrean tramos enteros.
    """
    rng = np.random.default_rng(SEED)
    y, p_mod, p_base = map(np.asarray, (y, p_mod, p_base))
    grupos = np.asarray(grupos)
    unicos = np.unique(grupos)
    idx_g = {g: np.flatnonzero(grupos == g) for g in unicos}
    difs = []
    for _ in range(n_boot):
        sel = np.concatenate([idx_g[g] for g in rng.choice(unicos, len(unicos))])
        difs.append(f1_score(y[sel], p_mod[sel], labels=clases, average='macro',
                             zero_division=0)
                    - f1_score(y[sel], p_base[sel], labels=clases,
                               average='macro', zero_division=0))
    return np.percentile(difs, [2.5, 97.5])


# =============================================================================
# 6. B1 — GRAVEDAD
# =============================================================================
# Dos objetivos, entrenados por separado:
#   banda_w        {1_leve, 2_media, 3_grave}: la distancia de Wasserstein que
#                  pidió el profesor el 25-sep. Las 0_normal no entran.
#   criticidad     la escala del experto. Si el paso 0 deja una clase con menos
#                  de MIN_CLASE ventanas, se junta con la vecina y el problema
#                  se plantea con las clases que sí existen -- en la corrida
#                  anterior eso era binario: grave / media.
#
# La pregunta que sí vale: ¿se lee la gravedad desde los estadísticos crudos
# de los sensores, sin ver el índice W? Si sí, el método es desplegable sin
# calcular Wasserstein. Si no, también es resultado.
# =============================================================================
print('\n' + '=' * 72)
print('6. B1 — GRAVEDAD')
print('=' * 72)

res_b1, preds_b1 = [], {}
for crit, u in UMBRALES.items():
    a_va = d_va[d_va.error_ae > u]
    a_te = d_te[d_te.error_ae > u]
    print(f'\n######## criterio del filtro: {crit} (umbral {u:.5f}) ########')
    print(f'Marcadas por el autoencoder: val {len(a_va):,} | test {len(a_te):,}')

    # --- B1a: banda_w --------------------------------------------------------
    tr = a_va[a_va.banda_w != '0_normal'].copy()
    te = a_te[a_te.banda_w != '0_normal'].copy()
    print(f'\n--- B1a: banda_w ---')
    print(f'  marcadas que el índice W llama 0_normal (quedan fuera): '
          f'val {len(a_va) - len(tr):,} | test {len(a_te) - len(te):,}')
    print('  Reparto (val / test):')
    print(pd.DataFrame({'val': tr.banda_w.value_counts(),
                        'test': te.banda_w.value_counts()})
          .reindex(ETIQUETAS_BANDA_W).fillna(0).astype(int).to_string())
    _mapa, orden_w = fusionar_escasas(tr.banda_w.to_numpy(), ETIQUETAS_BANDA_W)
    if len(orden_w) < len(ETIQUETAS_BANDA_W):
        print(f'  Clases con < {MIN_CLASE} ventanas en entrenamiento, fusionadas '
              f'con su vecina: {orden_w}')
    tr['y'], te['y'] = tr.banda_w.map(_mapa), te.banda_w.map(_mapa)
    X = {k: (tr[v], te[v]) for k, v in CONJUNTOS_B1.items()}
    t, p = correr_tarea('B1_banda_w', crit, X, tr.y.to_numpy(),
                        te.y.to_numpy(), tr.error_ae, te.error_ae,
                        orden_w, orden=orden_w,
                        imprimir_cm=(crit == CRITERIO_PRINCIPAL))
    res_b1.append(t)
    preds_b1[(crit, 'B1_banda_w')] = (p, te, orden_w)

    # --- B1b: criticidad del experto ----------------------------------------
    tr = a_va[a_va.criticidad_min < 4].copy()
    te = a_te[a_te.criticidad_min < 4].copy()
    print(f'\n--- B1b: criticidad del experto ---')
    print(f'  marcadas sin ninguna regla activa (quedan fuera): '
          f'val {len(a_va) - len(tr):,} | test {len(a_te) - len(te):,}')
    # 1 = más grave en la tabla; acá se ordena de menos a más grave para que
    # el error ordinal tenga sentido. Una clase escasa se junta con su vecina
    # -- en la corrida anterior la 3 quedaba vacía y B1b era BINARIO.
    _nom = {3: '3_leve', 2: '2_media', 1: '1_grave'}
    tr['c'], te['c'] = tr.criticidad_min.map(_nom), te.criticidad_min.map(_nom)
    print('  Reparto antes de fusionar (val / test):')
    print(pd.DataFrame({'val': tr.c.value_counts(), 'test': te.c.value_counts()})
          .reindex(['3_leve', '2_media', '1_grave']).fillna(0).astype(int)
          .to_string())
    _mapa, orden_c = fusionar_escasas(tr.c.to_numpy(),
                                      ['3_leve', '2_media', '1_grave'])
    if len(orden_c) < 3:
        print(f'  Clases con < {MIN_CLASE} ventanas en entrenamiento, fusionadas '
              f'con su vecina: {orden_c}'
              + ('   -> B1b queda BINARIO.' if len(orden_c) == 2 else ''))
    tr['y'], te['y'] = tr.c.map(_mapa), te.c.map(_mapa)
    X = {k: (tr[v], te[v]) for k, v in CONJUNTOS_B1.items()}
    t, p = correr_tarea('B1_criticidad', crit, X, tr.y.to_numpy(),
                        te.y.to_numpy(), tr.error_ae, te.error_ae, orden_c,
                        orden=orden_c, imprimir_cm=(crit == CRITERIO_PRINCIPAL))
    res_b1.append(t)
    preds_b1[(crit, 'B1_criticidad')] = (p, te, orden_c)

res_b1 = pd.concat(res_b1, ignore_index=True)


# =============================================================================
# 7. B2 — TIPO DE FALLA (componente de la tabla del profesor)
# =============================================================================
# La etiqueta es componente_b (bloque 2): criticidad.COMPONENTE de la regla
# dominante. No es un proxy: es la taxonomía que entregó el profesor. Pero
# como sale de reglas que son umbrales, B2 se entrena con SIN_EXTREMOS
# (restricción 2) y SOLO_SENSORES va al lado sólo para mostrar cuánto se
# apoyaría el bosque en los cruces de umbral si se lo dejara.
#
# Las clases con menos de MIN_CLASE ventanas en entrenamiento se juntan en
# 'otros' ANTES de medir nada. socket_liner (I, A11) no se activa por física
# -- T8 - T1 llega a 4,3 contra el umbral 32 -- y setting (A23) no es
# evaluable sin ZI266, así que esas dos no van a aparecer.
#
# Y la misma regla vale para el grupo que resulta. En la primera corrida real
# 'otros' juntó 11 de alimentación + 3 de filtro + 3 de estructura: 17
# ventanas de entrenamiento, bajo el mínimo que el propio script declara, y
# aun así entraba al F1 macro con el mismo peso que lubricación (1.578). Si
# 'otros' -- o cualquier clase -- queda bajo MIN_CLASE después de agrupar, sus
# ventanas salen de B2 en entrenamiento Y en prueba, y se informa cuántas.
# Lo que no se aprende no se mide.
# =============================================================================
print('\n' + '=' * 72)
print('7. B2 — TIPO DE FALLA')
print('=' * 72)

res_b2, preds_b2 = [], {}
for crit, u in UMBRALES.items():
    tr = d_va[(d_va.error_ae > u) & (d_va.componente_b != 'normal')].copy()
    te = d_te[(d_te.error_ae > u) & (d_te.componente_b != 'normal')].copy()
    print(f'\n######## criterio del filtro: {crit} ########')
    _cnt = tr.componente_b.value_counts()
    chicas = sorted(_cnt[_cnt < MIN_CLASE].index)
    grandes = set(_cnt[_cnt >= MIN_CLASE].index)
    print('Ventanas por componente, antes de agrupar (val / test):')
    print(pd.DataFrame({'val': _cnt, 'test': te.componente_b.value_counts()})
          .fillna(0).astype(int).sort_values('val', ascending=False).to_string())
    for _d in (tr, te):
        _d['y'] = np.where(_d.componente_b.isin(grandes), _d.componente_b,
                           'otros')
    print(f'  Agrupadas en "otros" (< {MIN_CLASE} en entrenamiento, o ausentes '
          f'de él): {chicas or "ninguna"}')
    _n_y = tr.y.value_counts()
    fuera = sorted(_n_y[_n_y < MIN_CLASE].index)
    if fuera:
        _sale_tr = tr.y.isin(fuera) | ~tr.y.isin(_n_y.index)
        _sale_te = te.y.isin(fuera) | ~te.y.isin(_n_y.index)
        print(f'  Después de agrupar quedan bajo {MIN_CLASE} en entrenamiento: '
              f'{ {c: int(_n_y[c]) for c in fuera} }')
        print(f'  -> salen de B2: val {int(_sale_tr.sum()):,} | test '
              f'{int(_sale_te.sum()):,} ventanas. No se aprenden con tan pocas,')
        print('     y no se miden. Se declaran como cobertura que B2 no tiene.')
        tr, te = tr[~_sale_tr].copy(), te[~_sale_te].copy()
    clases = sorted(set(tr.y) | set(te.y))
    print(f'  Clases de B2: {clases}')
    if len(set(tr.y)) < 2:
        print('  Queda una sola clase en entrenamiento: B2 no tiene nada que '
              'clasificar con este filtro. Se salta.')
        continue
    X = {k: (tr[v], te[v]) for k, v in CONJUNTOS_B2.items()}
    t, p = correr_tarea('B2_componente', crit, X, tr.y.to_numpy(),
                        te.y.to_numpy(), tr.error_ae, te.error_ae, clases,
                        imprimir_cm=(crit == CRITERIO_PRINCIPAL))
    res_b2.append(t)
    preds_b2[crit] = (p, tr, te, clases)

res_b2 = pd.concat(res_b2, ignore_index=True) if res_b2 else pd.DataFrame()


# =============================================================================
# 8. VARIABLE CAUSANTE — DOS RESPUESTAS INDEPENDIENTES
# =============================================================================
# (a) La del índice W: sensor_origen, el sensor que alcanza w_max en la
#     ventana. El bloque 5e del paso 02 dice que esto "contesta directamente
#     lo que pidió el profesor".
# (b) La del modelo: importancia por PERMUTACIÓN sobre prueba, agregada por
#     sensor sumando sus estadísticos. No feature_importances_: esa mide
#     cuántas veces un árbol corta en la variable, y está sesgada hacia las de
#     muchos valores distintos.
#
# Para que las dos sean comparables, la importancia se calcula POR CLASE:
# cuánto cae el recall de esa clase al desordenar los estadísticos de cada
# sensor. Así se puede preguntar, para cada tipo de falla, qué sensor
# necesita el modelo para reconocerla, y contrastarlo con el sensor que el
# índice W marca como origen en esas mismas ventanas.
#
# El modelo es el XGB sobre SIN_EXTREMOS, con el criterio principal del
# filtro. DECISIÓN TOMADA DESPUÉS DE VER PRUEBA, y así se declara: la versión
# anterior fijaba de antemano el RF, y en la primera corrida real el RF dio
# F1 0,49 en B2 contra 0,74 del XGB. La importancia por permutación de un
# modelo que acierta la mitad de las veces describe sus errores, no los
# tipos de falla: la atribución tiene que salir del modelo que efectivamente
# clasifica. El cambio no toca ninguna métrica de desempeño de B1 ni de B2 --
# sólo qué modelo se interroga para atribuir. Sin xgboost se cae al RF.
# =============================================================================
print('\n' + '=' * 72)
print('8. VARIABLE CAUSANTE')
print('=' * 72)


def sensor_de(feature):
    return feature.rsplit('_', 1)[0]


causante = pd.DataFrame()
if CRITERIO_PRINCIPAL in preds_b2:
    _, tr, te, clases = preds_b2[CRITERIO_PRINCIPAL]
    Xtr, Xte = tr[FEATURES_SIN_EXTREMOS], te[FEATURES_SIN_EXTREMOS]
    verificar_matriz(Xtr.columns, 'B2/causante')
    NOMBRE_ATRIB = 'XGB' if HAY_XGB else 'RF'
    m_atrib = modelos()[NOMBRE_ATRIB].fit(Xtr, tr.y)
    print(f'Modelo interrogado: {NOMBRE_ATRIB} / SIN_EXTREMOS')

    # Scorers escritos a mano y no con make_scorer ni con el nombre
    # 'f1_macro': make_scorer(recall_score) hereda pos_label=1 de
    # recall_score, y las versiones nuevas de scikit-learn lo validan contra
    # las clases aunque average='macro' lo ignore -- con etiquetas de texto
    # revienta. Además así sirven igual para el RF y para el XGB envuelto.
    def sc_f1(est, X, y):
        return f1_score(y, est.predict(X), labels=clases, average='macro',
                        zero_division=0)

    def por_sensor(imp):
        return (pd.Series(imp, index=FEATURES_SIN_EXTREMOS)
                .groupby(sensor_de).sum().reindex(VARIABLES))

    # --- global -----------------------------------------------------------------
    pi = permutation_importance(m_atrib, Xte, te.y, scoring=sc_f1,
                                n_repeats=10, random_state=SEED, n_jobs=-1)
    imp_global = por_sensor(pi.importances_mean)
    frec_origen = te.sensor_origen.value_counts().reindex(VARIABLES).fillna(0)
    rho, p_rho = spearmanr(imp_global, frec_origen)
    print('Global, sobre las ventanas de prueba que pasaron el filtro:')
    print(pd.DataFrame({'importancia_permutacion': imp_global.round(4),
                        'veces_sensor_origen': frec_origen.astype(int)})
          .sort_values('importancia_permutacion', ascending=False).to_string())
    print(f'\n  Spearman entre las dos columnas: rho = {rho:+.3f} (p = {p_rho:.3f})')

    # --- por clase ----------------------------------------------------------------
    filas = []
    print('\nPor tipo de falla:')
    print(pd.crosstab(te.y, te.sensor_origen).to_string())
    for c in clases:
        sel = te.y == c
        if sel.sum() == 0:
            continue
        def sc(est, X, y, c=c):          # recall de la clase c
            return recall_score(y, est.predict(X), labels=[c],
                                average='macro', zero_division=0)
        pic = permutation_importance(m_atrib, Xte, te.y, scoring=sc,
                                     n_repeats=10,
                                     random_state=SEED, n_jobs=-1)
        imp_c = por_sensor(pic.importances_mean).sort_values(ascending=False)
        top3 = list(imp_c.index[:3])
        orig = te.loc[sel, 'sensor_origen'].value_counts()
        filas.append({
            'clase': c, 'n_test': int(sel.sum()),
            'sensor_modelo_top1': imp_c.index[0],
            'importancia_top1': round(float(imp_c.iloc[0]), 4),
            'sensores_modelo_top3': ', '.join(top3),
            'sensor_origen_moda': orig.index[0],
            'pct_moda_sensor_origen': round(100 * orig.iloc[0] / sel.sum(), 1),
            'coincide_top1': imp_c.index[0] == orig.index[0],
            'sensor_origen_en_top3': orig.index[0] in top3,
        })
    causante = pd.DataFrame(filas)
    print()
    print(causante.to_string(index=False))
    if len(causante):
        print(f'\n  Coincide el top-1: {int(causante.coincide_top1.sum())} de '
              f'{len(causante)} clases | sensor_origen dentro del top-3 del '
              f'modelo: {int(causante.sensor_origen_en_top3.sum())} de '
              f'{len(causante)}')
    causante = pd.concat([causante, pd.DataFrame([{
        'clase': 'GLOBAL', 'n_test': len(te),
        'sensor_modelo_top1': imp_global.idxmax(),
        'importancia_top1': round(float(imp_global.max()), 4),
        'sensores_modelo_top3': ', '.join(imp_global.sort_values(
            ascending=False).index[:3]),
        'sensor_origen_moda': frec_origen.idxmax(),
        'pct_moda_sensor_origen': round(100 * frec_origen.max() / len(te), 1),
        'coincide_top1': imp_global.idxmax() == frec_origen.idxmax(),
        'sensor_origen_en_top3': frec_origen.idxmax() in
        imp_global.sort_values(ascending=False).index[:3],
        'spearman_global': round(float(rho), 3)}])], ignore_index=True)
    print('\nCÓMO LEERLO. Son dos caminos independientes: uno mira cuál sensor')
    print('se aleja más de su distribución normal (W), el otro cuál necesita')
    print('el bosque para reconocer el tipo de falla. Si coinciden, la')
    print('atribución es robusta. Si no, hay que explicarlo: W responde al')
    print('NIVEL del sensor más corrido, y en las variables lentas eso no es lo')
    print('mismo que el sensor que distingue un componente de otro.')
else:
    print('B2 no corrió con el criterio principal: no hay variable causante.')


# =============================================================================
# 9. GUARDAR, Y EL RESUMEN QUE VA A LA MEMORIA
# =============================================================================
print('\n' + '=' * 72)
print('9. GUARDANDO')
print('=' * 72)

_filtro_cols = filtro.set_index('criterio')
for _t in (res_b1, res_b2):
    if len(_t):
        for _c in [c for c in filtro.columns if c.startswith('pct_perdidas_')]:
            _t[_c] = _t.criterio_umbral.map(_filtro_cols[_c])
res_b1.round(4).to_csv(SALIDA_B1, sep=';', decimal=',', index=False,
                       encoding='utf-8-sig')
res_b2.round(4).to_csv(SALIDA_B2, sep=';', decimal=',', index=False,
                       encoding='utf-8-sig')
causante.to_csv(SALIDA_CAUSANTE, sep=';', decimal=',', index=False,
                encoding='utf-8-sig')
for _f in (SALIDA_B1, SALIDA_B2, SALIDA_CAUSANTE):
    print(f'Guardado: {_f}')

# El conjunto que va al cuerpo de la memoria. B1: sólo sensores, sin el
# error del AE ni el índice W. B2: sin extremos, por la restricción 2. El que
# gane por F1 se informa igual, pero el intervalo se calcula sobre éste.
CONJUNTO_DEFENDIDO = {'B1_banda_w': 'SOLO_SENSORES',
                      'B1_criticidad': 'SOLO_SENSORES',
                      'B2_componente': 'SIN_EXTREMOS'}

print('\n' + '=' * 72)
print(f'RESUMEN — criterio del filtro: {CRITERIO_PRINCIPAL}')
print('=' * 72)
_fp = filtro.set_index('criterio').loc[CRITERIO_PRINCIPAL]
print(f'El filtro deja afuera el {_fp.pct_perdidas_banda_w_3_grave} % de las '
      f'3_grave por banda_w y el {_fp.pct_perdidas_criticidad_1} % de las de '
      f'criticidad 1. Todo lo que sigue es sobre lo que sí pasó.')

for tarea, tabla in [('B1_banda_w', res_b1), ('B1_criticidad', res_b1),
                     ('B2_componente', res_b2)]:
    if not len(tabla):
        continue
    t = tabla[(tabla.tarea == tarea) &
              (tabla.criterio_umbral == CRITERIO_PRINCIPAL)]
    if not len(t):
        continue
    bases = t[t.modelo.str.startswith('base_')]
    mods = t[~t.modelo.str.startswith('base_')]
    mb = bases.loc[bases.f1_macro.idxmax()]
    mg = mods.loc[mods.f1_macro.idxmax()]
    _def = mods[mods.features == CONJUNTO_DEFENDIDO[tarea]]
    mm = _def.loc[_def.f1_macro.idxmax()]
    print(f'\n{tarea}')
    print('  F1 macro del mejor modelo por conjunto de features:')
    for conj, g in mods.groupby('features'):
        r = g.loc[g.f1_macro.idxmax()]
        print(f'    {conj:<14} {r.modelo:<4} {r.f1_macro:.4f}')
    print(f'  Ganó por F1: {mg.features} / {mg.modelo} con {mg.f1_macro:.4f}')
    print(f'  El que se defiende: {mm.features} / {mm.modelo} con '
          f'{mm.f1_macro:.4f}')
    print(f'  Mejor baseline: {mb.modelo} con F1 {mb.f1_macro:.4f}  ->  '
          f'diferencia {mm.f1_macro - mb.f1_macro:+.4f}')

    if tarea == 'B2_componente':
        p, _, te, clases = preds_b2[CRITERIO_PRINCIPAL]
    else:
        p, te, clases = preds_b1[(CRITERIO_PRINCIPAL, tarea)]
    y_te = te.y.to_numpy()
    lo, hi = ic_diferencia(y_te, p[(mm.features, mm.modelo)],
                           p[('-', mb.modelo)], te.tramo_id, clases)
    print(f'  IC 95 % de esa diferencia (bootstrap por tramo): '
          f'[{lo:+.4f}, {hi:+.4f}]')
    print('  -> ' + ('le gana CLARAMENTE: el intervalo no toca el cero.'
                     if lo > 0 else
                     'NO le gana claramente: el intervalo incluye el cero. '
                     'Ése es el resultado y se reporta así.'))

    if tarea == 'B2_componente':
        f_se = mods[mods.features == 'SIN_EXTREMOS'].f1_macro.max()
        f_ss = mods[mods.features == 'SOLO_SENSORES'].f1_macro.max()
        print(f'  SIN_EXTREMOS {f_se:.4f} vs SOLO_SENSORES {f_ss:.4f} '
              f'({f_se - f_ss:+.4f}).')
        if f_ss - f_se > 0.10:
            print('  Se desploma al quitar los extremos: el bosque estaba')
            print('  reaprendiendo los umbrales de las reglas. La conclusión que')
            print('  se defiende es la de SIN_EXTREMOS, y acota lo que se afirma.')
        else:
            print('  Se sostiene sin los extremos: el tipo de falla tiene firma')
            print('  en la dinámica (nivel, dispersión y tendencia), no sólo en')
            print('  los cruces de umbral.')
    if tarea.startswith('B1'):
        f_ss = mods[mods.features == 'SOLO_SENSORES'].f1_macro.max()
        f_me = mods[mods.features == 'MAS_ERROR'].f1_macro.max()
        print(f'  Agregar el error del autoencoder a los sensores: '
              f'{f_me - f_ss:+.4f} de F1.')

if len(causante):
    g = causante[causante.clase == 'GLOBAL'].iloc[0]
    por_clase = causante[causante.clase != 'GLOBAL']
    print(f'\nVariable causante ({NOMBRE_ATRIB} / SIN_EXTREMOS):')
    print(f'  global: modelo -> {g.sensor_modelo_top1} | sensor_origen más '
          f'frecuente -> {g.sensor_origen_moda} | '
          f'{"COINCIDEN" if g.coincide_top1 else "NO coinciden"} '
          f'(Spearman {g.spearman_global:+.3f})')
    print(f'  por tipo de falla: top-1 coincide en '
          f'{int(por_clase.coincide_top1.sum())} de {len(por_clase)}; '
          f'sensor_origen en el top-3 del modelo en '
          f'{int(por_clase.sensor_origen_en_top3.sum())} de {len(por_clase)}.')

print('\n' + '=' * 72)
print('LISTO. Los tres CSV tienen todas las combinaciones; este resumen es')
print('sólo el criterio principal. Para la memoria: la tabla entera, no un')
print('número suelto.')
print('=' * 72)
