# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04i — DEL AUTOENCODER AL TIPO DE FALLA: FIRMA DE PARETO Y TABLA DEL EXPERTO
=============================================================================
Lo pedido en la reunión del 20 de noviembre: pasar, ventana por ventana y sin hacerlo a mano,
de lo que dice el autoencoder al tipo de falla de la tabla del experto (criticidadCH8.xlsx).
Para cada ventana que el autoencoder marca como anómala (error mayor que el umbral P95):
  1. Dos distribuciones sobre las 14 variables: qué parte del MSE de la ventana pone cada una
     (%MSE) y cuánto subió su error respecto de lo normal (ΔMSE: error en la ventana menos el
     error medio de las ventanas sin alarma de validación; solo cuentan las subidas).
  2. Pareto: en cada distribución, las variables que juntas explican el 80 %.
  3. Firma de la ventana: las variables que están en los dos conjuntos, con su sentido (si su
     media subió o bajó respecto de lo normal).
  4. Cruce con la tabla del experto. Cada síntoma de la tabla es un conjunto de variables con el
     sentido que vigilan sus reglas. Se busca el síntoma que contiene la firma completa (todas
     sus variables a la vez); si hay varios, el más específico. Si ninguno la contiene, el que
     explica al menos el 80 % del ΔMSE de la firma; si tampoco, la menor combinación de síntomas
     que llega al 80 %. Lo que no se puede explicar queda como combinación no descrita en la
     tabla: un comportamiento nuevo, para que lo revise el experto.
  5. Contraste con las alarmas que sonaron en la misma ventana (acuerdo y azar).
Lo pedido en la reunión del 23 de noviembre: (a) el paso de los dos histogramas al síntoma como un
procedimiento que se pueda seguir a mano, el filtrado sucesivo que propuso el profesor (síntomas que
tienen la 1.ª variable de la firma, de esos los que tienen la 2.ª, y así; gana la máxima coincidencia),
comparado con el método del punto 4; (b) cuántas veces los dos histogramas marcan las mismas variables,
y cuántas veces la firma cabe en varios síntomas a la vez o un síntoma aparece en firmas distintas
(síntomas que se pisan en la tabla del experto, o síntomas nuevos).
Entrada:  {EQUIPO}_error_reconstruccion_W60.csv (paso 04), {EQUIPO}_ventanas_W60.csv y
          {EQUIPO}_features_W60.csv (paso 02); criticidad.py y reglas_engine.py
Salida:   04i_tabla_sintomas.csv, 04i_ventanas.csv, 04i_ventanas_variables.csv, 04i_resumen.csv,
          04i_sintomas.csv, 04i_combinaciones.csv, 04i_ejemplos.png, 04i_diagnostico.png
Lo normal y los umbrales se fijan en validación; prueba solo se informa.
"""
import os
import re
import sys

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
if RUTA_BASE not in sys.path:
    sys.path.append(RUTA_BASE)

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import criticidad
from reglas_engine import POR_ID, IDS as IDS_REGLAS

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# 0. CONSTANTES
# =============================================================================
W = 60
PERCENTIL = 95
LLAVES = ['tramo_id', 'ventana_id']
PARETO = float(os.environ.get('PARETO', 0.80))        # regla 80-20 del profesor
COBERTURA = float(os.environ.get('COBERTURA', 0.80))  # un síntoma "explica" la firma si cubre el 80 % de su ΔMSE
MAX_FIRMA = int(os.environ.get('MAX_FIRMA', 6))       # con más variables, la anomalía está repartida: difusa
Z_SENTIDO = 1.0              # |z| de la media desde el cual se dice que la variable subió o bajó
TEXTO_COMP = {'lubricacion': 'lubricación', 'motor': 'motor', 'alimentacion': 'alimentación', 'filtro': 'filtro',
              'eje_principal': 'eje principal', 'socket_liner': 'socket liner', 'contraeje': 'contraeje',
              'tanque_aceite': 'tanque de aceite', 'estructura': 'estructura', 'setting': 'setting'}
CATEGORIAS = ['contenida en un síntoma', 'explicada por un síntoma', 'combinación de síntomas',
              'no descrita en la tabla', 'difusa']
FLECHA = {'sube': '↑', 'baja': '↓', '': '~'}
NARANJO, GRIS, AZUL = '#E97132', '#BFBFBF', '#0F9ED5'

PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
if PUNTO_PARADA not in ('perdida', 'separacion'):
    raise SystemExit(f"PUNTO_PARADA tiene que ser 'perdida' o 'separacion', no {PUNTO_PARADA!r}.")
AE = 'err_AE_normales_sep' if PUNTO_PARADA == 'separacion' else 'err_AE_normales'
ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
for _f in [ARCHIVO_ERROR, ARCHIVO_VENTANAS]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste los pasos 02 y 04?')


def guardar(df, nombre):
    df.to_csv(f'{RUTA_EQ}/{nombre}', sep=';', decimal=',', index=False, encoding='utf-8-sig')


def texto_firma(pares):
    """[('V1','sube'), ('PEL','baja')] -> 'V1↑ · PEL↓'; las cuatro vibraciones juntas, 'V1-V4↑'."""
    pares = list(pares)
    vib = [p for p in pares if p[0] in ('V1', 'V2', 'V3', 'V4')]
    if len(vib) == 4 and len({s for _, s in vib}) == 1:
        primero = pares.index(vib[0])
        pares = [p for p in pares if p not in vib]
        pares.insert(primero, ('V1-V4', vib[0][1]))
    return ' · '.join(f'{v}{FLECHA.get(s, "~")}' for v, s in pares)


# =============================================================================
# 1. DATOS
# =============================================================================
print('=' * 72)
print(f'PASO 04i — DEL AUTOENCODER AL TIPO DE FALLA ({EQUIPO})')
print('=' * 72)
m = pd.read_csv(ARCHIVO_ERROR)
if AE not in m.columns or not np.allclose(m[AE], m.error_reconstruccion):
    raise SystemExit(f'El paso 04 se corrió con otro punto de parada: córrelo con PUNTO_PARADA={PUNTO_PARADA!r}.')
VARIABLES = [c[5:] for c in m.columns if c.startswith('evar_')]
EVAR = [f'evar_{v}' for v in VARIABLES]
RC = [r for r in IDS_REGLAS if r in pd.read_csv(ARCHIVO_VENTANAS, nrows=0).columns]
_ven = pd.read_csv(ARCHIVO_VENTANAS, usecols=LLAVES + ['timestamp_inicio'] + RC).drop_duplicates(LLAVES)
m = m.drop(columns=[c for c in RC + ['timestamp_inicio'] if c in m.columns]).merge(
    _ven, on=LLAVES, how='left', validate='one_to_one')
m = m[m.conjunto.isin(['val', 'test'])].reset_index(drop=True)
MEDIAS = [f'{v}_media' for v in VARIABLES]
HAY_Z = os.path.exists(ARCHIVO_FEATURES)
if HAY_Z:
    _f = pd.read_csv(ARCHIVO_FEATURES, usecols=lambda c: c in LLAVES + MEDIAS).drop_duplicates(LLAVES)
    HAY_Z = all(c in _f.columns for c in MEDIAS)
    if HAY_Z:
        m = m.drop(columns=[c for c in MEDIAS if c in m.columns]).merge(_f, on=LLAVES, how='left',
                                                                         validate='one_to_one')
if not HAY_Z:
    print('Sin las medias del paso 02: la firma queda sin sentido y el cruce se hace solo por variable.')

n = len(m)
EV = m[EVAR].to_numpy(dtype=float)                  # error de cada variable en cada ventana
ERR = m.error_reconstruccion.to_numpy(dtype=float)   # error de la ventana
CU = m[RC].fillna(0).to_numpy()
ACT = CU > 0
anomala = (m.n_reglas_activas > 0).to_numpy() if 'n_reglas_activas' in m.columns else ACT.any(axis=1)
normal = ~anomala
conj = m.conjunto.to_numpy()
va, te = conj == 'val', conj == 'test'
if not (va & normal).any():
    raise SystemExit('No hay ventanas sin alarma en validación: no se puede fijar el nivel normal.')
umbral = float(np.percentile(ERR[va & normal], PERCENTIL))        # umbral P95 del error total (el del 04)
marcada = ERR > umbral
EN = EV[va & normal].mean(axis=0)                                 # error normal de cada variable
mes = pd.to_datetime(m.timestamp_inicio).dt.strftime('%Y-%m').to_numpy()
if HAY_Z:
    _med = m[MEDIAS].to_numpy(dtype=float)
    _mu, _sd = np.nanmean(_med[va & normal], axis=0), np.nanstd(_med[va & normal], axis=0, ddof=1)
    ZS = (_med - _mu) / np.where(_sd > 0, _sd, 1.0)
else:
    ZS = np.full(EV.shape, np.nan)
SENTIDO = np.where(np.abs(ZS) >= Z_SENTIDO, np.where(ZS > 0, 'sube', 'baja'), '')
print(f'Ventanas: validación {int(va.sum()):,} | prueba {int(te.sum()):,} | umbral P95 = {umbral:.5f}')
print(f'Marcadas por el autoencoder: validación {int((va & marcada).sum()):,} | prueba {int((te & marcada).sum()):,}')


# =============================================================================
# 2. LA TABLA DEL EXPERTO COMO SÍNTOMAS: VARIABLES Y SENTIDO DE CADA FALLA
# =============================================================================
print('\n' + '=' * 72)
print('2. LA TABLA DEL EXPERTO (criticidadCH8.xlsx) COMO SÍNTOMAS')
print('=' * 72)


def sentido_op(op):
    return 'sube' if str(op).startswith('>') else 'baja'


def pares_regla(rid):
    """Variables que mira la regla y el sentido en que la dispara (a - b > x: a sube o b baja)."""
    d = POR_ID[rid]
    s = sentido_op(d['op'])
    if d['tipo'] == 'diferencia':
        return [(d['a'], s), (d['b'], 'baja' if s == 'sube' else 'sube')]
    pares = [(x, s) for x in ([d['var']] if 'var' in d else list(d.get('vars', [])))]
    if 'var2' in d:
        pares.append((d['var2'], sentido_op(d.get('op2', '>'))))
    return pares


def texto_base(falla):
    """La falla sin el grado entre paréntesis: 'Calentamiento ... (alta)' -> 'Calentamiento ...'."""
    return re.sub(r'\s*\((alarma|alta|critica|crítica|baja|trip)\)\s*$', '', falla).strip()


REGLAS_OK = [r for r in RC if r in criticidad.FALLA and all(v in VARIABLES for v, _ in pares_regla(r))]
_grupos = {}
for r in REGLAS_OK:
    _grupos.setdefault(frozenset(pares_regla(r)), []).append(r)
SINTOMAS = []                                    # una fila por síntoma
for pares, reglas in _grupos.items():
    general = max(reglas, key=lambda r: (criticidad.CRITICIDAD.get(r, 0), -REGLAS_OK.index(r)))
    orden = [(v, s) for v in VARIABLES for s in ('sube', 'baja') if (v, s) in pares]
    SINTOMAS.append({'pares': set(pares), 'vars': {v for v, _ in pares}, 'reglas': reglas, 'miembros': None,
                     'falla': texto_base(criticidad.FALLA[general]), 'variables': texto_firma(orden),
                     'modos': ' | '.join(dict.fromkeys(texto_base(criticidad.FALLA[r]) for r in reglas)),
                     'componente': TEXTO_COMP.get(pd.Series([criticidad.COMPONENTE.get(r, '') for r in reglas])
                                                  .value_counts().index[0], 'sin componente'),
                     'criticidad': min(criticidad.CRITICIDAD.get(r, 9) for r in reglas)})
for nombre, d in criticidad.REGLAS_COMPUESTAS.items():          # las filas combinadas de la tabla
    if all(r in REGLAS_OK for r in d['miembros']):
        pares = set().union(*(pares_regla(r) for r in d['miembros']))
        orden = [(v, s) for v in VARIABLES for s in ('sube', 'baja') if (v, s) in pares]
        SINTOMAS.append({'pares': pares, 'vars': {v for v, _ in pares}, 'reglas': [nombre],
                         'miembros': d['miembros'], 'falla': d['falla'], 'variables': texto_firma(orden),
                         'modos': d['falla'], 'componente': TEXTO_COMP.get(d['componente'], d['componente']),
                         'criticidad': d['criticidad']})
SINTOMAS.sort(key=lambda s: (VARIABLES.index(min(s['vars'], key=VARIABLES.index)), len(s['vars'])))
for i, s in enumerate(SINTOMAS, start=1):
    s['id'] = f'S{i:02d}'
    s['etiqueta'] = f"{s['falla']} ({s['variables']})"
tabla = pd.DataFrame([{'sintoma': s['id'], 'etiqueta': s['etiqueta'], 'variables': s['variables'],
                       'reglas': ', '.join(s['reglas']), 'modos_de_falla': s['modos'], 'componente': s['componente'],
                       'criticidad_mas_alta': s['criticidad'], 'n_variables': len(s['vars'])} for s in SINTOMAS])
guardar(tabla, '04i_tabla_sintomas.csv')
print(tabla[['sintoma', 'variables', 'reglas', 'componente']].to_string(index=False))
ETIQ = {s['id']: s['etiqueta'] for s in SINTOMAS}
COMP_S = {s['id']: s['componente'] for s in SINTOMAS}
POR_ID_S = {s['id']: s for s in SINTOMAS}
SINTOMA_DE_REGLA = {r: s['id'] for s in SINTOMAS if s['miembros'] is None for r in s['reglas']}


def sintomas_de_alarmas(i):
    """Los síntomas de las reglas que sonaron en la ventana (las combinadas, si sonaron todas)."""
    act = {RC[j] for j in np.flatnonzero(ACT[i])}
    out = {SINTOMA_DE_REGLA[r] for r in act if r in SINTOMA_DE_REGLA}
    out |= {s['id'] for s in SINTOMAS if s['miembros'] and set(s['miembros']) <= act}
    return out


def sintoma_alarma_critica(i):
    """El síntoma de la regla más crítica que sonó (empate: la que sonó más veces), como el 04g."""
    act = [j for j in np.flatnonzero(ACT[i]) if RC[j] in SINTOMA_DE_REGLA]
    if not act:
        return ''
    crit = np.array([criticidad.CRITICIDAD.get(RC[j], 9) for j in act])
    top = [act[k] for k in np.flatnonzero(crit == crit.min())]
    return SINTOMA_DE_REGLA[RC[max(top, key=lambda j: CU[i, j])]]


# =============================================================================
# 3. LAS DOS DISTRIBUCIONES Y EL PARETO DE CADA VENTANA
# =============================================================================
PMSE = 100 * EV / np.maximum(EV.sum(axis=1, keepdims=True), 1e-12)          # parte del MSE de la ventana
DPOS = np.clip(EV - EN, 0, None)                                             # cuánto subió el error
PDEL = 100 * DPOS / np.maximum(DPOS.sum(axis=1, keepdims=True), 1e-12)       # parte de esa subida


def pareto(P):
    """Máscara de las variables que, ordenadas de mayor a menor, juntas llegan al 80 %; y su orden."""
    orden = np.argsort(-P, axis=1, kind='stable')
    acum = np.cumsum(np.take_along_axis(P, orden, axis=1), axis=1)
    k = np.minimum((acum < 100 * PARETO - 1e-9).sum(axis=1) + 1, P.shape[1])
    rango = np.argsort(orden, axis=1)                                        # puesto de cada variable
    mascara = rango < k[:, None]
    mascara[P.sum(axis=1) <= 0] = False
    return mascara, rango + 1


EN_MSE, ORDEN_MSE = pareto(PMSE)
EN_DEL, ORDEN_DEL = pareto(PDEL)
FIRMA = EN_MSE & EN_DEL
N_FIRMA = FIRMA.sum(axis=1)


# =============================================================================
# 4. CRUCE DE CADA FIRMA CON LA TABLA
# =============================================================================
def diagnosticar(i):
    """Categoría, síntomas elegidos, cobertura, síntomas que la contienen, variables que quedan sin
    cubrir por los síntomas elegidos y variables que ningún síntoma de la tabla vigila en ese sentido."""
    idx = [j for j in np.argsort(-PDEL[i]) if FIRMA[i, j]]
    if not idx or len(idx) > MAX_FIRMA:
        return 'difusa', [], 0.0, 0, '', ''
    vs = [VARIABLES[j] for j in idx]
    ss = [SENTIDO[i, j] for j in idx]
    w = np.array([PDEL[i, j] for j in idx])
    w = w / w.sum() if w.sum() > 0 else np.full(len(idx), 1 / len(idx))

    def coincide(s):
        return np.array([v in s['vars'] and (not se or (v, se) in s['pares']) for v, se in zip(vs, ss)])

    co = {s['id']: coincide(s) for s in SINTOMAS}
    # choque: la variable está en el síntoma, pero se movió en el sentido contrario al que vigila
    choque = {s['id']: sum(v in s['vars'] and bool(se) and (v, se) not in s['pares'] for v, se in zip(vs, ss))
              for s in SINTOMAS}
    en_tabla = np.any(np.array(list(co.values())), axis=0)
    fuera = texto_firma([(v, se) for v, se, c in zip(vs, ss, en_tabla) if not c])
    jac = {s['id']: co[s['id']].sum() / len(set(vs) | s['vars']) for s in SINTOMAS}   # especificidad
    contienen = [t for t, c in co.items() if c.all()]
    if contienen:
        t = max(contienen, key=lambda t: (jac[t], -POR_ID_S[t]['criticidad']))
        return 'contenida en un síntoma', [t], 100.0, len(contienen), '', fuera
    cob = {t: float(w[c].sum()) for t, c in co.items()}
    t = max(cob, key=lambda t: (cob[t], -choque[t], jac[t]))
    if cob[t] >= COBERTURA:
        resto = texto_firma([(v, se) for v, se, c in zip(vs, ss, co[t]) if not c])
        return 'explicada por un síntoma', [t], 100 * cob[t], 0, resto, fuera
    elegidos, cubre = [], np.zeros(len(idx), dtype=bool)
    while w[cubre].sum() < COBERTURA:
        gana = {t: float(w[c & ~cubre].sum()) for t, c in co.items() if t not in elegidos}
        t = max(gana, key=lambda t: (gana[t], -choque[t], jac[t])) if gana else None
        if t is None or gana[t] <= 0:
            break
        elegidos.append(t)
        cubre |= co[t]
    resto = texto_firma([(v, se) for v, se, c in zip(vs, ss, cubre) if not c])
    if w[cubre].sum() >= COBERTURA and len(elegidos) >= 2:
        return 'combinación de síntomas', elegidos, 100 * float(w[cubre].sum()), 0, resto, fuera
    return 'no descrita en la tabla', elegidos, 100 * float(w[cubre].sum()), 0, resto, fuera


def filtrado_sucesivo(i):
    """El procedimiento que propuso el profesor el 23 de noviembre, para que se pueda seguir a mano.
    Las variables de la firma van de mayor a menor ΔMSE. Se toman los síntomas que tienen la primera
    (con su sentido); de esos, los que además tienen la segunda; luego la tercera, y así. Si ninguno de
    los que quedan tiene una variable, esa variable no se usa en esta vuelta y se sigue con la siguiente.
    Al final quedan los síntomas de máxima coincidencia: todos tienen las mismas variables de la firma.
    Gana el más específico (menos variables propias) y, si empatan, el más crítico. Las variables que ese
    síntoma no explica pasan a una segunda vuelta, y así hasta que no queda nada que un síntoma tenga.
    Devuelve los síntomas en orden, el paso a paso en texto y cuántos síntomas empataron en la 1.ª vuelta."""
    idx = [j for j in np.argsort(-PDEL[i]) if FIRMA[i, j]]
    if not idx or len(idx) > MAX_FIRMA:
        return [], '', 0
    def tiene(s, v, se):
        return v in s['vars'] and (not se or (v, se) in s['pares'])

    firma = [(VARIABLES[j], SENTIDO[i, j]) for j in idx]
    pendientes = [p for p in firma if any(tiene(s, *p) for s in SINTOMAS)]
    fuera = [p for p in firma if p not in pendientes]        # ningún síntoma de la tabla la tiene

    elegidos, vueltas, empatados = [], [], 0
    while pendientes:
        cand, usadas, pasos = None, [], []
        for v, se in pendientes:
            base = SINTOMAS if cand is None else cand
            nuevos = [s for s in base if tiene(s, v, se)]
            if nuevos:
                cand, usadas = nuevos, usadas + [(v, se)]
                pasos.append(f'{texto_firma([(v, se)])}: {len(nuevos)}')
            elif cand is not None:
                pasos.append(f'{texto_firma([(v, se)])}: 0')
        if cand is None:                                 # ninguna variable que quede está en la tabla
            break
        gana = min(cand, key=lambda s: (len(s['vars']), s['criticidad']))
        if not elegidos:
            empatados = sum(len(s['vars']) == len(gana['vars']) for s in cand)
        elegidos.append(gana['id'])
        vueltas.append(f'{len(vueltas) + 1}) ' + ' → '.join(pasos) + f' ⇒ {gana["id"]}')
        pendientes = [p for p in pendientes if p not in usadas]
    if fuera:
        vueltas.append(f'fuera de la tabla: {texto_firma(fuera)}')
    return elegidos, ' | '.join(vueltas), empatados


print('\n' + '=' * 72)
print(f'3-4. FIRMA DE PARETO ({100 * PARETO:.0f} %) Y SÍNTOMA DE LA TABLA, VENTANA POR VENTANA')
print('=' * 72)
filas, filas_v = [], []
for i in np.flatnonzero(marcada):
    cat, elegidos, cob, n_cont, resto, fuera = diagnosticar(i)
    idx = [j for j in np.argsort(-PDEL[i]) if FIRMA[i, j]]
    s_al = sintomas_de_alarmas(i)
    s_crit = sintoma_alarma_critica(i)
    comps_al = {COMP_S[t] for t in s_al}
    principal = elegidos[0] if elegidos else ''
    con_alarma = bool(anomala[i])
    filtro, pasos_f, empatados = filtrado_sucesivo(i)
    principal_f = filtro[0] if filtro else ''
    n_mayor = max(int(EN_MSE[i].sum()), int(EN_DEL[i].sum()))
    filas.append({
        'tramo_id': m.tramo_id.iat[i], 'ventana_id': m.ventana_id.iat[i], 'conjunto': conj[i],
        'timestamp_inicio': m.timestamp_inicio.iat[i], 'mes': mes[i], 'error': ERR[i], 'veces_umbral': ERR[i] / umbral,
        'n_pareto_mse': int(EN_MSE[i].sum()), 'n_pareto_delta': int(EN_DEL[i].sum()), 'n_firma': int(N_FIRMA[i]),
        'firma': texto_firma([(VARIABLES[j], SENTIDO[i, j]) for j in idx]),
        'categoria': cat, 'sintoma_principal': ETIQ.get(principal, ''),
        'componente_principal': COMP_S.get(principal, ''),
        'sintomas': ' + '.join(ETIQ[t] for t in elegidos), 'n_sintomas': len(elegidos),
        'cobertura_pct': cob, 'n_sintomas_que_la_contienen': n_cont, 'variables_restantes': resto,
        'variables_fuera_de_la_tabla': fuera,
        'con_alarma': con_alarma, 'reglas_activas': ', '.join(np.array(RC)[ACT[i]]),
        'sintomas_alarmas': ' + '.join(ETIQ[t] for t in sorted(s_al)), 'sintoma_alarma_critica': ETIQ.get(s_crit, ''),
        'componente_alarma_critica': COMP_S.get(s_crit, ''),
        'acuerdo_principal': (principal in s_al) if con_alarma else np.nan,
        'acuerdo_alguno': bool(set(elegidos) & s_al) if con_alarma else np.nan,
        'acuerdo_estricto': (principal == s_crit and principal != '') if con_alarma else np.nan,
        'acuerdo_componente': (COMP_S.get(principal) in comps_al) if con_alarma else np.nan,
        # los dos histogramas: ¿marcan las mismas variables?
        'paretos_identicos': bool(N_FIRMA[i] > 0 and EN_MSE[i].sum() == N_FIRMA[i] == EN_DEL[i].sum()),
        'coincidencia_paretos_pct': 100 * N_FIRMA[i] / n_mayor if n_mayor else 0.0,
        # el filtrado sucesivo del profesor, paso a paso
        'sintoma_principal_filtrado': ETIQ.get(principal_f, ''),
        'sintomas_filtrado': ' + '.join(ETIQ[t] for t in filtro), 'pasos_filtrado': pasos_f,
        'n_sintomas_empatados': empatados,
        'filtrado_igual_principal': bool(principal_f == principal) if principal or principal_f else np.nan,
        'acuerdo_principal_filtrado': (principal_f in s_al) if con_alarma else np.nan,
        '_principal': principal, '_principal_f': principal_f, '_elegidos': elegidos,
        '_s_al': s_al, '_s_crit': s_crit})
    for j, v in enumerate(VARIABLES):
        filas_v.append({'tramo_id': m.tramo_id.iat[i], 'ventana_id': m.ventana_id.iat[i], 'conjunto': conj[i],
                        'variable': v, 'pct_mse': PMSE[i, j], 'pct_delta': PDEL[i, j],
                        'puesto_mse': int(ORDEN_MSE[i, j]), 'puesto_delta': int(ORDEN_DEL[i, j]),
                        'en_pareto_mse': bool(EN_MSE[i, j]), 'en_pareto_delta': bool(EN_DEL[i, j]),
                        'en_firma': bool(FIRMA[i, j]), 'sentido': SENTIDO[i, j] or 'sin cambio de nivel',
                        'z_media': ZS[i, j]})
d = pd.DataFrame(filas)
if d.empty:
    raise SystemExit('El autoencoder no marcó ninguna ventana: no hay nada que diagnosticar.')
guardar(d.drop(columns=[c for c in d.columns if c.startswith('_')]).round(4), '04i_ventanas.csv')
guardar(pd.DataFrame(filas_v).round(4), '04i_ventanas_variables.csv')


# =============================================================================
# 5. RESUMEN, SÍNTOMAS CONTRA ALARMAS Y COMBINACIONES PARA EL EXPERTO
# =============================================================================
def azar_principal(t, col='_principal'):
    """Acuerdo esperado si los síntomas del modelo se repartieran al azar entre las ventanas."""
    q = t[col].value_counts(normalize=True)
    p_en = {s: np.mean([s in a for a in t._s_al]) for s in q.index if s}
    p_crit = t._s_crit.value_counts(normalize=True)
    return (100 * sum(q[s] * p_en.get(s, 0) for s in q.index if s),
            100 * sum(q[s] * p_crit.get(s, 0) for s in q.index if s))


res, sint, comb = [], [], []
for cj in ['val', 'test']:
    t = d[d.conjunto == cj]
    if t.empty:
        continue
    ta = t[t.con_alarma]
    fila = {'conjunto': cj, 'ventanas_marcadas': len(t), 'con_alarma': len(ta),
            **{f'pct_{c.replace(" ", "_")}': 100 * (t.categoria == c).mean() for c in CATEGORIAS},
            'mediana_n_pareto_mse': t.n_pareto_mse.median(), 'mediana_n_pareto_delta': t.n_pareto_delta.median(),
            'mediana_n_firma': t[t.n_firma > 0].n_firma.median(),
            'pct_firma_hasta_3_variables': 100 * t.n_firma.between(1, 3).mean(),
            # simultaneidad de los dos histogramas y de los síntomas
            'pct_paretos_identicos': 100 * t.paretos_identicos.mean(),
            'coincidencia_paretos_media_pct': t.coincidencia_paretos_pct.mean(),
            'pct_firma_en_un_sintoma': 100 * (t.n_sintomas_que_la_contienen == 1).mean(),
            'pct_firma_en_varios_sintomas_a_la_vez': 100 * (t.n_sintomas_que_la_contienen >= 2).mean(),
            'pct_necesita_varios_sintomas': 100 * (t.n_sintomas >= 2).mean(),
            # filtrado sucesivo contra el método del paso 04i
            'pct_filtrado_igual_principal': 100 * t.filtrado_igual_principal.dropna().astype(float).mean(),
            'pct_filtrado_con_empate': 100 * (t.n_sintomas_empatados >= 2).mean()}
    if len(ta):
        az_p, az_e = azar_principal(ta)
        az_f, _ = azar_principal(ta, '_principal_f')
        fila.update({'acuerdo_principal_pct': 100 * ta.acuerdo_principal.astype(float).mean(),
                     'azar_principal_pct': az_p, 'acuerdo_alguno_pct': 100 * ta.acuerdo_alguno.astype(float).mean(),
                     'acuerdo_estricto_pct': 100 * ta.acuerdo_estricto.astype(float).mean(), 'azar_estricto_pct': az_e,
                     'acuerdo_componente_pct': 100 * ta.acuerdo_componente.astype(float).mean(),
                     'acuerdo_principal_filtrado_pct': 100 * ta.acuerdo_principal_filtrado.astype(float).mean(),
                     'azar_principal_filtrado_pct': az_f})
    res.append(fila)
    for s in SINTOMAS:
        en_mod = ta._principal == s['id']
        con_s = t[[s['id'] in e for e in t._elegidos]]
        sint.append({'conjunto': cj, 'sintoma': s['id'], 'etiqueta': s['etiqueta'], 'componente': s['componente'],
                     'modelo_pct': 100 * (t._principal == s['id']).mean(),
                     'modelo_con_alarma_pct': 100 * en_mod.mean() if len(ta) else np.nan,
                     'alarma_critica_pct': 100 * (ta._s_crit == s['id']).mean() if len(ta) else np.nan,
                     'alarma_presente_pct': 100 * np.mean([s['id'] in a for a in ta._s_al]) if len(ta) else np.nan,
                     'coincide_cuando_lo_dice_el_modelo_pct':
                         100 * np.mean([s['id'] in a for a in ta.loc[en_mod, '_s_al']]) if en_mod.any() else np.nan,
                     # el mismo síntoma aparece en firmas distintas y junto a otros síntomas
                     'ventanas_con_el_sintoma': len(con_s), 'firmas_distintas': con_s.firma.nunique(),
                     'junto_a_otros_sintomas_pct': 100 * (con_s.n_sintomas >= 2).mean() if len(con_s) else np.nan})
    for (firma, cat), g in t[t.categoria != 'difusa'].groupby(['firma', 'categoria']):
        reglas = pd.Series([r for r in ', '.join(g.reglas_activas).split(', ') if r], dtype=object)
        comb.append({'conjunto': cj, 'firma': firma, 'categoria': cat, 'ventanas': len(g),
                     'pct_marcadas': 100 * len(g) / len(t), 'sintomas': g.sintomas.iat[0],
                     'variables_fuera_de_la_tabla': g.variables_fuera_de_la_tabla.iat[0],
                     'meses': ', '.join(sorted(g.mes.unique())), 'primera': g.timestamp_inicio.min(),
                     'pct_con_alarma': 100 * g.con_alarma.mean(),
                     'reglas_mas_frecuentes': ', '.join(reglas.value_counts().index[:3]) if len(reglas) else ''})
res = pd.DataFrame(res)
sint = pd.DataFrame(sint)
comb = (pd.DataFrame(comb).sort_values(['conjunto', 'ventanas'], ascending=[False, False]) if comb else
        pd.DataFrame(columns=['conjunto', 'firma', 'categoria', 'ventanas', 'pct_marcadas', 'sintomas',
                              'variables_fuera_de_la_tabla', 'meses', 'primera', 'pct_con_alarma',
                              'reglas_mas_frecuentes']))
guardar(res.round(2), '04i_resumen.csv')
guardar(sint.round(2), '04i_sintomas.csv')
guardar(comb.round(2), '04i_combinaciones.csv')

pd.set_option('display.width', 220)
for f in res.itertuples():
    print(f'\n{"Validación" if f.conjunto == "val" else "Prueba (solo se informa)"}: {f.ventanas_marcadas:,} ventanas '
          f'marcadas ({f.con_alarma:,} con alarma)')
    print(f'  Pareto: la mitad de las ventanas necesita {f.mediana_n_pareto_mse:.0f} variables para el 80 % del MSE '
          f'y {f.mediana_n_pareto_delta:.0f} para el 80 % del ΔMSE; la firma (las dos a la vez) tiene '
          f'{f.mediana_n_firma:.0f}; firmas de 1 a 3 variables: {f.pct_firma_hasta_3_variables:.0f} %')
    print('  ' + ' | '.join(f'{c}: {getattr(f, "pct_" + c.replace(" ", "_")):.0f} %' for c in CATEGORIAS))
    if hasattr(f, 'acuerdo_principal_pct') and pd.notna(f.acuerdo_principal_pct):
        print(f'  Con las alarmas: el síntoma del modelo es uno de los de las alarmas en {f.acuerdo_principal_pct:.0f} % '
              f'(azar {f.azar_principal_pct:.0f} %); es el de la alarma más crítica en {f.acuerdo_estricto_pct:.0f} % '
              f'(azar {f.azar_estricto_pct:.0f} %); mismo componente en {f.acuerdo_componente_pct:.0f} %')
    print(f'  Los dos histogramas: marcan exactamente las mismas variables en {f.pct_paretos_identicos:.0f} % de las '
          f'ventanas; en promedio comparten el {f.coincidencia_paretos_media_pct:.0f} % de sus variables')
    print(f'  La firma cabe en un solo síntoma en {f.pct_firma_en_un_sintoma:.1f} %, en varios a la vez en '
          f'{f.pct_firma_en_varios_sintomas_a_la_vez:.1f} %; necesita dos o más síntomas en '
          f'{f.pct_necesita_varios_sintomas:.0f} %')
    print(f'  Filtrado sucesivo: elige el mismo síntoma principal en {f.pct_filtrado_igual_principal:.0f} %; '
          f'queda con empate en {f.pct_filtrado_con_empate:.0f} %'
          + (f'; acuerdo con las alarmas {f.acuerdo_principal_filtrado_pct:.0f} % '
             f'(azar {f.azar_principal_filtrado_pct:.0f} %)'
             if hasattr(f, 'acuerdo_principal_filtrado_pct') and pd.notna(f.acuerdo_principal_filtrado_pct) else ''))
for cj in ['val', 'test']:
    c = comb[(comb.conjunto == cj) & comb.categoria.isin(['combinación de síntomas', 'no descrita en la tabla'])]
    if len(c):
        print(f'\nFirmas fuera de un síntoma único ({cj}), las más frecuentes:')
        print(c.head(8)[['firma', 'categoria', 'ventanas', 'pct_marcadas', 'variables_fuera_de_la_tabla', 'meses']]
              .round(1).to_string(index=False))


# =============================================================================
# 6. FIGURAS: EL DIBUJO DEL PROFESOR CON VENTANAS REALES, Y EL RESUMEN
# =============================================================================
def elegir_ejemplos():
    """Un caso claro (contenida y de acuerdo con las alarmas) y uno fuera de la tabla, de prueba si hay."""
    base = d[d.conjunto == 'test'] if (d.conjunto == 'test').any() else d
    claro = base[(base.categoria == 'contenida en un síntoma') & (base.acuerdo_principal == True)
                 & base.n_firma.between(2, 4)]
    if claro.empty:
        claro = base[base.categoria.isin(CATEGORIAS[:2])]
    raro = base[base.categoria.isin(['no descrita en la tabla', 'combinación de síntomas'])]
    out = []
    for g in (claro, raro):
        if len(g):
            top = g.firma.value_counts().index[0]                 # la firma más frecuente del grupo
            out.append(g[g.firma == top].sort_values('error').iloc[len(g[g.firma == top]) // 2])
    return out


ejemplos = elegir_ejemplos()
if ejemplos:
    fig, ejes = plt.subplots(len(ejemplos), 2, figsize=(13, 5.2 * len(ejemplos)), squeeze=False,
                             gridspec_kw={'hspace': 0.75, 'wspace': 0.35})
    clave = pd.Series(np.arange(n), index=pd.MultiIndex.from_arrays([m.tramo_id, m.ventana_id]))
    for fila_ej, ej in zip(ejes, ejemplos):
        i = int(clave[(ej.tramo_id, ej.ventana_id)])
        for ax, P, en, titulo in [(fila_ej[0], PMSE[i], EN_MSE[i], '% del MSE de la ventana'),
                                  (fila_ej[1], PDEL[i], EN_DEL[i], '% de la subida del error (ΔMSE)')]:
            o = np.argsort(-P)
            col = [NARANJO if FIRMA[i, j] else (AZUL if en[j] else GRIS) for j in o]
            ax.bar(range(len(o)), P[o], color=col)
            ax2 = ax.twinx()
            ax2.plot(range(len(o)), np.cumsum(P[o]), color='#404040', marker='o', ms=3, lw=1.2)
            ax2.axhline(100 * PARETO, color='#404040', ls='--', lw=0.9)
            ax2.set_ylim(0, 105)
            ax2.set_ylabel('acumulado (%)')
            ax.set_xticks(range(len(o)))
            ax.set_xticklabels([f'{VARIABLES[j]}{FLECHA[SENTIDO[i, j]] if FIRMA[i, j] else ""}' for j in o],
                               rotation=45, fontsize=8)
            ax.set_ylabel('%')
            ax.set_title(titulo, fontsize=10)
        fila_ej[0].annotate(f'{ej.conjunto} · {str(ej.timestamp_inicio)[:16]} · error {ej.veces_umbral:.1f} veces el '
                            f'umbral · firma: {ej.firma} · alarmas: {ej.reglas_activas or "ninguna"}\n'
                            f'→ {ej.categoria}: {ej.sintomas or "—"}', xy=(0, 1.16), xycoords='axes fraction',
                            ha='left', va='bottom', fontsize=9, annotation_clip=False)
    fig.suptitle(f'{EQUIPO}: firma de Pareto ({100 * PARETO:.0f} %) de una ventana y su síntoma en la tabla del experto '
                 '(naranjo: en la firma; azul: solo en un Pareto)', fontsize=11)
    fig.subplots_adjust(top=0.84 if len(ejemplos) > 1 else 0.72, bottom=0.1)
    fig.savefig(f'{RUTA_EQ}/04i_ejemplos.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 2]})
izq = np.zeros(len(res))
for c, col in zip(CATEGORIAS, ['#196B24', '#4EA72E', AZUL, NARANJO, GRIS]):
    v = res[f'pct_{c.replace(" ", "_")}'].to_numpy()
    a1.bar(res.conjunto, v, bottom=izq, color=col, label=c)
    izq += v
a1.set_ylabel('% de las ventanas marcadas')
a1.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
a1.set_title('Cómo se explica la firma', fontsize=10)
st = sint[sint.conjunto == ('test' if (sint.conjunto == 'test').any() else 'val')]
st = st[(st.modelo_con_alarma_pct > 0.5) | (st.alarma_critica_pct > 0.5)].sort_values('alarma_critica_pct')
y = np.arange(len(st))
a2.barh(y - 0.2, st.modelo_con_alarma_pct, 0.4, color=NARANJO, label='síntoma según el modelo')
a2.barh(y + 0.2, st.alarma_critica_pct, 0.4, color=AZUL, label='síntoma de la alarma más crítica')
a2.set_yticks(y)
a2.set_yticklabels([e if len(e) < 70 else e[:67] + '…' for e in st.etiqueta], fontsize=8)
a2.set_xlabel('% de las ventanas marcadas con alarma')
a2.legend(fontsize=8)
a2.set_title('Modelo contra experto, por síntoma', fontsize=10)
fig.suptitle(f'{EQUIPO}: del autoencoder al tipo de falla', fontsize=12)
fig.tight_layout()
fig.savefig(f'{RUTA_EQ}/04i_diagnostico.png', dpi=150)
plt.close(fig)
print('\nGuardados: 04i_tabla_sintomas, 04i_ventanas, 04i_ventanas_variables, 04i_resumen, 04i_sintomas, '
      '04i_combinaciones (csv), 04i_ejemplos y 04i_diagnostico (png).')
