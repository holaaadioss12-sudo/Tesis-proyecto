# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04g — VALIDACIÓN CONTRA LAS ALARMAS (EL JUICIO EXPERTO)
=============================================================================
Lo pedido en la reunión del 15 de noviembre: contrastar con las alarmas lo que dice
el error por variable del autoencoder. Las alarmas (las reglas de la tabla de
criticidad) hacen de juicio experto.
  1. Participación de cada variable en el MSE y cambio relativo de su error
     (anómalas contra normales, en %).
  2. ¿El error de cada variable indica alarma? Veces que supera su umbral P95, en qué
     proporción coincide con una alarma de esa variable y AUC; sí, débil o no se
     decide en validación.
  3. ¿Coincide el componente que señala el modelo con el de las alarmas? Por ventana
     (acuerdo, azar y kappa de Cohen) y en el total (qué porcentaje de las ventanas
     acusa cada componente según el modelo y según las alarmas).
  4. Desacuerdos y una lectura de cada uno. El error cuadrático no tiene signo (sube
     igual si la variable sube o baja) y casi todas las reglas vigilan solo el lado
     alto; cada ventana queda con su mes, la variable, si subió o bajó y si alguna
     regla vigila ese sentido.
Entrada:  {EQUIPO}_error_reconstruccion_W60.csv (paso 04), {EQUIPO}_ventanas_W60.csv y
          {EQUIPO}_features_W60.csv (paso 02), 05a_nivel_ventanas.csv (opcional)
Salida:   04g_indicador_variable.csv, 04g_distribucion.csv, 04g_acuerdo.csv, 04g_confusion.csv,
          04g_desacuerdos.csv, 04g_firmas.csv, 04g_ventanas.csv, 04g_validacion.png
Se decide en validación; prueba solo se informa.
"""
import os
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

warnings.filterwarnings('ignore', category=RuntimeWarning)     # medias de grupos vacíos


# =============================================================================
# 0. CONSTANTES
# =============================================================================
W = 60
PERCENTIL = 95
LLAVES = ['tramo_id', 'ventana_id']
BANDA = 'banda_N_empinada_natural'
NIVELES = ['1_leve', '2_media', '3_grave']
TEXTO_NIVEL = {'0_normal': 'sin alarma', '1_leve': 'leve', '2_media': 'media', '3_grave': 'grave'}
TEXTO_COMP = {'lubricacion': 'lubricación', 'motor': 'motor', 'alimentacion': 'alimentación', 'filtro': 'filtro',
              'eje_principal': 'eje principal', 'socket_liner': 'socket liner', 'contraeje': 'contraeje',
              'tanque_aceite': 'tanque de aceite', 'estructura': 'estructura', 'setting': 'setting'}
DESCRIPCION = {
    'CM': ('Corriente del motor principal', '%'),
    'PI': ('Nivel de material en la alimentación', '%'),
    'PDF': ('Presión diferencial del filtro de aceite', 'kPa'),
    'PEL': ('Presión del aceite de lubricación a la entrada', 'kPa'),
    'T7': ('Temperatura del buje excéntrico', '°C'),
    'T8': ('Temperatura del socket liner', '°C'),
    'T9': ('Temperatura del contraeje', '°C'),
    'T1': ('Temperatura del aceite de retorno', '°C'),
    'T2': ('Temperatura del aceite de entrada (después del enfriador)', '°C'),
    'T5': ('Temperatura del estanque de aceite', '°C'),
    'V1': ('Vibración, punto 1', 'mm/s'),
    'V2': ('Vibración, punto 2', 'mm/s'),
    'V3': ('Vibración, punto 3', 'mm/s'),
    'V4': ('Vibración, punto 4', 'mm/s'),
}
MIN_VENTANAS = 10
BLOQUE = 36                  # ventanas por bloque del bootstrap: 36 x 10 min = 6 h (como el 04c)
N_BOOT = 1000
AUC_SI, AUC_DEBIL = 0.70, 0.60   # 0,7: discriminación aceptable (Hosmer et al., 2013)
Z_FIRMA = 2.0                # |z| desde el cual una variable entra en la firma de una ventana
SIN_VARIABLE = 'sin variable sobre su umbral'
AZUL, NARANJO = '#0F9ED5', '#E97132'

PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
if PUNTO_PARADA not in ('perdida', 'separacion'):
    raise SystemExit(f"PUNTO_PARADA tiene que ser 'perdida' o 'separacion', no {PUNTO_PARADA!r}.")
AE = 'err_AE_normales_sep' if PUNTO_PARADA == 'separacion' else 'err_AE_normales'
ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_NIVEL = f'{RUTA_EQ}/05a_nivel_ventanas.csv'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
for _f in [ARCHIVO_ERROR, ARCHIVO_VENTANAS]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste los pasos 02 y 04?')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


def guardar(df, nombre):
    df.to_csv(salida(nombre), sep=';', decimal=',', index=False, encoding='utf-8-sig')


def variables_de(regla):
    r = POR_ID[regla]
    return [r[c] for c in ('var', 'var2', 'a', 'b') if c in r] + list(r.get('vars', []))


def auc(puntaje, positivo):
    """AUC por rangos (Mann-Whitney); los empates valen 0,5."""
    positivo = np.asarray(positivo, dtype=bool)
    n1, n0 = positivo.sum(), (~positivo).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(puntaje).rank(method='average').to_numpy()
    return (r[positivo].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# =============================================================================
# 1. DATOS
# =============================================================================
print('=' * 72)
print(f'PASO 04g — VALIDACIÓN CONTRA LAS ALARMAS ({EQUIPO})')
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
if os.path.exists(ARCHIVO_NIVEL):
    m = m.drop(columns=[BANDA], errors='ignore').merge(
        pd.read_csv(ARCHIVO_NIVEL, usecols=LLAVES + [BANDA]).drop_duplicates(LLAVES), on=LLAVES, how='left',
        validate='one_to_one')
    nivel = m[BANDA].to_numpy(dtype=object)
else:
    nivel = np.full(len(m), None, dtype=object)
    print('Sin 05a_nivel_ventanas.csv: se omite la separación por nivel.')
MEDIAS = [f'{v}_media' for v in VARIABLES]
HAY_Z = os.path.exists(ARCHIVO_FEATURES)
if HAY_Z:
    _f = pd.read_csv(ARCHIVO_FEATURES, usecols=lambda c: c in LLAVES + MEDIAS).drop_duplicates(LLAVES)
    HAY_Z = all(c in _f.columns for c in MEDIAS)
    if HAY_Z:
        m = m.drop(columns=[c for c in MEDIAS if c in m.columns]).merge(_f, on=LLAVES, how='left',
                                                                         validate='one_to_one')
if not HAY_Z:
    print('Sin las medias del paso 02: no se puede saber si la variable subió o bajó.')

n = len(m)
EV = m[EVAR].to_numpy(dtype=float)                  # error de cada variable en cada ventana
ERR = m.error_reconstruccion.to_numpy(dtype=float)   # error de la ventana: promedio de los 14
CU = m[RC].fillna(0).to_numpy()
ACT = CU > 0
anomala = (m.n_reglas_activas > 0).to_numpy() if 'n_reglas_activas' in m.columns else ACT.any(axis=1)
normal = ~anomala
conj = m.conjunto.to_numpy()
va, te = conj == 'val', conj == 'test'
if not (va & normal).any():
    raise SystemExit('No hay ventanas sin alarma en validación: no se puede fijar el nivel normal.')
ref = np.percentile(EV[va & normal], PERCENTIL, axis=0)          # umbral P95 de cada variable (el del 04)
umbral = float(np.percentile(ERR[va & normal], PERCENTIL))        # umbral P95 del error total
marcada = ERR > umbral
D = EV - ref                                                      # exceso sobre el umbral de cada variable
DPOS = np.clip(D, 0, None)
PCT_MSE = 100 * EV / np.maximum(EV.sum(axis=1, keepdims=True), 1e-12)        # participación en el MSE
PCT_EXC = 100 * DPOS / np.maximum(DPOS.sum(axis=1, keepdims=True), 1e-12)    # participación en el exceso
hay_exceso = D.max(axis=1) > 0
mes = pd.to_datetime(m.timestamp_inicio).dt.strftime('%Y-%m').to_numpy()
bloque = m.tramo_id.to_numpy().astype(np.int64) * 100000 + m.ventana_id.to_numpy() // BLOQUE
print(f'Ventanas: validación {int(va.sum()):,} ({int((va & anomala).sum()):,} con alarma) | '
      f'prueba {int(te.sum()):,} ({int((te & anomala).sum()):,} con alarma)')
print(f'Marcadas por el autoencoder: validación {int((va & marcada).sum()):,} | prueba {int((te & marcada).sum()):,}')


# =============================================================================
# 2. QUÉ VIGILA CADA REGLA: VARIABLES, COMPONENTE Y SENTIDO
# =============================================================================
R2V = np.array([[v in variables_de(r) for v in VARIABLES] for r in RC], dtype=int)
VARS_OK = (ACT.astype(int) @ R2V) > 0              # (n, 14): la variable es de una regla que sonó


def componente_principal(v):
    """Componente de las reglas que miran la variable como variable principal (el del 04c)."""
    for papel in ('var', 'a', 'vars', 'var2', 'b'):
        reglas = [r for r in IDS_REGLAS if r in criticidad.COMPONENTE and
                  (v in POR_ID[r].get('vars', []) if papel == 'vars' else POR_ID[r].get(papel) == v)]
        if reglas:
            c = pd.Series([criticidad.COMPONENTE[r] for r in reglas]).value_counts().index[0]
            return TEXTO_COMP.get(c, c)
    return 'sin componente'


def sentido(op):
    return 'sube' if str(op).startswith('>') else 'baja'


COMP_V = {v: componente_principal(v) for v in VARIABLES}
COMP_R = np.array([TEXTO_COMP.get(criticidad.COMPONENTE.get(r, ''), 'sin componente') for r in RC], dtype=object)
CRIT_R = np.array([criticidad.CRITICIDAD.get(r, 9) for r in RC])
SENTIDOS = {v: set() for v in VARIABLES}           # qué sentido de cada variable vigila alguna regla
for r in RC:
    d = POR_ID[r]
    if d['tipo'] == 'diferencia':                  # a - b > x: vigila que a suba o que b baje
        pares = [(d['a'], sentido(d['op'])), (d['b'], 'baja' if sentido(d['op']) == 'sube' else 'sube')]
    else:
        pares = [(x, sentido(d['op'])) for x in ([d['var']] if 'var' in d else list(d.get('vars', [])))]
        if 'var2' in d:
            pares.append((d['var2'], sentido(d.get('op2', '>'))))
    for x, s in pares:
        if x in SENTIDOS:
            SENTIDOS[x].add(s)
TEXTO_SENTIDOS = {v: ' y '.join(sorted(s, reverse=True)) or 'ninguno' for v, s in SENTIDOS.items()}
COMPONENTES = list(dict.fromkeys([COMP_V[v] for v in VARIABLES] + list(COMP_R)))
print('\nSentido que vigilan las reglas: ' + ', '.join(f'{v} {TEXTO_SENTIDOS[v]}' for v in VARIABLES))

# lo que dicen las alarmas en cada ventana
comps_alarma = [set(COMP_R[ACT[i]]) for i in range(n)]
comp_alarma = np.full(n, '', dtype=object)          # la regla más crítica que sonó; empate: más conteos
for i in np.flatnonzero(ACT.any(axis=1)):
    act = np.flatnonzero(ACT[i])
    top = act[CRIT_R[act] == CRIT_R[act].min()]
    comp_alarma[i] = pd.Series(CU[i, top], index=COMP_R[top]).groupby(level=0).sum().idxmax()
reglas_txt = np.array([', '.join(np.array(RC)[ACT[i]]) for i in range(n)], dtype=object)

# lo que dice el modelo en cada ventana
top_exc = np.argmax(D, axis=1)                      # la variable que más se pasa de su umbral (04c)
top_mse = np.argmax(EV, axis=1)                     # la de mayor error (como el paper)
var_exc_bruta = np.array(VARIABLES, dtype=object)[top_exc]
var_exc = np.where(hay_exceso, var_exc_bruta, '')
var_mse = np.array(VARIABLES, dtype=object)[top_mse]
comp_exc_bruta = np.array([COMP_V[v] for v in var_exc_bruta], dtype=object)
comp_exc = np.where(hay_exceso, comp_exc_bruta, SIN_VARIABLE)
comp_mse = np.array([COMP_V[v] for v in var_mse], dtype=object)

# ¿la variable subió o bajó? z con signo de su media en la ventana, contra las normales de validación
if HAY_Z:
    _med = m[MEDIAS].to_numpy(dtype=float)
    _mu, _sd = np.nanmean(_med[va & normal], axis=0), np.nanstd(_med[va & normal], axis=0, ddof=1)
    ZS = (_med - _mu) / np.where(_sd > 0, _sd, 1.0)
else:
    ZS = np.full(EV.shape, np.nan)
z_top = ZS[np.arange(n), top_exc]
sentido_top = np.where(np.isnan(z_top), '', np.where(z_top > 0, 'sube', 'baja'))
vigilado = np.array([s in SENTIDOS[v] if s else False for v, s in zip(var_exc_bruta, sentido_top)])


# =============================================================================
# 3. ¿EL ERROR DE CADA VARIABLE INDICA ALARMA? (sí / débil / no, decidido en validación)
# =============================================================================
print('\n' + '=' * 72)
print('3. ¿EL ERROR DE CADA VARIABLE INDICA ALARMA DE ESA VARIABLE?')
print('=' * 72)


def remuestreos(idx):
    """Índices (dentro de idx) de N_BOOT remuestreos por bloques de 6 h."""
    grupos = pd.Series(np.arange(len(idx))).groupby(bloque[idx]).indices
    claves = np.array(list(grupos))
    rng = np.random.default_rng(42)
    return [np.concatenate([grupos[k] for k in rng.choice(claves, len(claves))]) for _ in range(N_BOOT)]


def veredicto(a, lo, n_pos):
    if n_pos < MIN_VENTANAS or np.isnan(a):
        return 'sin alarmas suficientes'
    if a >= AUC_SI and lo > 0.5:
        return 'sí'
    if a >= AUC_DEBIL and lo > 0.5:
        return 'débil'
    return 'no'


def pct(a, b):
    return 100 * a / b if b else np.nan


filas_i = []
for cj, c in [('val', va), ('test', te)]:
    idx = np.flatnonzero(c)
    muestras = remuestreos(idx)
    an = anomala[idx]
    for j, v in enumerate(VARIABLES):
        e, prop, sobre = EV[idx, j], VARS_OK[idx, j], EV[idx, j] > ref[j]
        sube = ZS[idx, j] > 0
        a = auc(e, prop)
        boot = np.array([auc(e[s], prop[s]) for s in muestras]) if prop.sum() >= MIN_VENTANAS else np.array([np.nan])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else (np.nan, np.nan)
        en, ea = e[~an].mean(), e[an].mean()
        filas_i.append({
            'conjunto': cj, 'variable': v, 'descripcion': DESCRIPCION.get(v, ('', ''))[0],
            'componente': COMP_V[v], 'sentidos_vigilados': TEXTO_SENTIDOS[v], 'umbral_P95': ref[j],
            'error_normal': en, 'error_anomala': ea, 'delta': ea - en,
            'delta_relativo_pct': pct(ea - en, en), 'veces': ea / en if en else np.nan,
            'participacion_normal_pct': PCT_MSE[idx, j][~an].mean(),
            'participacion_anomala_pct': PCT_MSE[idx, j][an].mean(),
            'ventanas': len(idx), 'ventanas_alarma_propia': int(prop.sum()),
            'base_propia_pct': pct(prop.sum(), len(idx)),
            'veces_sobre_umbral': int(sobre.sum()), 'sobre_umbral_pct': pct(sobre.sum(), len(idx)),
            'precision_propia_pct': pct((sobre & prop).sum(), sobre.sum()),
            'precision_alguna_pct': pct((sobre & an).sum(), sobre.sum()),
            'recall_propia_pct': pct((sobre & prop).sum(), prop.sum()),
            'lift': pct((sobre & prop).sum(), sobre.sum()) / pct(prop.sum(), len(idx)) if prop.sum() else np.nan,
            'sobre_umbral_sube': int((sobre & sube).sum()), 'sobre_umbral_baja': int((sobre & ~sube).sum()),
            'precision_sube_pct': pct((sobre & sube & prop).sum(), (sobre & sube).sum()),
            'precision_baja_pct': pct((sobre & ~sube & prop).sum(), (sobre & ~sube).sum()),
            'auc_propia': a, 'auc_ic_inf': lo, 'auc_ic_sup': hi, 'auc_alguna': auc(e, an),
            'veredicto_conjunto': veredicto(a, lo, int(prop.sum()))})
ind = pd.DataFrame(filas_i)
_ver = ind[ind.conjunto == 'val'].set_index('variable').veredicto_conjunto
ind['veredicto'] = ind.variable.map(_ver)                    # el de validación decide
if not HAY_Z:
    ind[['sobre_umbral_sube', 'sobre_umbral_baja', 'precision_sube_pct', 'precision_baja_pct']] = np.nan
ind['explicacion'] = [
    f'{r.variable} supera su umbral en {r.veces_sobre_umbral} ventanas; en el {r.precision_propia_pct:.0f} % hay '
    f'alarma de la variable (base {r.base_propia_pct:.0f} %) y de sus ventanas con alarma capta el '
    f'{r.recall_propia_pct:.0f} %' if r.veces_sobre_umbral and r.ventanas_alarma_propia else
    f'{r.variable}: sin {"alarmas" if not r.ventanas_alarma_propia else "ventanas sobre su umbral"} en {r.conjunto}'
    for r in ind.itertuples()]
guardar(ind.round(5), '04g_indicador_variable.csv')
_v = ind[ind.conjunto == 'val']
print(_v[['variable', 'componente', 'veces_sobre_umbral', 'precision_propia_pct', 'base_propia_pct',
          'recall_propia_pct', 'delta_relativo_pct', 'auc_propia', 'auc_ic_inf', 'veredicto']]
      .round(2).to_string(index=False))
print(f'\nVeredicto (validación): sí = AUC >= {AUC_SI} con IC sobre 0,5; débil = {AUC_DEBIL} a {AUC_SI}; no = menos.')
print('precision_propia: de las veces que la variable superó su umbral, % con alarma de esa variable;')
print('recall_propia: de las ventanas con alarma de la variable, % en que superó su umbral.')
print('delta_relativo: (error en anómalas − error en normales) / error en normales, en %.')
_r = _v.veredicto.value_counts()
print('En validación: ' + ', '.join(f'{k} = {int(x)}' for k, x in _r.items()) + f' (de {len(_v)} variables).')
# el promedio de los deltas de cada ventana es el delta de los promedios (es lineal)
_d1 = (EV[va & anomala] - EV[va & normal].mean(axis=0)).mean(axis=0)
_d2 = EV[va & anomala].mean(axis=0) - EV[va & normal].mean(axis=0)
print(f'Comprobación: promedio de los deltas por ventana − delta de los promedios = {np.abs(_d1 - _d2).max():.1e}.')


# =============================================================================
# 4. ¿COINCIDE EL COMPONENTE DEL MODELO CON EL DE LAS ALARMAS?
# =============================================================================
print('\n' + '=' * 72)
print('4. COMPONENTE SEGÚN EL MODELO CONTRA COMPONENTE SEGÚN LAS ALARMAS')
print('=' * 72)


CATS = COMPONENTES + [SIN_VARIABLE, '']            # '' = sin alarma
COD = {c: i for i, c in enumerate(CATS)}
MEMB = np.zeros((n, len(CATS)), dtype=bool)          # MEMB[i, c]: sonó una regla del componente c
for i in range(n):
    for c in comps_alarma[i]:
        MEMB[i, COD[c]] = True


def codigos(arr):
    return np.array([COD[x] for x in arr], dtype=int)


def kappa(a, b):
    """kappa de Cohen (1960) entre dos etiquetas por ventana (códigos): kappa, acuerdo observado y por azar."""
    po = float(np.mean(a == b))
    pe = float(np.bincount(a, minlength=len(CATS)) @ np.bincount(b, minlength=len(CATS))) / len(a) ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan, po, pe


def lectura_kappa(k):
    """Escala de Landis y Koch (1977)."""
    if np.isnan(k):
        return ''
    for lim, txt in [(0, 'pobre'), (0.2, 'leve'), (0.4, 'aceptable'), (0.6, 'moderado'), (0.8, 'considerable')]:
        if k <= lim:
            return txt
    return 'casi perfecto'


def azar_amplio(cm, idx):
    """Acuerdo amplio esperado si el modelo eligiera el componente al azar con sus propias frecuencias
    (cm: códigos del componente del modelo en las ventanas idx)."""
    return float((MEMB[idx] @ (np.bincount(cm, minlength=len(CATS)) / len(cm))).mean())


CRITERIOS = {'mayor exceso sobre su umbral (delta)': (top_exc, comp_exc, comp_exc_bruta),
             'mayor % del MSE (como el paper)': (top_mse, comp_mse, comp_mse)}
grupos = {}
for cj, c in [('val', va), ('test', te)]:
    grupos[(cj, 'marcadas con alarma')] = c & marcada & anomala
    grupos[(cj, 'todas las anómalas')] = c & anomala
    for nv in NIVELES:
        grupos[(cj, f'marcadas con alarma · {TEXTO_NIVEL[nv]}')] = c & marcada & anomala & (nivel == nv)
filas_a = []
for crit, (top, comp, comp_bruta) in CRITERIOS.items():
    dist0 = np.bincount(top[va & normal], minlength=len(VARIABLES)) / max(int((va & normal).sum()), 1)
    for (cj, gn), s in grupos.items():
        idx = np.flatnonzero(s)
        if len(idx) < MIN_VENTANAS:
            continue
        cm = codigos((comp_bruta if gn == 'todas las anómalas' else comp)[idx])   # sin marca: la que más se acerca
        ca = codigos(comp_alarma[idx])
        ok_var = VARS_OK[idx, top[idx]]
        az_var = (VARS_OK[idx] * dist0).sum(axis=1)
        amplio = MEMB[idx, cm]
        k, po, pe = kappa(cm, ca)
        pa = azar_amplio(cm, idx)
        bk, bv = [], []
        for t in remuestreos(idx):
            bk.append(kappa(cm[t], ca[t])[0])
            bv.append(100 * (amplio[t].mean() - azar_amplio(cm, idx[t])))
        pm, pa_d = np.bincount(cm, minlength=len(CATS)) / len(cm), np.bincount(ca, minlength=len(CATS)) / len(ca)
        filas_a.append({
            'conjunto': cj, 'grupo': gn, 'criterio': crit, 'ventanas': len(idx),
            'acuerdo_variable_pct': 100 * ok_var.mean(), 'azar_variable_pct': 100 * az_var.mean(),
            'acuerdo_estricto_pct': 100 * po, 'azar_estricto_pct': 100 * pe,
            'kappa': k, 'kappa_ic_inf': np.nanpercentile(bk, 2.5), 'kappa_ic_sup': np.nanpercentile(bk, 97.5),
            'lectura_kappa': lectura_kappa(k),
            'acuerdo_amplio_pct': 100 * amplio.mean(), 'azar_amplio_pct': 100 * pa,
            'ventaja_amplio': 100 * (amplio.mean() - pa),
            'ventaja_ic_inf': np.nanpercentile(bv, 2.5), 'ventaja_ic_sup': np.nanpercentile(bv, 97.5),
            'distancia_distribuciones_pct': 50 * np.abs(pm - pa_d).sum()})
acu = pd.DataFrame(filas_a)
guardar(acu.round(4), '04g_acuerdo.csv')
_p = acu[acu.grupo.isin(['marcadas con alarma', 'todas las anómalas'])]
print(_p[['conjunto', 'grupo', 'criterio', 'ventanas', 'acuerdo_amplio_pct', 'azar_amplio_pct', 'acuerdo_estricto_pct',
          'kappa', 'lectura_kappa', 'acuerdo_variable_pct', 'azar_variable_pct']].round(2).to_string(index=False))
print('\nacuerdo_amplio: el componente del modelo es uno de los de las alarmas que sonaron. acuerdo_estricto: es')
print('el de la alarma más crítica. kappa: acuerdo estricto descontado el azar (Cohen, 1960; escala de Landis')
print('y Koch, 1977). acuerdo_variable: la variable del modelo es de una regla que sonó (como el 04c).')

# distribución en el total: qué % de las ventanas acusa cada componente según el modelo y según las alarmas
filas_d = []
for cj, c in [('val', va), ('test', te)]:
    for gn, s in [('marcadas con alarma', c & marcada & anomala), ('marcadas (todas)', c & marcada),
                  ('todas las anómalas', c & anomala)]:
        idx = np.flatnonzero(s)
        if not len(idx):
            continue
        cm = comp_exc_bruta[idx] if gn == 'todas las anómalas' else comp_exc[idx]
        for cp in COMPONENTES + [SIN_VARIABLE, 'sin alarma']:
            cols = [j for j, v in enumerate(VARIABLES) if COMP_V[v] == cp]
            f = {'conjunto': cj, 'grupo': gn, 'componente': cp, 'ventanas_grupo': len(idx),
                 'modelo_exceso_pct': 100 * np.mean(cm == cp),
                 'modelo_mse_pct': 100 * np.mean(comp_mse[idx] == cp),
                 'participacion_mse_pct': PCT_MSE[np.ix_(idx, cols)].sum(axis=1).mean() if cols else 0.0,
                 'participacion_exceso_pct': PCT_EXC[np.ix_(idx, cols)].sum(axis=1).mean() if cols else 0.0,
                 'alarma_dominante_pct': 100 * np.mean(comp_alarma[idx] == ('' if cp == 'sin alarma' else cp)),
                 'alarma_presente_pct': 100 * (np.mean(~anomala[idx]) if cp == 'sin alarma'
                                               else np.mean([cp in comps_alarma[i] for i in idx]))}
            f['diferencia_pct'] = f['modelo_exceso_pct'] - f['alarma_dominante_pct']
            if any(f[x] > 0 for x in ['modelo_exceso_pct', 'modelo_mse_pct', 'alarma_dominante_pct',
                                      'alarma_presente_pct', 'participacion_mse_pct']):
                filas_d.append(f)
dist = pd.DataFrame(filas_d)
guardar(dist.round(3), '04g_distribucion.csv')
for cj in ['val', 'test']:
    t = dist[(dist.conjunto == cj) & (dist.grupo == 'marcadas con alarma')].sort_values('alarma_dominante_pct',
                                                                                      ascending=False)
    if len(t):
        print(f'\n{"Validación" if cj == "val" else "Prueba"} · marcadas con alarma ({int(t.ventanas_grupo.iloc[0]):,} '
              'ventanas): % de ventanas por componente')
        print(t[['componente', 'modelo_exceso_pct', 'modelo_mse_pct', 'participacion_mse_pct', 'alarma_dominante_pct',
                 'alarma_presente_pct']].round(1).to_string(index=False))
print('\nmodelo_exceso: la variable que más se pasa de su umbral es de ese componente (la etiqueta del 04c).')
print('modelo_mse: la de mayor % del MSE. participacion_mse: promedio, entre las ventanas, del % del MSE que')
print('suman las variables del componente. alarma_dominante: la regla más crítica que sonó es de ese')
print('componente. alarma_presente: sonó al menos una regla de ese componente (puede sumar más de 100).')

# matriz de confusión del criterio del 04c en las marcadas con alarma
filas_c = []
for cj, c in [('val', va), ('test', te)]:
    s = c & marcada & anomala
    if not s.any():
        continue
    t = pd.crosstab(pd.Series(comp_exc[s], name='componente_modelo'), pd.Series(comp_alarma[s], name='componente_alarma'))
    for cm_, fila in t.iterrows():
        for ca_, k in fila.items():
            if k:
                filas_c.append({'conjunto': cj, 'componente_modelo': cm_, 'componente_alarma': ca_, 'ventanas': int(k),
                                'pct_fila': 100 * k / fila.sum(), 'pct_total': 100 * k / s.sum()})
conf = pd.DataFrame(filas_c)
guardar(conf.round(2), '04g_confusion.csv')


# =============================================================================
# 5. DESACUERDOS: CUÁNDO NO COINCIDE Y POR QUÉ
# =============================================================================
print('\n' + '=' * 72)
print('5. DESACUERDOS: CUÁNDO NO COINCIDE Y UNA LECTURA DE POR QUÉ')
print('=' * 72)
amplio = MEMB[np.arange(n), codigos(comp_exc)]
misma_var = hay_exceso & VARS_OK[np.arange(n), top_exc]
caso = np.select([~marcada & ~anomala, ~marcada & anomala, marcada & ~anomala, amplio, misma_var],
                 ['normal no marcada', 'alarma no detectada', 'marcada sin alarma', 'coincide',
                  'misma variable, otro componente'], 'otro componente')


def lectura(i):
    """Texto de la ventana marcada y su tipo (para contarlos)."""
    if not marcada[i]:
        return '', ''
    if caso[i] == 'coincide':
        return f'coincide con las alarmas ({comp_exc[i]})', 'coincide'
    if not hay_exceso[i]:
        return 'ninguna variable pasa su umbral; sube el error total', 'ninguna variable pasa su umbral'
    v, s = var_exc[i], sentido_top[i]
    if caso[i] == 'misma variable, otro componente':
        return (f'{v} es de la regla que sonó; la tabla asigna esa regla a {comp_alarma[i]}',
                'misma variable, la tabla la asigna a otro componente')
    if not s:
        return f'{v}: sin dato de si subió o bajó', 'sin dato del sentido'
    verbo = 'subió' if s == 'sube' else 'bajó'
    if not vigilado[i]:
        return (f'{v} {verbo} y ninguna regla vigila ese sentido: el MSE sube igual (no tiene signo)',
                f'{verbo} y ninguna regla vigila ese sentido')
    return (f'{v} {verbo} sin llegar al límite de su regla: posible aviso temprano o cambio de operación',
            f'{verbo} sin llegar al límite de su regla')


def firma(i):
    """Las variables más alejadas de lo normal en la ventana (|z| >= Z_FIRMA), con su sentido."""
    if not HAY_Z:
        return ''
    z = ZS[i]
    orden = [j for j in np.argsort(-np.abs(np.nan_to_num(z))) if abs(z[j]) >= Z_FIRMA][:3]
    return ' · '.join(f'{VARIABLES[j]} {"alta" if z[j] > 0 else "baja"}' for j in orden) or 'ninguna fuera de lo normal'


_lec = [lectura(i) for i in range(n)]
lecturas = np.array([x[0] for x in _lec], dtype=object)
tipo_lectura = np.array([x[1] for x in _lec], dtype=object)
firmas = np.array([firma(i) if marcada[i] or anomala[i] else '' for i in range(n)], dtype=object)

ven = pd.DataFrame({
    'tramo_id': m.tramo_id, 'ventana_id': m.ventana_id, 'timestamp_inicio': m.timestamp_inicio, 'mes': mes,
    'conjunto': conj, 'nivel': pd.Series(nivel).map(TEXTO_NIVEL).fillna('sin nivel').to_numpy(),
    'anomala': anomala.astype(int), 'marcada': marcada.astype(int), 'error_reconstruccion': ERR,
    'variable_modelo': var_exc, 'componente_modelo': comp_exc,
    'pct_exceso_variable': np.where(hay_exceso, PCT_EXC[np.arange(n), top_exc], np.nan),
    'variable_mse': var_mse, 'componente_mse': comp_mse, 'pct_mse_variable': PCT_MSE[np.arange(n), top_mse],
    'sentido': np.where(hay_exceso, sentido_top, ''), 'z_media': np.where(hay_exceso, z_top, np.nan),
    'sentido_vigilado': np.where(hay_exceso & (sentido_top != ''), vigilado.astype(int), np.nan),
    'reglas': reglas_txt, 'componente_alarma': comp_alarma,
    'componentes_alarma': [', '.join(sorted(s)) for s in comps_alarma],
    'acuerdo_variable': misma_var.astype(int), 'acuerdo_estricto': (comp_exc == comp_alarma).astype(int),
    'acuerdo_amplio': amplio.astype(int), 'caso': caso, 'lectura': lecturas, 'tipo_lectura': tipo_lectura,
    'firma': firmas})
guardar(ven.round(6), '04g_ventanas.csv')

_c = ven[ven.caso != 'normal no marcada']
des = (_c.groupby(['conjunto', 'caso', 'mes', 'variable_modelo', 'componente_modelo', 'sentido', 'sentido_vigilado',
                   'componente_alarma'], dropna=False).size().rename('ventanas').reset_index())
guardar(des, '04g_desacuerdos.csv')
fir = (ven[ven.caso.isin(['otro componente', 'marcada sin alarma'])]
       .groupby(['conjunto', 'caso', 'firma'])
       .agg(ventanas=('mes', 'size'), meses=('mes', lambda x: ', '.join(sorted(set(x)))),
            primera=('timestamp_inicio', 'min'), variable_modelo=('variable_modelo', lambda x: x.mode().iat[0]))
       .reset_index().sort_values(['conjunto', 'caso', 'ventanas'], ascending=[True, True, False]))
fir = fir.groupby(['conjunto', 'caso']).head(15)
guardar(fir, '04g_firmas.csv')

for cj in ['val', 'test']:
    t = ven[(ven.conjunto == cj) & (ven.marcada == 1)]
    if not len(t):
        continue
    print(f'\n{"Validación" if cj == "val" else "Prueba"}: {len(t):,} ventanas marcadas')
    print((100 * t.caso.value_counts(normalize=True)).round(1).rename('%').to_string())
    print('Lecturas más frecuentes de las que no coinciden:')
    print(t[~t.caso.isin(['coincide'])].tipo_lectura.value_counts().head(6).to_string())
print('\n"bajó y ninguna regla vigila ese sentido": el caso de la temperatura que baja; el MSE sube varias')
print('veces sin que haya alarma posible, porque el error al cuadrado no distingue subir de bajar.')
print('"sin llegar al límite de su regla": la variable se aleja de lo normal en un sentido vigilado; puede')
print('anticipar una alarma o reflejar un cambio de operación. Revisar con 04g_firmas.csv y el mes.')


# =============================================================================
# 6. GRÁFICO
# =============================================================================
fig, ejes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={'width_ratios': [1, 1, 1.1]})
for ax, cj in zip(ejes[:2], ['val', 'test']):
    t = dist[(dist.conjunto == cj) & (dist.grupo == 'marcadas con alarma')]
    t = t[(t.modelo_exceso_pct > 0) | (t.alarma_dominante_pct > 0)].sort_values('alarma_dominante_pct')
    y = np.arange(len(t))
    ax.barh(y + 0.2, t.modelo_exceso_pct, height=0.38, color=AZUL, label='Modelo: variable que más se pasa')
    ax.barh(y - 0.2, t.alarma_dominante_pct, height=0.38, color=NARANJO, label='Alarmas: la más crítica que sonó')
    for yy, a, b in zip(y, t.modelo_exceso_pct, t.alarma_dominante_pct):
        ax.text(a + 0.8, yy + 0.2, f'{a:.0f}', va='center', fontsize=8, color='#333333')
        ax.text(b + 0.8, yy - 0.2, f'{b:.0f}', va='center', fontsize=8, color='#333333')
    ax.set_yticks(y)
    ax.set_yticklabels(t.componente)
    ax.set_xlim(0, 1.12 * np.nanmax([t.modelo_exceso_pct.max(), t.alarma_dominante_pct.max(), 1.0]))
    n_g = int(t.ventanas_grupo.iloc[0]) if len(t) else 0
    ax.set(xlabel='% de las ventanas marcadas con alarma',
           title=f'{"Validación" if cj == "val" else "Prueba (solo se informa)"}: {n_g:,} ventanas')
    ax.grid(alpha=0.3, axis='x')
    ax.legend(fontsize=8, loc='lower right')
ax = ejes[2]
_iv = ind[ind.conjunto == 'val']
t = _iv[(_iv.veredicto != 'sin alarmas suficientes') & _iv.auc_propia.notna()].sort_values('auc_propia')
sin_datos = _iv.loc[~_iv.variable.isin(t.variable), 'variable'].tolist()
y = np.arange(len(t))
ax.barh(y, t.auc_propia, height=0.6, color=AZUL)
ax.errorbar(t.auc_propia, y, xerr=[(t.auc_propia - t.auc_ic_inf).clip(lower=0).fillna(0),
                                   (t.auc_ic_sup - t.auc_propia).clip(lower=0).fillna(0)],
            fmt='none', ecolor='#333333', capsize=2, lw=1)
for yy, vd in zip(y, t.veredicto):
    ax.text(1.02, yy, vd, transform=ax.get_yaxis_transform(), va='center', fontsize=8, color='#333333')
for x0, txt in [(0.5, 'azar'), (AUC_DEBIL, 'débil'), (AUC_SI, 'sí')]:
    ax.axvline(x0, color='#7F7F7F', ls=':' if x0 == 0.5 else '--', lw=1)
    ax.text(x0, 1.01, txt, transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#595959')
ax.set_yticks(y)
ax.set_yticklabels(t.variable)
ax.set_xlim(0.3, 1.0)
ax.set_xlabel('AUC del error de la variable para su propia alarma (validación)'
              + (f'\nSin alarmas suficientes: {", ".join(sin_datos)}' if sin_datos else ''))
ax.set_title('¿El error de cada variable indica su alarma?', pad=16)
ax.grid(alpha=0.3, axis='x')
fig.suptitle(f'{EQUIPO}: el modelo contra las alarmas', fontsize=11)
fig.tight_layout()
fig.savefig(salida('04g_validacion.png'), dpi=150)
plt.close(fig)

print(f'\nGuardado en {RUTA_EQ}: 04g_indicador_variable.csv, 04g_distribucion.csv, 04g_acuerdo.csv,')
print('04g_confusion.csv, 04g_desacuerdos.csv, 04g_firmas.csv, 04g_ventanas.csv y 04g_validacion.png.')
