# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04c — DELTA DEL ERROR POR VARIABLE Y LOCALIZACIÓN EXPLICADA
=============================================================================
Lo pedido en la reunión del 6 de noviembre, siguiendo a Holly et al. (2022):
delta del error de cada variable entre ventanas anómalas y normales, variables
que explican cada ventana marcada, tabla de consulta para el experto, métricas
y consistencia en el tiempo.
Entrada:  {EQUIPO}_error_reconstruccion_W60.csv (paso 04), {EQUIPO}_ventanas_W60.csv y
          {EQUIPO}_features_W60.csv (paso 02), 05a_nivel_ventanas.csv (opcional: separa por nivel)
Salida:   04c_delta_variables.csv/png, 04c_atribucion.csv, 04c_ae_vs_z.csv, 04c_tabla_experto.csv,
          04c_causa_raiz.csv, 04c_etiquetas_resumen.csv, 04c_metricas.csv, 04c_deteccion_nivel.csv,
          04c_consistencia.csv, 04c_localizacion_ejemplos.png, 04c_error_alrededor_inicios.csv,
          04c_error_en_el_tiempo.png, 04c_error_promedio_inicios.png,
          04c_localizacion_ventanas.csv (lo usa el paso 05e)
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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from sklearn.metrics import roc_auc_score

import criticidad
from reglas_engine import POR_ID, IDS as IDS_REGLAS


# =============================================================================
# 0. CONSTANTES
# =============================================================================
W = 60
PERCENTIL = 95
M_SIG = 3                    # variables más significativas por ventana (algoritmo 1 del paper)
GAPS = (1, 3, 6)             # ventanas sin marca toleradas dentro de una misma anomalía
MIN_VENTANAS = 10
LLAVES = ['tramo_id', 'ventana_id']
BANDA = 'banda_N_empinada_natural'
NIVELES = ['1_leve', '2_media', '3_grave']
TEXTO_NIVEL = {'0_normal': 'sin alarma', '1_leve': 'leve', '2_media': 'media', '3_grave': 'grave'}
TEXTO_COMP = {'lubricacion': 'lubricación', 'motor': 'motor', 'alimentacion': 'alimentación', 'filtro': 'filtro',
              'eje_principal': 'eje principal', 'socket_liner': 'socket liner', 'contraeje': 'contraeje',
              'tanque_aceite': 'tanque de aceite', 'estructura': 'estructura', 'setting': 'setting'}
FAMILIAS = {
    'lubricacion': ['D', 'A5', 'E'],
    'corriente':   ['A', 'A1', 'A4'],
    'atollo':      ['B', 'A3'],
    'filtro':      ['C'],
    'temperatura': ['F', 'A7', 'A6', 'A10', 'A9', 'A8', 'I', 'A11', 'A13', 'A12',
                    'K', 'A15', 'A14', 'A16', 'L', 'A21', 'A22'],
    'vibracion':   ['O', 'A20', 'A19', 'A18', 'A17'],
}
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


# =============================================================================
# 1. DATOS
# =============================================================================
print('=' * 72)
print(f'PASO 04c — DELTA DEL ERROR Y LOCALIZACIÓN EXPLICADA ({EQUIPO})')
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
HAY_NIVEL = os.path.exists(ARCHIVO_NIVEL)
if HAY_NIVEL:
    m = m.drop(columns=[BANDA], errors='ignore').merge(
        pd.read_csv(ARCHIVO_NIVEL, usecols=LLAVES + [BANDA]).drop_duplicates(LLAVES), on=LLAVES, how='left',
        validate='one_to_one')
    nivel = m[BANDA].to_numpy(dtype=object)
else:
    nivel = np.full(len(m), None, dtype=object)
    print('Sin 05a_nivel_ventanas.csv: se omite la separación por nivel (corre el 05a para tenerla).')

MEDIAS = [f'{v}_media' for v in VARIABLES]
HAY_Z = os.path.exists(ARCHIVO_FEATURES)
if HAY_Z:
    _f = pd.read_csv(ARCHIVO_FEATURES, usecols=lambda c: c in LLAVES + MEDIAS).drop_duplicates(LLAVES)
    HAY_Z = all(c in _f.columns for c in MEDIAS)
    if HAY_Z:
        m = m.drop(columns=[c for c in MEDIAS if c in m.columns]).merge(_f, on=LLAVES, how='left',
                                                                         validate='one_to_one')
if not HAY_Z:
    print('Sin las medias del paso 02: se omite la localización sin autoencoder (z de las medias).')

EV = m[EVAR].to_numpy(dtype=float)                 # error de cada variable (ecuación 9 del paper)
ERR = m.error_reconstruccion.to_numpy(dtype=float)  # error total (ecuación 7)
if not np.allclose(EV.mean(axis=1), ERR, rtol=1e-3, atol=1e-7):
    raise SystemExit('Los evar_ no promedian el error total: el paso 04 no los calculó con el mismo modelo.')
CU = m[RC].fillna(0).to_numpy()
ACT = CU > 0
anomala = (m.n_reglas_activas > 0).to_numpy() if 'n_reglas_activas' in m.columns else ACT.any(axis=1)
normal = ~anomala
conj = m.conjunto.to_numpy()
va, te = conj == 'val', conj == 'test'
if not (va & normal).any():
    raise SystemExit('No hay ventanas sin alarma en validación: no se puede fijar el nivel normal.')
FAM = np.column_stack([CU[:, [RC.index(r) for r in rs if r in RC]].sum(axis=1)
                       if any(r in RC for r in rs) else np.zeros(len(m)) for rs in FAMILIAS.values()])
familia = np.where(FAM.sum(axis=1) > 0, np.array(list(FAMILIAS))[FAM.argmax(axis=1)], 'sin_alarma')

ref = np.percentile(EV[va & normal], PERCENTIL, axis=0)         # umbral P95 de cada variable (el del 04)
umbral = float(np.percentile(ERR[va & normal], PERCENTIL))       # umbral P95 del error total
marcada = ERR > umbral
D = EV - ref                                                     # exceso de cada variable sobre su umbral
if HAY_Z:                                                        # línea base sin autoencoder
    _med = m[MEDIAS].to_numpy(dtype=float)
    _mu, _sd = np.nanmean(_med[va & normal], axis=0), np.nanstd(_med[va & normal], axis=0, ddof=1)
    Z = np.nan_to_num(np.abs((_med - _mu) / np.where(_sd > 0, _sd, 1.0)), nan=0.0)
    ref_z = np.percentile(Z[va & normal], PERCENTIL, axis=0)
print(f'Ventanas: validación {va.sum():,} | prueba {te.sum():,} | variables: {len(VARIABLES)}')
print(f'Umbral P{PERCENTIL} del error total (ventanas sin alarma de validación): {umbral:.5f}')
print(f'Marcadas por el autoencoder: validación {(va & marcada).sum():,} | prueba {(te & marcada).sum():,}')


# =============================================================================
# 2. DELTA DEL ERROR POR VARIABLE: MSE DE LAS ANÓMALAS − MSE DE LAS NORMALES
# =============================================================================
print('\n' + '=' * 72)
print('2. DELTA DEL ERROR POR VARIABLE (anómalas − normales, en el mismo conjunto)')
print('=' * 72)
PRES = FAM > 0


def grupos_de(c):
    g = {'todas las anómalas': c & anomala}
    if HAY_NIVEL:
        g.update({TEXTO_NIVEL[nv]: c & (nivel == nv) for nv in NIVELES})
    for j, f in enumerate(FAMILIAS):
        g[f'solo {f}'] = c & PRES[:, j] & (PRES.sum(axis=1) == 1)
    return g


filas_d = []
for cj, c in [('val', va), ('test', te)]:
    base = c & normal
    if not base.any():
        continue
    mse_n = EV[base].mean(axis=0)
    for gn, s in grupos_de(c).items():
        if s.sum() < MIN_VENTANAS:
            continue
        mse_g = EV[s].mean(axis=0)
        d = mse_g - mse_n
        aporte = (EV[s] / EV[s].sum(axis=1, keepdims=True)).mean(axis=0)
        for j, v in enumerate(VARIABLES):
            filas_d.append({'conjunto': cj, 'grupo': gn, 'ventanas': int(s.sum()), 'variable': v,
                            'mse_normal': mse_n[j], 'mse_grupo': mse_g[j], 'delta': d[j],
                            'pct_delta': 100 * d[j] / d.sum() if d.sum() > 0 else np.nan,
                            'veces': mse_g[j] / mse_n[j] if mse_n[j] > 0 else np.nan,
                            'pct_error': 100 * aporte[j]})
delta = pd.DataFrame(filas_d)
guardar(delta.round(6), '04c_delta_variables.csv')

for cj in ('val', 'test'):
    t = delta[(delta.conjunto == cj) & (delta.grupo == 'todas las anómalas')].sort_values('delta', ascending=False)
    if t.empty:
        continue
    print(f'\n{"Validación" if cj == "val" else "Prueba (solo se informa)"}: {t.ventanas.iloc[0]:,} ventanas '
          f'anómalas; Δ del error total = {t.delta.mean():.5f}')
    print(t[['variable', 'mse_normal', 'mse_grupo', 'delta', 'pct_delta', 'veces', 'pct_error']]
          .head(8).round(4).to_string(index=False))
print('\nΔ: MSE de la variable en las anómalas − en las normales. % del Δ suma 100: el Δ del error')
print('total es el promedio de los 14 Δ; si alguna variable baja (Δ negativo), otra puede pasar de 100 %.')
print('pct_error: parte del error que pone la variable; incluye su error normal, así que una variable')
print('con error normal alto pesa aunque cambie poco. El Δ descuenta ese nivel normal.')
for cj in ('val', 'test'):
    t = delta[(delta.conjunto == cj) & (delta.grupo != 'todas las anómalas')]
    if t.empty:
        continue
    print(f'\n{"Validación" if cj == "val" else "Prueba"}: variables con más % del Δ por grupo')
    for gn, s in t.groupby('grupo', sort=False):
        if s.delta.sum() <= 0:
            print(f'  {gn:<20} ({s.ventanas.iloc[0]:>5,}): Δ total negativo, tiene menos error que las normales')
            continue
        top = s.sort_values('delta', ascending=False).head(3)
        print(f'  {gn:<20} ({s.ventanas.iloc[0]:>5,}): ' + ', '.join(
            f'{r.variable} {r.pct_delta:.0f} %' for r in top.itertuples()))

fig, ax = plt.subplots(figsize=(11, 4.6))
x = np.arange(len(VARIABLES))
for k, (cj, col) in enumerate([('val', '#9ecae1'), ('test', '#3182bd')]):
    t = delta[(delta.conjunto == cj) & (delta.grupo == 'todas las anómalas')].set_index('variable').reindex(VARIABLES)
    ax.bar(x + (k - 0.5) * 0.38, t.pct_delta, 0.38, color=col,
           label=f'% del Δ ({"validación" if cj == "val" else "prueba"})')
    if cj == 'test':
        ax.scatter(x + 0.19, t.pct_error, marker='_', s=220, color='k', linewidths=2.2,
                   label='% del error (prueba)', zorder=3)
ax.axhline(100 / len(VARIABLES), ls=':', color='gray', lw=1)
ax.set_xticks(x)
ax.set_xticklabels(VARIABLES)
ax.set_ylabel('%')
ax.set_title(f'{EQUIPO}: qué variables explican que el error suba en las ventanas anómalas')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(salida('04c_delta_variables.png'), dpi=150)
plt.close(fig)


# =============================================================================
# 3. LOCALIZACIÓN: LAS VARIABLES QUE MÁS SE PASAN DE SU UMBRAL (ALGORITMO 1)
# =============================================================================
print('\n' + '=' * 72)
print('3. LOCALIZACIÓN EN CADA VENTANA MARCADA: VARIABLES QUE MÁS SE PASAN DE SU UMBRAL')
print('=' * 72)
n = len(m)
orden = np.argsort(-D, axis=1)
Dpos = np.clip(D, 0, None)
suma = Dpos.sum(axis=1)
for k in range(M_SIG):
    j = orden[:, k]
    dk = D[np.arange(n), j]
    ok = marcada & (dk > 0)
    m[f'var_delta_{k + 1}'] = np.where(ok, np.array(VARIABLES)[j], '')
    m[f'delta_{k + 1}'] = np.where(ok, dk, np.nan)
    m[f'pct_exceso_{k + 1}'] = np.where(ok, 100 * dk / np.maximum(suma, 1e-12), np.nan)
m['n_sobre_umbral'] = (D > 0).sum(axis=1)
print(f'Ventanas marcadas sin ninguna variable sobre su umbral: {int((marcada & (suma == 0)).sum()):,}')
print(f'Variables sobre su umbral en las marcadas (mediana): '
      f'{np.median(m.n_sobre_umbral[marcada]) if marcada.any() else 0:.0f} de {len(VARIABLES)}')
print('Como el algoritmo 1 del paper, pero ordenando por el exceso sobre el umbral de cada variable')
print('en vez del error bruto, para que una variable con error normal alto no gane siempre.')


# =============================================================================
# 4. ¿LOCALIZA BIEN? LA VARIABLE ELEGIDA CONTRA LA REGLA QUE SONÓ
# =============================================================================
print('\n' + '=' * 72)
print('4. ¿LA VARIABLE ELEGIDA ES LA DE LA REGLA QUE SONÓ? (con y sin autoencoder)')
print('=' * 72)
R2V = np.array([[v in variables_de(r) for v in VARIABLES] for r in RC], dtype=int)
VARS_OK = (ACT.astype(int) @ R2V) > 0
CRITERIOS = {'mayor error (como el paper)': np.argmax(EV, axis=1),
             'mayor error sobre su umbral': np.argmax(EV / np.maximum(ref, 1e-12), axis=1),
             'mayor delta (exceso sobre su umbral)': np.argmax(D, axis=1)}
CON_AE = set(CRITERIOS)
if HAY_Z:
    CRITERIOS['mayor z de la media (sin autoencoder)'] = np.argmax(Z, axis=1)
    CRITERIOS['mayor z sobre su P95 (sin autoencoder)'] = np.argmax(Z - ref_z, axis=1)
filas_a = []
for crit, top in CRITERIOS.items():
    dist0 = np.bincount(top[va & normal], minlength=len(VARIABLES)) / max(int((va & normal).sum()), 1)
    for cj, c in [('val', va), ('test', te)]:
        for gn, s in [('todas las anómalas', c & anomala), ('marcadas por el AE', c & anomala & marcada)]:
            if not s.any():
                continue
            ok = VARS_OK[s]
            ac = 100 * ok[np.arange(int(s.sum())), top[s]].mean()
            az = 100 * (ok * dist0).sum(axis=1).mean()
            filas_a.append({'criterio': crit, 'conjunto': cj, 'grupo': gn, 'ventanas': int(s.sum()),
                            'acierta_pct': round(ac, 1), 'azar_pct': round(az, 1), 'ventaja': round(ac - az, 1)})
atr = pd.DataFrame(filas_a)
guardar(atr, '04c_atribucion.csv')
print(atr.to_string(index=False))
_v = atr[(atr.conjunto == 'val') & (atr.grupo == 'todas las anómalas')]
if len(_v):
    _ae = _v[_v.criterio.isin(CON_AE)]
    print(f'\nMás ventaja sobre el azar en validación: {_ae.loc[_ae.ventaja.idxmax(), "criterio"]}')
print('acierta_pct: la variable elegida es una de las que mira la regla que sonó. azar_pct: lo que')
print('acertaría eligiendo como en las ventanas normales. ventaja = acierta − azar.')

# ¿el autoencoder localiza mejor que el z de las medias?
if HAY_Z and len(_v):
    DIST0 = {c: np.bincount(t[va & normal], minlength=len(VARIABLES)) / max(int((va & normal).sum()), 1)
             for c, t in CRITERIOS.items()}
    _bloque = m.tramo_id.to_numpy().astype(np.int64) * 100000 + m.ventana_id.to_numpy() // 36
    _rng = np.random.default_rng(42)

    def ventaja_ventanas(c, idx):
        ok = VARS_OK[idx]
        return 100 * (ok[np.arange(len(idx)), CRITERIOS[c][idx]] - (ok * DIST0[c]).sum(axis=1))

    _ae, _z = _v[_v.criterio.isin(CON_AE)], _v[~_v.criterio.isin(CON_AE)]
    MEJOR_AE, MEJOR_Z = _ae.loc[_ae.ventaja.idxmax(), 'criterio'], _z.loc[_z.ventaja.idxmax(), 'criterio']
    print(f'\n¿EL AUTOENCODER LOCALIZA MEJOR QUE EL z DE LAS MEDIAS? (elegidos en validación)')
    print(f'  con autoencoder: {MEJOR_AE} | sin autoencoder: {MEJOR_Z}')
    filas_z = []
    for cj, c in [('val', va), ('test', te)]:
        for gn, s in [('todas las anómalas', c & anomala), ('marcadas por el AE', c & anomala & marcada)]:
            idx = np.flatnonzero(s)
            if len(idx) < MIN_VENTANAS:
                continue
            dif = ventaja_ventanas(MEJOR_AE, idx) - ventaja_ventanas(MEJOR_Z, idx)
            grupos = pd.Series(np.arange(len(idx))).groupby(_bloque[idx]).indices
            claves = np.array(list(grupos))
            vals = [dif[np.concatenate([grupos[k] for k in _rng.choice(claves, len(claves))])].mean()
                    for _ in range(1000)]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            filas_z.append({'conjunto': cj, 'grupo': gn, 'ventanas': len(idx), 'con_autoencoder': MEJOR_AE,
                            'sin_autoencoder': MEJOR_Z, 'dif_ventaja': round(dif.mean(), 1),
                            'ic_inf': round(lo, 1), 'ic_sup': round(hi, 1)})
    ae_vs_z = pd.DataFrame(filas_z)
    guardar(ae_vs_z, '04c_ae_vs_z.csv')
    print(ae_vs_z.drop(columns=['con_autoencoder', 'sin_autoencoder']).to_string(index=False))
    _r = ae_vs_z[(ae_vs_z.conjunto == 'val') & (ae_vs_z.grupo == 'todas las anómalas')]
    if len(_r):
        _r = _r.iloc[0]
        if _r.ic_inf > 0:
            print('En validación el autoencoder localiza mejor que el z de las medias: ese es su aporte.')
        elif _r.ic_sup < 0:
            print('OJO: en validación el z de las medias localiza mejor que el autoencoder.')
        else:
            print('En validación no hay diferencia clara entre localizar con o sin autoencoder.')
    print('dif_ventaja: ventaja con autoencoder − ventaja sin autoencoder, en puntos; IC 95 % remuestreando')
    print('bloques de 6 h. Prueba solo se informa.')


# =============================================================================
# 5. TABLA DE CONSULTA PARA EL EXPERTO (sección 4.5 del paper)
# =============================================================================
print('\n' + '=' * 72)
print('5. TABLA DE CONSULTA PARA EL EXPERTO: VARIABLE -> COMPONENTE -> MODO DE FALLA')
print('=' * 72)
def componente_principal(v):
    """Componente de las reglas que miran la variable como variable principal."""
    for papel in ('var', 'a', 'vars', 'var2', 'b'):
        reglas = [r for r in IDS_REGLAS if r in criticidad.COMPONENTE and
                  (v in POR_ID[r].get('vars', []) if papel == 'vars' else POR_ID[r].get(papel) == v)]
        if reglas:
            c = pd.Series([criticidad.COMPONENTE[r] for r in reglas]).value_counts().index[0]
            return TEXTO_COMP.get(c, c)
    return 'sin componente'


filas_t, COMP_VAR, COMP_PRINCIPAL = [], {}, {}
for j, v in enumerate(VARIABLES):
    reglas = [r for r in IDS_REGLAS if r in criticidad.CRITICIDAD and v in variables_de(r)]
    comps = sorted({criticidad.COMPONENTE[r] for r in reglas if r in criticidad.COMPONENTE})
    fallas = list(dict.fromkeys(criticidad.FALLA[r] for r in sorted(reglas, key=criticidad.CRITICIDAD.get)
                                if r in criticidad.FALLA))
    COMP_VAR[v] = ', '.join(comps) if comps else 'sin regla'
    COMP_PRINCIPAL[v] = componente_principal(v)
    desc, unidad = DESCRIPCION.get(v, ('', ''))
    filas_t.append({'variable': v, 'descripcion': desc, 'unidad': unidad, 'reglas': ', '.join(reglas),
                    'componente_principal': COMP_PRINCIPAL[v],
                    'componentes': COMP_VAR[v], 'modos_de_falla': ' | '.join(fallas),
                    'criticidad_mas_alta': min((criticidad.CRITICIDAD[r] for r in reglas), default=np.nan),
                    'umbral_P95_error': ref[j], 'confirmado_por_experto': '', 'comentario_experto': ''})
tabla = pd.DataFrame(filas_t)
guardar(tabla, '04c_tabla_experto.csv')
print(tabla[['variable', 'componente_principal', 'componentes', 'reglas']].to_string(index=False))
print('componente_principal: el de las reglas que miran la variable como variable principal; es el')
print('que lleva la etiqueta de cada ventana. Columnas confirmado_por_experto y comentario_experto vacías:')
print('para que la planta la corrobore.')
m['componentes_1'] = [COMP_VAR.get(v, '') if v else '' for v in m.var_delta_1]

# etiqueta de dos ejes de cada ventana: componente (variable) · nivel
_nivel_txt = pd.Series(nivel).map(TEXTO_NIVEL).fillna('sin nivel').to_numpy()
m['componente'] = [COMP_PRINCIPAL.get(v, '') if v else '' for v in m.var_delta_1]
m['nivel_texto'] = _nivel_txt
m['etiqueta'] = np.where(~marcada, 'no marcada',
                         np.where(m.var_delta_1 != '', m.componente + ' (' + m.var_delta_1 + ') · ' + _nivel_txt,
                                  'sin variable sobre su umbral · ' + _nivel_txt))


# =============================================================================
# 6. CAUSA RAÍZ PROBABLE DE LAS VENTANAS MARCADAS
# =============================================================================
print('\n' + '=' * 72)
print('6. CAUSA RAÍZ PROBABLE: VARIABLE QUE MÁS SE PASA DE SU UMBRAL Y SU COMPONENTE')
print('=' * 72)
_v1 = m.var_delta_1.to_numpy()
filas_c = []
for cj, c in [('val', va), ('test', te)]:
    gr = {'marcadas con alarma': c & marcada & anomala, 'marcadas sin alarma': c & marcada & normal}
    if HAY_NIVEL:
        gr.update({f'marcadas {TEXTO_NIVEL[nv]}': c & marcada & (nivel == nv) for nv in NIVELES})
    for gn, s in gr.items():
        s = s & (_v1 != '')
        if not s.any():
            continue
        for v, k in pd.Series(_v1[s]).value_counts().items():
            filas_c.append({'conjunto': cj, 'grupo': gn, 'variable': v, 'componentes': COMP_VAR.get(v, ''),
                            'ventanas': int(k), 'pct': round(100 * k / s.sum(), 1)})
causa = pd.DataFrame(filas_c, columns=['conjunto', 'grupo', 'variable', 'componentes', 'ventanas', 'pct'])
guardar(causa, '04c_causa_raiz.csv')
for gn in causa.grupo.unique():
    t = causa[(causa.conjunto == 'test') & (causa.grupo == gn)].head(4)
    if len(t):
        print(f'Prueba · {gn}: ' + '; '.join(f'{r.variable} {r.pct:.0f} % ({r.componentes})'
                                             for r in t.itertuples()))
print('"marcadas sin alarma": el modelo las marca y ninguna regla sonó; posible aviso temprano o falsa alarma.')

# etiqueta componente · nivel de las ventanas marcadas
filas_e = []
for cj, c in [('val', va), ('test', te)]:
    s = c & marcada
    if not s.any():
        continue
    t = pd.crosstab(m.componente[s], m.nivel_texto[s])
    for comp, fila in t.iterrows():
        for nv, k in fila.items():
            if k:
                filas_e.append({'conjunto': cj, 'componente': comp or 'sin variable sobre su umbral', 'nivel': nv,
                                'ventanas': int(k), 'pct_de_marcadas': round(100 * k / s.sum(), 1)})
etiquetas = pd.DataFrame(filas_e, columns=['conjunto', 'componente', 'nivel', 'ventanas', 'pct_de_marcadas'])
guardar(etiquetas, '04c_etiquetas_resumen.csv')
_t = etiquetas[etiquetas.conjunto == 'test']
if len(_t):
    _orden = [n for n in ['sin alarma', 'leve', 'media', 'grave', 'sin nivel'] if n in set(_t.nivel)]
    _p = _t.pivot_table(index='componente', columns='nivel', values='pct_de_marcadas', aggfunc='sum',
                        fill_value=0)[_orden]
    print('\nEtiqueta componente · nivel de las ventanas marcadas de prueba (% de las marcadas):')
    print(_p.assign(total=_p.sum(axis=1)).sort_values('total', ascending=False).round(1).to_string())


# =============================================================================
# 7. EJEMPLOS DE LOCALIZACIÓN EXPLICADA (figuras 9 y 10 del paper)
# =============================================================================
print('\n' + '=' * 72)
print('7. EJEMPLOS DE LOCALIZACIÓN EXPLICADA (prueba)')
print('=' * 72)
cand = te & marcada & anomala & (suma > 0)
ejemplos = []
if HAY_NIVEL:
    for nv in NIVELES:
        idx = np.where(cand & (nivel == nv))[0]
        if len(idx):
            ejemplos.append((TEXTO_NIVEL[nv], idx[np.argsort(ERR[idx])[len(idx) // 2]]))
if not ejemplos and cand.any():
    o = np.where(cand)[0][np.argsort(ERR[cand])]
    ejemplos = [('típica', o[len(o) // 2]), ('mayor error', o[-1])]
if ejemplos:
    x = np.arange(len(VARIABLES))
    fig, ejes = plt.subplots(2, len(ejemplos), figsize=(5.2 * len(ejemplos), 6.6), squeeze=False)
    for col, (etq, i) in enumerate(ejemplos):
        sobre = D[i] > 0
        color = np.where(sobre, '#d62728', '#7f7f7f')
        a1, a2 = ejes[0][col], ejes[1][col]
        a1.vlines(x, 0, EV[i], color=color, lw=2)
        a1.scatter(x, EV[i], color=color, s=18, zorder=3)
        a1.scatter(x, ref, marker='_', color='k', s=160, linewidths=1.8, zorder=4, label='umbral P95 de la variable')
        a1.set_title(f'{etq}: error {ERR[i]:.3f} ({ERR[i] / umbral:.1f} veces el umbral)', fontsize=10)
        a1.set_ylabel('Error de la variable (MSE)')
        pct = 100 * Dpos[i] / suma[i]
        a2.bar(x, pct, color=np.where(sobre, '#d62728', '#c7c7c7'))
        a2.set_ylabel('% del exceso sobre los umbrales')
        for a in (a1, a2):
            a.set_xticks(x)
            a.set_xticklabels(VARIABLES, rotation=90, fontsize=8)
            a.grid(alpha=0.3, axis='y')
        print(f'\nEjemplo {etq}: error {ERR[i]:.4f}, {int(sobre.sum())} variables sobre su umbral')
        print('  % del exceso | variable | componentes')
        for j in [j for j in np.argsort(-pct) if pct[j] > 0][:5]:
            print(f'  {pct[j]:>11.1f} | {VARIABLES[j]:<8} | {COMP_VAR[VARIABLES[j]]}')
    ejes[0][0].legend(fontsize=8)
    fig.suptitle(f'{EQUIPO}: localización explicada de ventanas típicas de prueba')
    fig.tight_layout()
    fig.savefig(salida('04c_localizacion_ejemplos.png'), dpi=150)
    plt.close(fig)


# =============================================================================
# 8. MÉTRICAS COMO LA TABLA 2 DEL PAPER (umbral P95)
# =============================================================================
print('\n' + '=' * 72)
print('8. MÉTRICAS CONTRA LAS ETIQUETAS DE LAS REGLAS (umbral P95 del error total)')
print('=' * 72)
filas_m = []
for cj, c in [('val', va), ('test', te)]:
    y, p = anomala[c], marcada[c]
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    filas_m.append({'conjunto': cj, 'ventanas': int(c.sum()), 'anomalas': int(y.sum()), 'marcadas': int(p.sum()),
                    'precision': prec, 'recall': rec, 'f1': 2 * prec * rec / max(prec + rec, 1e-12),
                    'jaccard': tp / max(tp + fp + fn, 1), 'especificidad': tn / max(tn + fp, 1),
                    'auc': roc_auc_score(y, ERR[c]) if 0 < y.sum() < len(y) else np.nan})
met = pd.DataFrame(filas_m).round(3)
guardar(met, '04c_metricas.csv')
print(met.to_string(index=False))

# detección por nivel
if HAY_NIVEL:
    filas_n = []
    for cj, c in [('val', va), ('test', te)]:
        for nv, g in [('sin alarma', c & normal)] + [(TEXTO_NIVEL[n], c & (nivel == n)) for n in NIVELES]:
            filas_n.append({'conjunto': cj, 'nivel': nv, 'ventanas': int(g.sum()),
                            'marcadas': int((g & marcada).sum()),
                            'pct_marcadas': round(100 * (g & marcada).sum() / max(g.sum(), 1), 1)})
    det_nivel = pd.DataFrame(filas_n)
    guardar(det_nivel, '04c_deteccion_nivel.csv')
    print('\n% de ventanas marcadas por nivel (sin alarma: falsas alarmas; leve, media y grave: recall del nivel):')
    print(det_nivel.pivot(index='nivel', columns='conjunto', values='pct_marcadas')
          .reindex(['sin alarma', 'leve', 'media', 'grave'])[['val', 'test']].to_string())
print('El paper obtiene F1 0,56 contra etiquetas automáticas hechas con umbrales de expertos, que es')
print('lo mismo que son nuestras reglas. Aquí el umbral es el P95 de las normales de validación.')


# =============================================================================
# 9. CONSISTENCIA EN EL TIEMPO (ecuaciones 12 y 13 del paper)
# =============================================================================
print('\n' + '=' * 72)
print('9. CONSISTENCIA: ¿LAS MARCAS VIENEN EN BLOQUES CONTINUOS O PARPADEAN?')
print('=' * 72)
SEGS = {}
for cj, c in [('val', va), ('test', te)]:
    s = m.loc[c, LLAVES].copy()
    s['fila'] = np.where(c)[0]
    s = s.sort_values(LLAVES)
    corte = (s.tramo_id.diff() != 0) | (s.ventana_id.diff() != 1)
    SEGS[cj] = [g.fila.to_numpy() for _, g in s.groupby(corte.cumsum().to_numpy())]


def consistencia(flags, segmentos, g):
    cub = largo = n_an = 0
    for idx in segmentos:
        f = flags[idx]
        pos = np.where(f)[0]
        if len(pos) == 0:
            continue
        cortes = np.where(np.diff(pos) > g + 1)[0]
        for s0, s1 in zip(np.r_[pos[0], pos[cortes + 1]], np.r_[pos[cortes], pos[-1]]):
            largo += s1 - s0 + 1
            cub += int(f[s0:s1 + 1].sum())
            n_an += 1
    return (cub / largo if largo else np.nan), n_an, (largo / n_an if n_an else np.nan)


filas_k = []
for cj in ('val', 'test'):
    if not SEGS[cj]:
        continue
    todas = np.concatenate(SEGS[cj])
    for senal, flags in [('autoencoder (error > P95)', marcada), ('reglas (alguna alarma)', anomala)]:
        for g in GAPS:
            k, n_an, dur = consistencia(flags, SEGS[cj], g)
            filas_k.append({'conjunto': cj, 'senal': senal, 'tolerancia_ventanas': g, 'tolerancia_min': 10 * g,
                            'anomalias': n_an, 'duracion_media_ventanas': round(dur, 1), 'consistencia': round(k, 3),
                            'pct_ventanas_marcadas': round(100 * flags[todas].mean(), 1)})
cons = pd.DataFrame(filas_k)
guardar(cons, '04c_consistencia.csv')
print(cons.to_string(index=False))
print('consistencia = parte del tramo de cada anomalía que queda marcada; 1 = bloque continuo. Las')
print('marcas separadas por hasta tolerancia_ventanas sin marca son la misma anomalía. El paper')
print('obtiene 0,92 para su autoencoder y 0,62 para las etiquetas automáticas.')


# =============================================================================
# 10. EL ERROR DE CADA VARIABLE EN EL TIEMPO (figura 7 del paper)
# =============================================================================
print('\n' + '=' * 72)
print('10. ¿EL ERROR DE CADA VARIABLE SUBE CUANDO HAY ANOMALÍA? (en el tiempo, figura 7 del paper)')
print('=' * 72)
ANTES, DESPUES, QUIETAS = 36, 18, 12             # 6 h antes, 3 h después, 2 h sin el evento antes
_dv = delta[(delta.conjunto == 'val') & (delta.grupo == 'todas las anómalas')].sort_values('delta', ascending=False)
TOP_T = list(_dv.variable.head(4)) if len(_dv) else VARIABLES[:4]
RATIO = EV / np.maximum(ref, 1e-12)                             # error de cada variable, en veces su umbral
RATIO_TOT = ERR / umbral
EVENTO = 'grave' if HAY_NIVEL else 'alarma'
evento = (nivel == '3_grave') if HAY_NIVEL else anomala


def inicios(cj):
    """Inicios del evento sin el evento en las QUIETAS ventanas previas del mismo tramo continuo."""
    out = []
    for idx in SEGS.get(cj, []):
        e = evento[idx]
        out += [(idx, p) for p in range(QUIETAS, len(idx)) if e[p] and not e[p - QUIETAS:p].any()]
    return out


SERIES = [('error total', RATIO_TOT)] + [(v, RATIO[:, VARIABLES.index(v)]) for v in TOP_T]
filas_t = []
for cj in ('val', 'test'):
    ini = inicios(cj)
    for nom, r in SERIES:
        for k in range(-ANTES, DESPUES + 1):
            vals = [r[idx[p + k]] for idx, p in ini if 0 <= p + k < len(idx)]
            if vals:
                filas_t.append({'conjunto': cj, 'serie': nom, 'minutos': 10 * k, 'inicios': len(vals),
                                'mediana_veces_umbral': float(np.median(vals)),
                                'p25': float(np.percentile(vals, 25)), 'p75': float(np.percentile(vals, 75))})
tiempo = pd.DataFrame(filas_t)
guardar(tiempo.round(4), '04c_error_alrededor_inicios.csv')
MARCAS = [-360, -180, -60, -30, -10, 0, 30, 60, 180]
for cj in ('val', 'test'):
    t = tiempo[tiempo.conjunto == cj]
    if t.empty:
        continue
    n_ini = int(t[(t.serie == 'error total') & (t.minutos == 0)].inicios.max())
    print(f'\n{"Validación" if cj == "val" else "Prueba (solo se informa)"}: {n_ini} inicios de {EVENTO} '
          f'(sin {EVENTO} en las 2 h previas). Mediana del error, en veces su umbral P95:')
    p = t.pivot(index='serie', columns='minutos', values='mediana_veces_umbral').reindex([s for s, _ in SERIES])
    p = p[[c for c in MARCAS if c in p.columns]]
    p.columns = [('inicio' if c == 0 else f'{c:+d} min') for c in p.columns]
    print(p.round(2).to_string())
print('1 = el umbral P95 de esa variable en las ventanas normales de validación. Si la mediana pasa de 1')
print('recién en el inicio, el error acompaña la anomalía; si pasa antes, la anticipa.')

# figura: un episodio de validación, como la figura 7 del paper
_ini = [(idx, p) for idx, p in inicios('val') if p >= ANTES and len(idx) - p > DESPUES]
if _ini:
    idx, p = max(_ini, key=lambda ip: (~anomala[ip[0][ip[1] - ANTES:ip[1]]]).sum())
    sel = idx[p - ANTES:p + DESPUES + 1]
    horas = (np.arange(len(sel)) - ANTES) / 6
    colores = ['#52514e', '#2a78d6', '#eb6834', '#1baf7a', '#eda100']
    fig, ejes = plt.subplots(len(SERIES), 1, figsize=(11, 2.1 * len(SERIES) + 0.8), sharex=True)
    for ax, (nom, r), col in zip(ejes, SERIES, colores):
        for k, s in enumerate(sel):
            if anomala[s]:
                ax.axvspan(horas[k] - 1 / 12, horas[k] + 1 / 12, color='#e7e6e2', lw=0,
                           label='ventana con alarma' if k == int(np.argmax(anomala[sel])) else None)
            if evento[s]:
                ax.axvspan(horas[k] - 1 / 12, horas[k] + 1 / 12, color='#d03b3b', alpha=0.25, lw=0,
                           label=f'ventana {EVENTO}' if k == int(np.argmax(evento[sel])) else None)
        ax.plot(horas, np.maximum(r[sel], 1e-3), color=col, lw=2, marker='o', ms=3)
        ax.axhline(1, color='#0b0b0b', lw=1, ls='--')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel('veces su\numbral', fontsize=8)
        ax.set_title(nom if nom == 'error total' else f'{nom} ({DESCRIPCION[nom][0].lower()})', loc='left', fontsize=10)
        ax.grid(alpha=0.25)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
    ejes[0].legend(fontsize=8, frameon=False, loc='upper left')
    ejes[-1].set_xlabel(f'Horas desde el inicio de la ventana {EVENTO}')
    fecha = pd.Timestamp(m.timestamp_inicio.iloc[idx[p]]).strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f'{EQUIPO}: error de reconstrucción en el tiempo, inicio de {EVENTO} del {fecha} (validación). '
                 'Línea punteada: umbral P95', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(salida('04c_error_en_el_tiempo.png'), dpi=150)
    plt.close(fig)
    print(f'\nEjemplo graficado: inicio de {EVENTO} del {fecha} (validación).')

# figura: la mediana alrededor de todos los inicios
if len(tiempo):
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, cj in zip(ejes, ('val', 'test')):
        t = tiempo[tiempo.conjunto == cj]
        for (nom, _), col in zip(SERIES, ['#52514e', '#2a78d6', '#eb6834', '#1baf7a', '#eda100']):
            s = t[t.serie == nom]
            ax.plot(s.minutos / 60, s.mediana_veces_umbral, color=col, lw=2.4 if nom == 'error total' else 2,
                    ls='--' if nom == 'error total' else '-', label=nom)
        ax.axhline(1, color='#0b0b0b', lw=1, ls=':')
        ax.axvspan(0, DESPUES / 6, color='#d03b3b', alpha=0.08, lw=0)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f'Horas desde el inicio de {EVENTO}')
        n_ini = int(t[(t.serie == 'error total') & (t.minutos == 0)].inicios.max()) if len(t) else 0
        ax.set_title(f'{"Validación" if cj == "val" else "Prueba"}: mediana de {n_ini} inicios', loc='left', fontsize=10)
        ax.grid(alpha=0.25)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
    ejes[0].set_ylabel('Error, veces su umbral P95')
    ejes[0].legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle(f'{EQUIPO}: el error de cada variable alrededor de los inicios de {EVENTO}', x=0.01, ha='left',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(salida('04c_error_promedio_inicios.png'), dpi=150)
    plt.close(fig)


# =============================================================================
# 11. GUARDAR
# =============================================================================
m['nivel'] = nivel
m['familia_dominante'] = familia
m['anomala'] = anomala.astype(int)
m['marcada'] = marcada.astype(int)
m['punto_parada'] = PUNTO_PARADA
for j, v in enumerate(VARIABLES):
    m[f'delta_{v}'] = D[:, j]
cols = (LLAVES + ['timestamp_inicio', 'conjunto', 'nivel', 'familia_dominante', 'anomala', 'error_reconstruccion',
                  'marcada', 'n_sobre_umbral']
        + [f'{p}_{k}' for k in range(1, M_SIG + 1) for p in ('var_delta', 'delta', 'pct_exceso')]
        + ['componentes_1', 'componente', 'nivel_texto', 'etiqueta', 'punto_parada']
        + [f'delta_{v}' for v in VARIABLES])
guardar(m[cols], '04c_localizacion_ventanas.csv')
print(f'\nGuardado en {RUTA_EQ}: 04c_delta_variables.csv/png, 04c_atribucion.csv, 04c_ae_vs_z.csv, 04c_tabla_experto.csv, '
      f'04c_causa_raiz.csv, 04c_etiquetas_resumen.csv, 04c_metricas.csv, 04c_deteccion_nivel.csv, '
      f'04c_consistencia.csv, '
      f'04c_localizacion_ejemplos.png, 04c_error_alrededor_inicios.csv, '
      f'04c_error_en_el_tiempo.png, 04c_error_promedio_inicios.png, '
      f'04c_localizacion_ventanas.csv')
