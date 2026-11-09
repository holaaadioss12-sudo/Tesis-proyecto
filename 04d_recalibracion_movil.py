# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04d — RECALIBRACIÓN MÓVIL: ¿LO QUE LE PASA A CR009 ES DERIVA?
=============================================================================
El puntaje de cada ventana se divide por el nivel normal reciente: la mediana
del puntaje de las N ventanas sin alarma anteriores a ella (solo mira hacia
atrás, no usa nada de la ventana ni del futuro). Se compara con el puntaje fijo
del paso 04, para el autoencoder y para la línea base z_medias del paso 06.
Entrada:  {EQUIPO}_error_reconstruccion_W60.csv (paso 04), {EQUIPO}_ventanas_W60.csv y
          {EQUIPO}_features_W60.csv (paso 02)
Salida:   04d_recalibracion.csv, 04d_comparaciones.csv, 04d_mensual.csv, 04d_variables_mensual.csv,
          04d_puntaje_ventanas.csv, 04d_deriva.png
N y la variante se eligen en validación; prueba solo se informa.
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
from sklearn.metrics import roc_auc_score


# =============================================================================
# 0. CONSTANTES
# =============================================================================
W = 60
SEED = 42
PERCENTIL = 95
N_MOVIL = (50, 100, 200, 500)    # ventanas de historia para el nivel normal reciente
MINIMO = 30                      # historia mínima para calcularlo
N_BOOT = 1000
MIN_MES = 20                     # ventanas mínimas por clase para el AUC de un mes
LLAVES = ['tramo_id', 'ventana_id']
VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
UNIDAD = {'CM': '%', 'PI': '%', 'PDF': 'kPa', 'PEL': 'kPa', **{v: '°C' for v in ['T7', 'T8', 'T9', 'T1', 'T2', 'T5']},
          **{v: 'mm/s' for v in ['V1', 'V2', 'V3', 'V4']}}

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


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


def guardar(df, nombre):
    df.to_csv(salida(nombre), sep=';', decimal=',', index=False, encoding='utf-8-sig')


# =============================================================================
# 1. DATOS EN ORDEN DE TIEMPO
# =============================================================================
print('=' * 72)
print(f'PASO 04d — RECALIBRACIÓN MÓVIL ({EQUIPO})')
print('=' * 72)
err = pd.read_csv(ARCHIVO_ERROR)
if AE not in err.columns or not np.allclose(err[AE], err.error_reconstruccion):
    raise SystemExit(f'El paso 04 se corrió con otro punto de parada: córrelo con PUNTO_PARADA={PUNTO_PARADA!r}.')
d = err[LLAVES + ['conjunto', 'error_reconstruccion', 'n_reglas_activas']].merge(
    pd.read_csv(ARCHIVO_VENTANAS, usecols=LLAVES + ['timestamp_inicio'], parse_dates=['timestamp_inicio'])
    .drop_duplicates(LLAVES), on=LLAVES, how='left', validate='one_to_one')
if d.timestamp_inicio.isna().any():
    raise SystemExit('Hay ventanas del paso 04 sin fecha en el paso 02: ¿son de la misma corrida?')
DETECTORES = {'autoencoder': 'error_reconstruccion'}
if os.path.exists(ARCHIVO_FEATURES):
    medias = [f'{v}_media' for v in VARIABLES]
    _f = pd.read_csv(ARCHIVO_FEATURES, usecols=lambda c: c in LLAVES + medias).drop_duplicates(LLAVES)
    medias = [c for c in medias if c in _f.columns]
    d = d.merge(_f, on=LLAVES, how='left', validate='one_to_one')
    _ref = d[(d.conjunto == 'val') & (d.n_reglas_activas == 0)]
    _mu, _sd = _ref[medias].mean(), _ref[medias].std().replace(0, 1.0)
    d['z_medias'] = ((d[medias] - _mu) / _sd).abs().max(axis=1)       # la línea base del paso 06
    if d.z_medias.notna().all():
        DETECTORES['z_medias'] = 'z_medias'
d = d.sort_values(['timestamp_inicio'] + LLAVES).reset_index(drop=True)
quieta = (d.n_reglas_activas == 0).to_numpy()
y = ~quieta
va, te = (d.conjunto == 'val').to_numpy(), (d.conjunto == 'test').to_numpy()
print(f'Ventanas: validación {va.sum():,} | prueba {te.sum():,} | sin alarma: {quieta.sum():,}')
print(f'Detectores: {", ".join(DETECTORES)}')


def base_quietas(s, n):
    """Mediana del puntaje de las n ventanas sin alarma anteriores (sin incluir la actual)."""
    k = np.cumsum(quieta) - quieta
    r = pd.Series(s[quieta]).rolling(n, min_periods=MINIMO).median().to_numpy()
    out = np.full(len(s), np.nan)
    ok = k >= MINIMO
    out[ok] = r[k[ok] - 1]
    return out


def base_todas(s, n):
    """Percentil 25 del puntaje de las n ventanas anteriores, sin usar etiquetas."""
    return pd.Series(s).rolling(n, min_periods=MINIMO).quantile(0.25).shift(1).to_numpy()


comun = (np.cumsum(quieta) - quieta >= MINIMO) & (np.arange(len(d)) >= MINIMO)
mv, mt = va & comun, te & comun
print(f'Ventanas de validación sin historia suficiente (se excluyen de la comparación): {int((va & ~comun).sum()):,}')


# =============================================================================
# 2. PUNTAJE FIJO CONTRA PUNTAJE MÓVIL
# =============================================================================
print('\n' + '=' * 72)
print('2. PUNTAJE FIJO (paso 04) CONTRA PUNTAJE MÓVIL (dividido por el nivel normal reciente)')
print('=' * 72)


def metricas(s):
    u = float(np.percentile(s[mv & quieta], PERCENTIL))
    p = s > u
    tp, fp, fn = int((p & y & mt).sum()), int((p & ~y & mt).sum()), int((~p & y & mt).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {'auc_val': roc_auc_score(y[mv], s[mv]), 'auc_test': roc_auc_score(y[mt], s[mt]), 'umbral': u,
            'pct_quietas_marcadas_val': 100 * p[mv & quieta].mean(),
            'pct_quietas_marcadas_test': 100 * p[mt & quieta].mean(),
            'precision_test': prec, 'recall_test': rec, 'f1_test': 2 * prec * rec / max(prec + rec, 1e-12)}


filas, PUNTAJES, ELEGIDO = [], {}, {}
for det, col in DETECTORES.items():
    s = d[col].to_numpy(dtype=float)
    PUNTAJES[(det, 'fijo', 0)] = s
    filas.append({'detector': det, 'metodo': 'fijo', 'N': 0, **metricas(s)})
    for n in N_MOVIL:
        for metodo, base in [('móvil: quietas', base_quietas(s, n)), ('móvil: sin etiquetas', base_todas(s, n))]:
            sm = s / np.maximum(base, 1e-12)
            PUNTAJES[(det, metodo, n)] = sm
            filas.append({'detector': det, 'metodo': metodo, 'N': n, **metricas(sm)})
tab = pd.DataFrame(filas)
for det in DETECTORES:
    t = tab[(tab.detector == det) & (tab.metodo != 'fijo')]
    i = t.auc_val.idxmax()
    ELEGIDO[det] = (tab.loc[i, 'metodo'], int(tab.loc[i, 'N']))
tab['elegido'] = [(r.metodo, r.N) == ELEGIDO.get(r.detector) for r in tab.itertuples()]
guardar(tab.round(4), '04d_recalibracion.csv')
pd.set_option('display.width', 240)
print(tab.round(3).to_string(index=False))
print(f'\nN: ventanas de historia (una ventana = 10 min). quietas: mediana de las N últimas ventanas sin')
print('alarma (usa las etiquetas pasadas, que en operación se conocen en el momento); sin etiquetas:')
print('percentil 25 de las N últimas ventanas. Se elige la variante y N con el AUC de validación.')
print('pct_quietas_marcadas: con el umbral P95 de validación debería quedar cerca de 5 %; mucho más en')
print('prueba es deriva.')


# =============================================================================
# 3. ¿MEJORA? DIFERENCIAS DE AUC EN PRUEBA CON INTERVALO POR TRAMOS (como el 06)
# =============================================================================
print('\n' + '=' * 72)
print('3. ¿MEJORA EN PRUEBA? (intervalo de confianza 95 % remuestreando tramos)')
print('=' * 72)
rng = np.random.default_rng(SEED)
idx_t = np.where(mt)[0]
_tr = d.tramo_id.to_numpy()[idx_t]
grupos = pd.Series(np.arange(len(idx_t))).groupby(_tr).indices
claves = np.array(list(grupos))


def ic_diferencia(sa, sb):
    a, b, yy = sa[idx_t], sb[idx_t], y[idx_t]
    dif = roc_auc_score(yy, a) - roc_auc_score(yy, b)
    vals = []
    for _ in range(N_BOOT):
        sel = np.concatenate([grupos[c] for c in rng.choice(claves, len(claves))])
        if 0 < yy[sel].sum() < len(sel):
            vals.append(roc_auc_score(yy[sel], a[sel]) - roc_auc_score(yy[sel], b[sel]))
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return dif, lo, hi


comparaciones = [(f'{det} móvil − {det} fijo', PUNTAJES[(det, *ELEGIDO[det])], PUNTAJES[(det, 'fijo', 0)])
                 for det in DETECTORES]
if 'z_medias' in DETECTORES:
    comparaciones += [('autoencoder fijo − z_medias fijo', PUNTAJES[('autoencoder', 'fijo', 0)],
                       PUNTAJES[('z_medias', 'fijo', 0)]),
                      ('autoencoder móvil − z_medias móvil', PUNTAJES[('autoencoder', *ELEGIDO['autoencoder'])],
                       PUNTAJES[('z_medias', *ELEGIDO['z_medias'])])]
filas_c = []
for nombre, sa, sb in comparaciones:
    dif, lo, hi = ic_diferencia(sa, sb)
    veredicto = ('mejora' if lo > 0 else 'empeora' if hi < 0 else 'sin diferencia demostrable')
    filas_c.append({'comparacion': nombre, 'dif_auc_test': dif, 'ic_inf': lo, 'ic_sup': hi, 'lectura': veredicto})
    print(f'  {nombre:<40} {dif:+.3f}  IC 95 % [{lo:+.3f}, {hi:+.3f}]  -> {veredicto}')
guardar(pd.DataFrame(filas_c).round(4), '04d_comparaciones.csv')
for det, (metodo, n) in ELEGIDO.items():
    print(f'Elegido en validación para {det}: {metodo}, N = {n} ({n * 10 / 60:.0f} h de historia)')


# =============================================================================
# 4. LA DERIVA MES A MES
# =============================================================================
print('\n' + '=' * 72)
print('4. MES A MES: NIVEL DE LAS VENTANAS SIN ALARMA Y VENTANAS SIN ALARMA MARCADAS')
print('=' * 72)
s_fijo = PUNTAJES[('autoencoder', 'fijo', 0)]
s_mov = PUNTAJES[('autoencoder', *ELEGIDO['autoencoder'])]
u_fijo = tab[(tab.detector == 'autoencoder') & (tab.metodo == 'fijo')].umbral.iloc[0]
u_mov = tab[(tab.detector == 'autoencoder') & tab.elegido].umbral.iloc[0]
mes = d.timestamp_inicio.dt.to_period('M').astype(str).to_numpy()
filas_m = []
for me in sorted(set(mes)):
    s = (mes == me) & comun
    q = s & quieta
    fila = {'mes': me, 'conjunto': '/'.join(sorted(set(d.conjunto[s]))), 'ventanas': int(s.sum()),
            'sin_alarma': int(q.sum()),
            'error_mediano_sin_alarma': float(np.median(s_fijo[q])) if q.any() else np.nan,
            'pct_sin_alarma_marcadas_fijo': 100 * float((s_fijo[q] > u_fijo).mean()) if q.any() else np.nan,
            'pct_sin_alarma_marcadas_movil': 100 * float((s_mov[q] > u_mov).mean()) if q.any() else np.nan}
    ok = q.sum() >= MIN_MES and (s & y).sum() >= MIN_MES
    fila['auc_fijo'] = roc_auc_score(y[s], s_fijo[s]) if ok else np.nan
    fila['auc_movil'] = roc_auc_score(y[s], s_mov[s]) if ok else np.nan
    filas_m.append(fila)
mensual = pd.DataFrame(filas_m)
guardar(mensual.round(4), '04d_mensual.csv')
print(mensual.round(3).to_string(index=False))
print(f'Con el umbral fijo, una operación estable deja ~5 % de las ventanas sin alarma marcadas cada mes.')

# cambio de cada variable en las ventanas sin alarma, mes a mes
if 'z_medias' in DETECTORES:
    _ref = (d.conjunto == 'val').to_numpy() & quieta
    _mu, _sd = d.loc[_ref, medias].median(), d.loc[_ref, medias].std().replace(0, 1.0)
    filas_v = []
    for me in sorted(set(mes)):
        q = (mes == me) & quieta
        if q.sum() < MIN_MES:
            continue
        med = d.loc[q, medias].median()
        for c in medias:
            filas_v.append({'mes': me, 'variable': c[:-6], 'unidad': UNIDAD.get(c[:-6], ''),
                            'ventanas_sin_alarma': int(q.sum()), 'mediana': med[c],
                            'mediana_validacion': _mu[c], 'cambio_desv': (med[c] - _mu[c]) / _sd[c]})
    var_mes = pd.DataFrame(filas_v)
    guardar(var_mes.round(4), '04d_variables_mensual.csv')
    if len(var_mes):
        print('\nCambio de cada variable en las ventanas sin alarma (desviaciones respecto de la mediana de '
              'validación):')
        _piv = var_mes.pivot(index='mes', columns='variable', values='cambio_desv')[[c[:-6] for c in medias]]
        print(_piv.round(1).to_string())
        print('Lo que más se movió cada mes (|cambio| de 1 desviación o más):')
        for me, r in _piv.iterrows():
            top = r[r.abs() >= 1].abs().sort_values(ascending=False).head(3).index
            if len(top):
                t = var_mes[var_mes.mes == me].set_index('variable')
                print(f'  {me}: ' + '; '.join(
                    f'{v} {t.loc[v, "cambio_desv"]:+.1f} desv. (mediana {t.loc[v, "mediana"]:.1f} '
                    f'{t.loc[v, "unidad"]}, validación {t.loc[v, "mediana_validacion"]:.1f})' for v in top))


# =============================================================================
# 5. FIGURA Y PUNTAJE POR VENTANA
# =============================================================================
base_ae = s_fijo / np.where(s_mov > 0, s_mov, np.nan)
t0 = d.timestamp_inicio[te].min() if te.any() else None
fig, ejes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
q = quieta & comun
ejes[0].scatter(d.timestamp_inicio[q], s_fijo[q], s=4, alpha=0.35, color='#6baed6', label='ventanas sin alarma')
ejes[0].plot(d.timestamp_inicio[comun], base_ae[comun], color='#08519c', lw=1.2,
             label=f'nivel normal reciente ({ELEGIDO["autoencoder"][0]}, N = {ELEGIDO["autoencoder"][1]})')
ejes[0].axhline(u_fijo, color='#d62728', lw=1.2, ls='--', label=f'umbral fijo P{PERCENTIL}')
ejes[0].set_yscale('log')
ejes[0].set_ylabel('Error de reconstrucción')
ejes[0].set_title(f'{EQUIPO}: error de las ventanas sin alarma en el tiempo')
ejes[1].scatter(d.timestamp_inicio[q], s_mov[q], s=4, alpha=0.35, color='#74c476')
ejes[1].axhline(u_mov, color='#d62728', lw=1.2, ls='--', label=f'umbral P{PERCENTIL} del puntaje móvil')
ejes[1].set_yscale('log')
ejes[1].set_ylabel('Error / nivel normal reciente')
ejes[1].set_title('Puntaje móvil de las mismas ventanas')
for a in ejes:
    if t0 is not None:
        a.axvline(t0, color='k', lw=1, ls=':')
    a.legend(fontsize=8, loc='upper left')
    a.grid(alpha=0.3)
if t0 is not None:
    ejes[0].text(t0, ejes[0].get_ylim()[1], ' prueba', va='top', fontsize=9)
fig.tight_layout()
fig.savefig(salida('04d_deriva.png'), dpi=150)
plt.close(fig)

d['alarma'] = y.astype(int)
d['nivel_normal_reciente'] = base_ae
d['puntaje_movil'] = s_mov
d['marcada_fija'] = (s_fijo > u_fijo).astype(int)
d['marcada_movil'] = (s_mov > u_mov).astype(int)
d['punto_parada'] = PUNTO_PARADA
guardar(d[LLAVES + ['timestamp_inicio', 'conjunto', 'alarma', 'error_reconstruccion', 'nivel_normal_reciente',
                    'puntaje_movil', 'marcada_fija', 'marcada_movil', 'punto_parada']], '04d_puntaje_ventanas.csv')

fila_f = tab[(tab.detector == 'autoencoder') & (tab.metodo == 'fijo')].iloc[0]
fila_m = tab[(tab.detector == 'autoencoder') & tab.elegido].iloc[0]
print('\n' + '=' * 72)
print('RESUMEN (autoencoder, prueba)')
print('=' * 72)
print(f'AUC: fijo {fila_f.auc_test:.3f} -> móvil {fila_m.auc_test:.3f}')
print(f'Ventanas sin alarma marcadas: fijo {fila_f.pct_quietas_marcadas_test:.1f} % -> móvil '
      f'{fila_m.pct_quietas_marcadas_test:.1f} % (lo esperado sin deriva: ~5 %)')
print('Si el AUC sube y las marcadas sin alarma vuelven cerca de 5 %, el problema era deriva: la')
print('propuesta es recalibrar el umbral con el nivel normal reciente. Si no cambia, la diferencia')
print('del equipo no es deriva y tiene sentido buscar hiperparámetros.')
print(f'\nGuardado en {RUTA_EQ}: 04d_recalibracion.csv, 04d_comparaciones.csv, 04d_mensual.csv, '
      f'04d_variables_mensual.csv, 04d_puntaje_ventanas.csv, 04d_deriva.png')
