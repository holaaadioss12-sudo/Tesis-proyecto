# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 10d — ¿LA GRAVEDAD FORMA GRUPOS DENTRO DE UN MISMO TIPO DE ALARMA?
=============================================================================
El paso 10b mostró que el latente del autoencoder no se agrupa por gravedad
(ARI en prueba <= 0,09) y que los grupos que sí aparecen se parecen más a la
regla dominante que a la banda (NMI_tipo > NMI_banda). Queda una lectura
posible: que el tipo de alarma tape a la gravedad. Si una ventana de
lubricación y una mecánica están lejos en el latente, el agrupamiento corta
primero por tipo y la gravedad, que es una diferencia más chica, no alcanza a
formar grupos propios.

La prueba es fijar el tipo y mirar sólo dentro de él:
  1. Toma el latente que guardó el paso 10b (no corre el modelo de nuevo) y
     las mismas ventanas marcadas: error > P95 de validación sin alarma.
  2. Asigna a cada ventana el tipo de su regla dominante (etiqueta TF·IDF)
     según la columna 'Tipo' de la planilla de criticidad (Hoja1):
     operacional, lubricación o mecánica.
  3. Dentro de cada tipo, y también dentro de cada regla dominante frecuente
     (el control más estricto), con las bandas que tengan al menos MIN_CLASE
     ventanas en validación:
        - normaliza y reduce (PCA) con las ventanas de validación de ese
          grupo, para que la varianza entre tipos ya no mande;
        - silueta de las bandas REALES: ¿las bandas están separadas?
          (escala de Kaufman y Rousseeuw: <= 0,25 sin estructura);
        - K-means con tantos grupos como bandas, K-means con el k de mejor
          silueta en validación (2..6) y GMM: ARI y NMI con las bandas;
        - K-means sembrado con los centros reales (diagnóstico);
        - sonda supervisada (regresión logística) contra el azar
          estratificado: ¿la gravedad está aunque no forme grupos?
     Todo se ajusta en validación; la prueba sólo se informa.
  4. Una figura por tipo: el latente de prueba coloreado por banda.

Entradas: 10b_latente.npz                                  (paso 10b)
          {EQUIPO}_error_reconstruccion_W60.csv            (paso 05)
          10_nivel_ventanas.csv                            (paso 10)
Salidas:  10d_por_tipo.csv, 10d_por_tipo.png
Corre en segundos.
=============================================================================
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
    RUTA_EQ = f'{RUTA_BASE}/CR010'     # CR010 movido a su subcarpeta
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, f1_score)


# =============================================================================
# 0. CONSTANTES
# =============================================================================
SEED = 42
W = 60
LLAVES = ['tramo_id', 'ventana_id']
AE = 'err_AE_normales'          # checkpoint de mínima pérdida (PARCHE P8)
PERCENTIL_UMBRAL = 95
BANDA = 'banda_N_empinada_natural'   # PARCHE P14: la escala principal del paso 10
NIVELES = ['1_leve', '2_media', '3_grave']
K_RANGO = range(2, 7)
PCA_VARIANZA, PCA_MAX = 0.80, 10
MAX_SILUETA = 5000
MIN_CLASE = 10                  # ventanas mínimas por banda en validación
MAX_REGLAS = 6                  # reglas dominantes que se prueban por separado

# Columna 'Tipo' de la planilla de criticidad (criticidadCH8.xlsx, Hoja1).
# A6 aparece también en la combinación operacional 'A + A6'; sola es mecánica
# (eje principal). A23 (setting) no se puede evaluar con las variables que hay.
TIPO_REGLA = {
    **dict.fromkeys(['A', 'A1', 'A4', 'B', 'A3', 'A23'], 'operacional'),
    **dict.fromkeys(['C', 'D', 'E', 'A5', 'A8', 'A9', 'A10', 'K', 'A14',
                     'A15', 'A16', 'L', 'A21', 'A22'], 'lubricacion'),
    **dict.fromkeys(['F', 'A6', 'A7', 'I', 'A11', 'A12', 'A13', 'O', 'A17',
                     'A18', 'A19', 'A20'], 'mecanica'),
}
TIPOS = ['operacional', 'lubricacion', 'mecanica']

def buscar(nombre):
    """La copia más reciente del archivo en la carpeta del equipo o en sus
    subcarpetas (las salidas se van moviendo a carpetas Cambios_...). Nunca
    entra a la carpeta de otro equipo."""
    hallados = []
    for raiz, carpetas, archivos in os.walk(RUTA_EQ):
        carpetas[:] = [c for c in carpetas if not c.upper().startswith('CR0')]
        if nombre in archivos:
            hallados.append(os.path.join(raiz, nombre))
    if not hallados:
        raise SystemExit(f'No encontré {nombre} en {RUTA_EQ} ni en sus subcarpetas. '
                         '¿Corriste los pasos 05, 10 y 10b?')
    ruta = max(hallados, key=os.path.getmtime)
    print(f'  {nombre}: {ruta}')
    return ruta


print('Archivos que se usan (revisa que sean de la misma corrida que el 10b):')
ARCHIVO_LATENTE = buscar('10b_latente.npz')
ARCHIVO_ERROR = buscar(f'{EQUIPO}_error_reconstruccion_W{W}.csv')
ARCHIVO_NIVEL = buscar('10_nivel_ventanas.csv')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


np.random.seed(SEED)


# =============================================================================
# 1. LATENTE, BANDAS Y VENTANAS MARCADAS (igual que el paso 10b)
# =============================================================================
print('=' * 72)
print('PASO 10d — LA GRAVEDAD DENTRO DE CADA TIPO DE ALARMA')
print('=' * 72)

lat = np.load(ARCHIVO_LATENTE, allow_pickle=True)
Z = lat['z'].astype(np.float64)
d = pd.DataFrame({'tramo_id': lat['tramo_id'], 'ventana_id': lat['ventana_id'],
                  'conjunto': lat['conjunto'].astype(str)})
err = pd.read_csv(ARCHIVO_ERROR)
if 'etiqueta_tfidf' not in err.columns:
    raise SystemExit(f'{ARCHIVO_ERROR} no tiene etiqueta_tfidf: no hay regla dominante.')
_cols = [c for c in LLAVES + ['conjunto', AE, 'es_normal', 'etiqueta_tfidf']
         if c in err.columns]
d = d.merge(err[_cols], on=LLAVES + ['conjunto'], how='left', validate='one_to_one')
nivel = pd.read_csv(ARCHIVO_NIVEL)
d = d.merge(nivel[LLAVES + ['N_lineal', BANDA]], on=LLAVES, how='left',
            validate='one_to_one')
if d[AE].isna().any() or d[BANDA].isna().any():
    raise SystemExit('El latente del 10b no calza con los pasos 05 y 10: '
                     'vuelve a correr el 10b.')

d['sin_alarma'] = d.es_normal if 'es_normal' in d.columns else d.N_lineal == 0
u = float(np.percentile(d.loc[(d.conjunto == 'val') & d.sin_alarma, AE],
                        PERCENTIL_UMBRAL))
d['marcada'] = d[AE] > u
d['tipo'] = d.etiqueta_tfidf.map(TIPO_REGLA)
print(f'Umbral del autoencoder (P{PERCENTIL_UMBRAL} sin alarma, validación): {u:.5f}')

base_va = ((d.conjunto == 'val') & d.marcada & (d[BANDA] != '0_normal')).to_numpy()
base_te = ((d.conjunto == 'test') & d.marcada & (d[BANDA] != '0_normal')).to_numpy()
# Mismo formato que el paso 10b: los números tienen que ser los mismos.
print(f'Marcadas: validación {(d.marcada & (d.conjunto == "val")).sum():,} '
      f'(N > 0: {base_va.sum():,}) | prueba {(d.marcada & (d.conjunto == "test")).sum():,} '
      f'(N > 0: {base_te.sum():,})')
_sin_tipo = (base_va | base_te) & d.tipo.isna().to_numpy()
if _sin_tipo.any():
    print(f'Ventanas marcadas con N > 0 sin tipo: {_sin_tipo.sum():,} '
          f'(reglas {sorted(d.etiqueta_tfidf[_sin_tipo].unique())}). Quedan fuera.')

print('\nVentanas marcadas con N > 0, por tipo y banda:')
_t = d[base_va | base_te]
print(pd.crosstab([_t.conjunto, _t.tipo], _t[BANDA]).to_string())


# =============================================================================
# 2. LA PRUEBA, GRUPO POR GRUPO
# =============================================================================
def silueta(Xs, lab):
    if len(set(lab)) < 2:
        return np.nan
    idx = np.arange(len(lab))
    if len(idx) > MAX_SILUETA:
        idx = np.random.RandomState(SEED).choice(idx, MAX_SILUETA, replace=False)
    return float(silhouette_score(Xs[idx], lab[idx]))


def estructura(s):
    """Escala de Kaufman y Rousseeuw (1990) para la silueta media."""
    if np.isnan(s):
        return '-'
    return ('sin estructura' if s <= 0.25 else 'debil' if s <= 0.50
            else 'razonable' if s <= 0.70 else 'fuerte')


filas, mapas = [], {}


def probar(nivel_grupo, grupo, sel_va, sel_te):
    fila = {'nivel': nivel_grupo, 'grupo': grupo,
            'n_val': int(sel_va.sum()), 'n_test': int(sel_te.sum())}
    b_va_all = d[BANDA].to_numpy()[sel_va]
    n_banda = pd.Series(b_va_all).value_counts().reindex(NIVELES).fillna(0).astype(int)
    for nv in NIVELES:
        fila[f'val_{nv}'] = int(n_banda[nv])
        fila[f'test_{nv}'] = int((d[BANDA].to_numpy()[sel_te] == nv).sum())
    bandas = [nv for nv in NIVELES if n_banda[nv] >= MIN_CLASE]
    fila['bandas'] = '+'.join(b[2:] for b in bandas)
    sv = sel_va & d[BANDA].isin(bandas).to_numpy()
    st = sel_te & d[BANDA].isin(bandas).to_numpy()
    b_va, b_te = d[BANDA].to_numpy()[sv], d[BANDA].to_numpy()[st]
    if len(bandas) < 2 or len(set(b_te)) < 2:
        fila['nota'] = (f'menos de 2 bandas con {MIN_CLASE} ventanas en validación'
                        if len(bandas) < 2 else 'una sola banda en prueba')
        filas.append(fila)
        return
    kb = len(bandas)

    sc = StandardScaler().fit(Z[sv])
    zv, zt = sc.transform(Z[sv]), sc.transform(Z[st])
    pca = PCA(random_state=SEED).fit(zv)
    n_comp = int(min(PCA_MAX, len(zv) - 1, np.searchsorted(
        np.cumsum(pca.explained_variance_ratio_), PCA_VARIANZA) + 1))
    pca = PCA(n_components=n_comp, random_state=SEED).fit(zv)
    Xv, Xt = pca.transform(zv), pca.transform(zt)
    fila['componentes'] = n_comp

    # ¿Las bandas reales están separadas? No depende de ningún algoritmo.
    fila['silueta_bandas_val'] = round(silueta(Xv, b_va), 3)
    fila['silueta_bandas_test'] = round(silueta(Xt, b_te), 3)
    fila['estructura_bandas'] = estructura(fila['silueta_bandas_val'])

    km = KMeans(kb, n_init=20, random_state=SEED).fit(Xv)
    lt = km.predict(Xt)
    fila['silueta_kmeans_val'] = round(silueta(Xv, km.labels_), 3)
    fila['ARI_kmeans_val'] = round(adjusted_rand_score(b_va, km.labels_), 3)
    fila['ARI_kmeans_test'] = round(adjusted_rand_score(b_te, lt), 3)
    fila['NMI_kmeans_test'] = round(normalized_mutual_info_score(b_te, lt), 3)

    # k libre: el de mejor silueta en validación (sin mirar las etiquetas).
    sil_k = {}
    for k in K_RANGO:
        if k < len(Xv):
            sil_k[k] = silueta(Xv, KMeans(k, n_init=20, random_state=SEED)
                               .fit(Xv).labels_)
    k_m = max(sil_k, key=sil_k.get)
    km_l = KMeans(k_m, n_init=20, random_state=SEED).fit(Xv)
    fila['k_libre'] = k_m
    fila['ARI_klibre_test'] = round(adjusted_rand_score(b_te, km_l.predict(Xt)), 3)

    try:
        g = GaussianMixture(kb, covariance_type='full', n_init=5, reg_covar=1e-4,
                            random_state=SEED).fit(Xv)
        fila['ARI_gmm_test'] = round(adjusted_rand_score(b_te, g.predict(Xt)), 3)
    except ValueError:              # covarianza degenerada con pocas ventanas
        fila['ARI_gmm_test'] = np.nan

    # Diagnóstico: partir de los centros REALES de cada banda.
    cent = np.vstack([Xv[b_va == nv].mean(0) for nv in bandas])
    km_s = KMeans(kb, init=cent, n_init=1, random_state=SEED).fit(Xv)
    fila['ARI_sembrado_test'] = round(adjusted_rand_score(b_te, km_s.predict(Xt)), 3)

    # Sonda: ¿la gravedad está en el latente aunque no forme grupos?
    lr = LogisticRegression(max_iter=2000, class_weight='balanced').fit(zv, b_va)
    fila['F1_sonda_test'] = round(f1_score(b_te, lr.predict(zt), average='macro'), 3)
    dm = DummyClassifier(strategy='stratified', random_state=SEED).fit(zv, b_va)
    fila['F1_azar_test'] = round(f1_score(b_te, dm.predict(zt), average='macro'), 3)
    filas.append(fila)

    if nivel_grupo == 'tipo':
        p2 = PCA(n_components=2, random_state=SEED).fit(zv)
        mapas[grupo] = (p2.transform(zt), b_te, fila)


# Referencia: todas las ventanas juntas, con la misma receta.
_con_tipo = d.tipo.notna().to_numpy()
probar('todos', 'todos los tipos', base_va & _con_tipo, base_te & _con_tipo)
for t in TIPOS:
    _g = (d.tipo == t).to_numpy()
    probar('tipo', t, base_va & _g, base_te & _g)
# Control más estricto: una sola regla dominante. Se eligen en validación las
# reglas con al menos dos bandas que lleguen a MIN_CLASE ventanas.
_r = d[base_va].groupby('etiqueta_tfidf')[BANDA].value_counts().unstack(fill_value=0)
_r = _r.reindex(columns=NIVELES, fill_value=0)
_reglas = _r[(_r >= MIN_CLASE).sum(axis=1) >= 2].sum(axis=1)
for r in _reglas.sort_values(ascending=False).index[:MAX_REGLAS]:
    _g = (d.etiqueta_tfidf == r).to_numpy()
    probar('regla', f'{r} ({TIPO_REGLA.get(r, "?")})', base_va & _g, base_te & _g)

tabla = pd.DataFrame(filas)
tabla.to_csv(salida('10d_por_tipo.csv'), index=False)

_ver = ['nivel', 'grupo', 'n_val', 'n_test', 'bandas', 'silueta_bandas_val',
        'estructura_bandas', 'ARI_kmeans_test', 'k_libre', 'ARI_klibre_test',
        'ARI_gmm_test', 'ARI_sembrado_test', 'F1_sonda_test', 'F1_azar_test', 'nota']
print('\n' + '=' * 72)
print('RESULTADO (ajustado en validación, informado en prueba)')
print('=' * 72)
pd.set_option('display.width', 220)
print(tabla[[c for c in _ver if c in tabla.columns]].to_string(index=False))
print('\nsilueta_bandas: separación de las bandas REALES en el latente del grupo.')
print('  Kaufman y Rousseeuw: <= 0,25 sin estructura; 0,26-0,50 débil;')
print('  0,51-0,70 razonable; > 0,70 fuerte. Negativa: las bandas se mezclan.')
print('ARI: 0 = azar, 1 = los mismos grupos que leve / media / grave. Es el')
print('  criterio principal: la silueta baja también con dimensiones sin señal,')
print('  aunque K-means igual recupere las bandas.')
print('F1_sonda > F1_azar: la gravedad está en el latente aunque no forme grupos.')


# =============================================================================
# 3. FIGURA: EL LATENTE DE PRUEBA DE CADA TIPO, COLOREADO POR BANDA
# =============================================================================
TINTA, TINTA2, GRILLA, FONDO = '#0b0b0b', '#52514e', '#e1e0d9', '#fcfcfb'
RAMPA = {'1_leve': '#86b6ef', '2_media': '#2a78d6', '3_grave': '#104281'}
NOMBRE = {'operacional': 'Operacional', 'lubricacion': 'Lubricación',
          'mecanica': 'Mecánica'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': TINTA2, 'xtick.color': TINTA2,
                     'ytick.color': TINTA2})
fig, ax = plt.subplots(1, 3, figsize=(15, 5), facecolor=FONDO)
for a, t in zip(ax, TIPOS):
    a.set_facecolor(FONDO)
    a.grid(color=GRILLA, linewidth=0.6)
    a.set_axisbelow(True)
    a.set_xlabel('Componente principal 1')
    if t not in mapas:
        a.text(0.5, 0.5, 'Sin dos bandas con\nventanas suficientes', ha='center',
               va='center', color=TINTA2, transform=a.transAxes)
        a.set_title(NOMBRE[t], color=TINTA, loc='left')
        continue
    xy, b, f = mapas[t]
    for nv, nom in zip(NIVELES, ['Leve', 'Media', 'Grave']):
        s = b == nv
        if s.any():
            a.scatter(xy[s, 0], xy[s, 1], s=12, c=RAMPA[nv], alpha=0.75,
                      linewidths=0, label=f'{nom} ({s.sum():,})')
    a.set_title(f'{NOMBRE[t]} · silueta bandas {f["silueta_bandas_test"]:.2f} · '
                f'ARI {f["ARI_kmeans_test"]:.2f}', color=TINTA, loc='left')
    a.legend(frameon=False, loc='best', markerscale=2)
ax[0].set_ylabel('Componente principal 2')
fig.suptitle(f'{EQUIPO} · latente del autoencoder dentro de cada tipo de alarma '
             f'(ventanas de prueba marcadas con N > 0)', color=TINTA, x=0.01, ha='left')
fig.tight_layout()
fig.savefig(salida('10d_por_tipo.png'), dpi=160, facecolor=FONDO)
plt.close(fig)


# =============================================================================
# 4. RESUMEN
# =============================================================================
print('\n' + '=' * 72)
print(f'RESUMEN — {EQUIPO}')
print('=' * 72)
_ok = tabla.dropna(subset=['ARI_kmeans_test']) if 'ARI_kmeans_test' in tabla else tabla.iloc[:0]
for r in _ok.itertuples():
    print(f'{r.grupo:<22} bandas {r.bandas:<18} silueta bandas {r.silueta_bandas_val:+.2f} '
          f'({r.estructura_bandas}) | ARI K-means {r.ARI_kmeans_test:.3f} | '
          f'sonda F1 {r.F1_sonda_test:.2f} vs azar {r.F1_azar_test:.2f}')
_dentro = _ok[_ok.nivel != 'todos']
if len(_dentro):
    print(f'\nDentro de un mismo tipo o regla: ARI máximo en prueba (K-means) '
          f'{_dentro.ARI_kmeans_test.max():.3f}; silueta máxima de las bandas '
          f'reales {_dentro.silueta_bandas_val.max():.2f}.')
print('Guardado: 10d_por_tipo.csv, 10d_por_tipo.png')
