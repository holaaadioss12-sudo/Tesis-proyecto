# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 05e — AGRUPAMIENTO CON EL DELTA POR VARIABLE: ¿AGRUPA O DETECTA MEJOR?
=============================================================================
Lo pidió el profesor el 6 de noviembre: probar el agrupamiento con el exceso del
error de cada variable sobre su umbral (paso 04c) en vez del latente (paso 05c).
Entrada:  04c_localizacion_ventanas.csv y 04c_tabla_experto.csv (paso 04c);
          05c_agrupamiento.csv y 05c_eleccion.json (opcional, para comparar con el latente)
Salida:   05e_agrupamiento.csv, 05e_comparacion.csv, 05e_deteccion.csv, 05e_perfil_grupos.csv, 05e_perfil.png
Se decide en validación; prueba solo se informa.
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
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    HDBSCAN = None
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
                             roc_auc_score, roc_curve)


# =============================================================================
# 0. CONSTANTES
# =============================================================================
SEED = 42
PERCENTIL = 95
K_RANGO = range(2, 11)
MAX_RUIDO = 0.20
MAX_SILUETA = 5000
NIVELES = ['1_leve', '2_media', '3_grave']
TEXTO_NIVEL = {'0_normal': 'sin alarma', '1_leve': 'leve', '2_media': 'media', '3_grave': 'grave'}
FORMA = 'forma del exceso (qué variables)'
RELATIVO = 'exceso relativo (qué variables y cuánto)'

PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
ARCHIVO_VENT = f'{RUTA_EQ}/04c_localizacion_ventanas.csv'
ARCHIVO_TABLA = f'{RUTA_EQ}/04c_tabla_experto.csv'
for _f in [ARCHIVO_VENT, ARCHIVO_TABLA]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste el paso 04c?')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


def guardar(df, nombre):
    df.to_csv(salida(nombre), sep=';', decimal=',', index=False, encoding='utf-8-sig')


# =============================================================================
# 1. DATOS
# =============================================================================
print('=' * 72)
print(f'PASO 05e — AGRUPAMIENTO CON EL DELTA POR VARIABLE ({EQUIPO})')
print('=' * 72)
v = pd.read_csv(ARCHIVO_VENT, sep=';', decimal=',')
tabla = pd.read_csv(ARCHIVO_TABLA, sep=';', decimal=',')
if (v.punto_parada != PUNTO_PARADA).any():
    raise SystemExit(f'El 04c se corrió con PUNTO_PARADA={v.punto_parada.iloc[0]!r}: corre el 05e igual '
                     f'(o vuelve a correr el 04c con {PUNTO_PARADA!r}).')
VARIABLES = list(tabla.variable)
REF = tabla.umbral_P95_error.to_numpy(dtype=float)
D = v[[f'delta_{x}' for x in VARIABLES]].to_numpy(dtype=float)
DPOS = np.clip(D, 0, None)
REL = D / REF + 1                                   # error de la variable / su umbral P95
ERR = v.error_reconstruccion.to_numpy(dtype=float)
conj = v.conjunto.to_numpy()
va, te = conj == 'val', conj == 'test'
anom = v.anomala.to_numpy() == 1
marc = v.marcada.to_numpy() == 1
nivel = v.nivel.to_numpy(dtype=object)
tipo = v.familia_dominante.to_numpy(dtype=object)
HAY_NIVEL = bool(np.isin(nivel, NIVELES).any())
print(f'Ventanas: validación {va.sum():,} | prueba {te.sum():,} | marcadas: {(va & marc).sum():,} / '
      f'{(te & marc).sum():,}' + ('' if HAY_NIVEL else ' | sin niveles: corre el 05a y el 04c'))


def forma(M):
    s = DPOS[M].sum(axis=1, keepdims=True)
    return np.sqrt(DPOS[M] / np.where(s > 0, s, 1))       # transformación de Hellinger


def relativo(M):
    return np.log(np.maximum(REL[M], 1.0))                 # log de las veces sobre el umbral


def silueta(X, lab):
    ok = lab >= 0
    if len(set(lab[ok])) < 2:
        return np.nan
    i = np.where(ok)[0]
    if len(i) > MAX_SILUETA:
        i = np.random.RandomState(SEED).choice(i, MAX_SILUETA, replace=False)
    return float(silhouette_score(X[i], lab[i]))


def externos(lab, b, t, a):
    pos = np.isin(b, NIVELES) if HAY_NIVEL else a
    if pos.sum() < 2:
        return np.nan, np.nan, np.nan
    nmi_n = normalized_mutual_info_score(b[pos], lab[pos]) if HAY_NIVEL else np.nan
    ari = adjusted_rand_score(b[pos], lab[pos]) if HAY_NIVEL else np.nan
    return ari, nmi_n, normalized_mutual_info_score(t[pos], lab[pos])


def predecir(modelo, X):
    return modelo.predict(X) if len(X) else np.array([], dtype=int)


# =============================================================================
# 2. AGRUPAMIENTO DE LAS VENTANAS MARCADAS (las mismas que agrupa el 05c)
# =============================================================================
print('\n' + '=' * 72)
print('2. AGRUPAMIENTO DE LAS VENTANAS MARCADAS CON EL DELTA (en vez del latente)')
print('=' * 72)
MV = va & marc & (DPOS.sum(axis=1) > 0)
MT = te & marc & (DPOS.sum(axis=1) > 0)


def agrupar(rep, Xv, Xt):
    filas, etq = [], {}

    def anota(metodo, k, lv, lt):
        ari_v, _, _ = externos(lv, nivel[MV], tipo[MV], anom[MV])
        ari_t, nmi_n, nmi_t = externos(lt, nivel[MT], tipo[MT], anom[MT])
        filas.append({'representacion': rep, 'metodo': metodo, 'grupos': k,
                      'ruido_pct': round(100 * float((lv < 0).mean()), 1),
                      'silueta_val': round(silueta(Xv, lv), 3), 'silueta_test': round(silueta(Xt, lt), 3),
                      'ARI_nivel_val': round(ari_v, 3), 'ARI_nivel_test': round(ari_t, 3),
                      'NMI_nivel_test': round(nmi_n, 3), 'NMI_tipo_test': round(nmi_t, 3)})
        etq[metodo] = (lv, lt)

    for k in K_RANGO:
        if k >= len(Xv):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xv)
        anota(f'K-means k={k}', k, km.labels_, predecir(km, Xt))
    bic = {}
    for k in K_RANGO:
        try:
            bic[k] = GaussianMixture(k, covariance_type='full', reg_covar=1e-4, n_init=3,
                                     random_state=SEED).fit(Xv).bic(Xv)
        except ValueError:
            pass
    if bic:
        kb = min(bic, key=bic.get)
        gm = GaussianMixture(kb, covariance_type='full', reg_covar=1e-4, n_init=5, random_state=SEED).fit(Xv)
        anota(f'GMM k={kb} (BIC)', kb, gm.predict(Xv), predecir(gm, Xt))
    if HDBSCAN is not None and len(Xv) > 10:
        h = HDBSCAN(min_cluster_size=int(min(30, max(5, len(Xv) // 10))), copy=True).fit(Xv)
        g = len(set(h.labels_) - {-1})
        lt = (KNeighborsClassifier(1).fit(Xv, h.labels_).predict(Xt) if g and len(Xt)
              else np.full(len(Xt), -1))
        anota('HDBSCAN', g, h.labels_, lt)
    return filas, etq


filas_all, ETQ = [], {}
for rep, f in [(FORMA, forma), (RELATIVO, relativo)]:
    Xv, Xt = f(MV), f(MT)
    if rep == RELATIVO:
        sc = StandardScaler().fit(Xv)
        Xv, Xt = sc.transform(Xv), (sc.transform(Xt) if len(Xt) else Xt)
    fl, et = agrupar(rep, Xv, Xt)
    filas_all += fl
    ETQ.update({(rep, k): x for k, x in et.items()})
agr = pd.DataFrame(filas_all)
guardar(agr, '05e_agrupamiento.csv')
_ok = agr[(agr.grupos >= 2) & (agr.ruido_pct <= 100 * MAX_RUIDO)].dropna(subset=['silueta_val'])
mejores = _ok.loc[_ok.groupby('representacion').silueta_val.idxmax()]
pd.set_option('display.width', 240)
print(mejores.to_string(index=False))
print(f'\nSe elige por silueta en validación, con hasta {MAX_RUIDO:.0%} de ruido. forma: qué variables se')
print('pasan de su umbral, sin el tamaño; relativo: qué variables y cuántas veces se pasan.')


# =============================================================================
# 3. COMPARACIÓN CON EL LATENTE DEL 05c
# =============================================================================
print('\n' + '=' * 72)
print('3. ¿AGRUPA MEJOR QUE EL LATENTE? (mismas ventanas marcadas)')
print('=' * 72)
COLS = ['grupos', 'ruido_pct', 'silueta_val', 'silueta_test', 'ARI_nivel_val', 'ARI_nivel_test',
        'NMI_nivel_test', 'NMI_tipo_test']
filas_c = []
for r in mejores.itertuples():
    filas_c.append({'espacio': f'delta: {r.representacion}', 'particion': r.metodo,
                    **{c: getattr(r, c) for c in COLS}})
for rep in (FORMA, RELATIVO):
    k3 = agr[(agr.representacion == rep) & (agr.metodo == 'K-means k=3')]
    if len(k3):
        filas_c.append({'espacio': f'delta: {rep}', 'particion': 'K-means k=3 (forzado)',
                        **{c: k3.iloc[0][c] for c in COLS}})
_lat, _el = salida('05c_agrupamiento.csv'), salida('05c_eleccion.json')
if os.path.exists(_lat) and os.path.exists(_el):
    lat = pd.read_csv(_lat, sep=';', decimal=',')
    el = json.load(open(_el, encoding='utf-8'))
    if el.get('punto_parada') != PUNTO_PARADA:
        print(f"OJO: el 05c se corrió con PUNTO_PARADA={el.get('punto_parada')!r}.")
    f = lat[(lat.representacion == el['representacion']) & (lat.metodo == el['metodo'])]
    if len(f):
        r = f.iloc[0]
        filas_c.append({'espacio': 'latente (05c)', 'particion': f"{r.metodo} ({r.espacio})",
                        'grupos': r.grupos, 'ruido_pct': r.ruido_pct,
                        'silueta_val': r.silueta_val_latente, 'silueta_test': r.silueta_test_latente,
                        **{c: r[c] for c in COLS[4:]}})
    f3 = lat[(lat.representacion == el['representacion'].split(' + UMAP')[0]) & (lat.metodo == 'K-means k=3')]
    if len(f3):
        r = f3.iloc[0]
        filas_c.append({'espacio': 'latente (05c)', 'particion': 'K-means k=3 (forzado)',
                        **{c: r[c] for c in COLS}})
    print(f'Mayor ARI con los niveles en validación, entre todas las particiones: latente '
          f'{lat.ARI_nivel_val.max():.3f} | delta {agr.ARI_nivel_val.max():.3f}')
else:
    print('Sin resultados del 05c: se omite la comparación con el latente.')
comp = pd.DataFrame(filas_c)
guardar(comp, '05e_comparacion.csv')
print(comp.to_string(index=False))
print('\nLa silueta se mide en el espacio de cada uno y no se compara directo entre espacios; lo comparable')
print('es el ARI y la NMI. ARI con los niveles: 1 = los grupos son leve / media / grave, 0 = azar.')
print('NMI de tipo mayor que NMI de nivel: los grupos se parecen más al tipo de alarma que al nivel.')


# =============================================================================
# 4. ¿DETECTA MEJOR? TODAS LAS VENTANAS, CONTRA EL ERROR TOTAL
# =============================================================================
print('\n' + '=' * 72)
print('4. ¿DETECTA MEJOR? (todas las ventanas; etiqueta: alguna regla activa)')
print('=' * 72)
Xb_v, Xb_t = relativo(va), relativo(te)
sc = StandardScaler().fit(Xb_v)
Xb_v, Xb_t = sc.transform(Xb_v), sc.transform(Xb_t)
y_v, y_t = anom[va], anom[te]
filas_k, mejor = [], None
for k in K_RANGO:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xb_v)
    lv, lt = km.labels_, km.predict(Xb_t)
    tasa = np.array([y_v[lv == g].mean() if (lv == g).any() else 0.0 for g in range(k)])
    auc_v = roc_auc_score(y_v, tasa[lv]) if 0 < y_v.sum() < len(y_v) else np.nan
    auc_t = roc_auc_score(y_t, tasa[lt]) if 0 < y_t.sum() < len(y_t) else np.nan
    filas_k.append({'k': k, 'silueta_val': round(silueta(Xb_v, lv), 3), 'auc_val': round(auc_v, 3),
                    'auc_test': round(auc_t, 3)})
    if mejor is None or auc_v > mejor['auc_val']:
        mejor = {'k': k, 'auc_val': auc_v, 'lv': lv, 'lt': lt, 'tasa': tasa}
print(pd.DataFrame(filas_k).to_string(index=False))
print(f"k elegido por AUC en validación: {mejor['k']}")


def umbral_youden(y, s):
    fpr, tpr, u = roc_curve(y, s)
    return float(u[int(np.argmax(tpr - fpr))])       # máximo de sensibilidad + especificidad


def prf(y, pred):
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12), tn / max(tn + fp, 1)


umbral_total = float(np.percentile(ERR[va & ~anom], PERCENTIL))
DET = [('error total del autoencoder (MSE)', ERR[va], ERR[te], umbral_total, f'P{PERCENTIL} de las normales'),
       ('variable que más se pasa de su umbral', REL[va].max(axis=1), REL[te].max(axis=1), 1.0,
        'alguna variable sobre su P95'),
       (f"agrupamiento del exceso relativo (K-means, k = {mejor['k']})", mejor['tasa'][mejor['lv']],
        mejor['tasa'][mejor['lt']], None, '')]
filas_d = []
for nombre, sv, st, u_fijo, desc in DET:
    fila = {'detector': nombre,
            'auc_val': roc_auc_score(y_v, sv) if 0 < y_v.sum() < len(y_v) else np.nan,
            'auc_test': roc_auc_score(y_t, st) if 0 < y_t.sum() < len(y_t) else np.nan,
            'umbral_fijo': desc}
    if u_fijo is not None:
        fila['f1_test_umbral_fijo'] = prf(y_t, st > u_fijo)[2]
    u = umbral_youden(y_v, sv)
    p, r, f1, esp = prf(y_t, st >= u)
    fila.update({'umbral_youden_val': u, 'precision_test': p, 'recall_test': r, 'especificidad_test': esp,
                 'f1_test': f1})
    filas_d.append(fila)
det = pd.DataFrame(filas_d).round(3)
guardar(det, '05e_deteccion.csv')
print(det.to_string(index=False))
print('\nauc_test y f1_test son de prueba. El umbral se fija en validación con el máximo de sensibilidad +')
print('especificidad (Youden): con el máximo de F1 se marcaría casi todo, porque las anómalas son mayoría.')
print('El agrupamiento')
print('usa las etiquetas de validación para saber qué grupos son anómalos (su AUC de validación es')
print('optimista); el error total y el máximo por variable no usan etiquetas.')


# =============================================================================
# 5. PERFIL DE LOS GRUPOS
# =============================================================================
print('\n' + '=' * 72)
print('5. ¿QUÉ ES CADA GRUPO?')
print('=' * 72)
filas_p, mapas = [], []
f = mejores[mejores.representacion == FORMA]
if len(f):
    lv, _ = ETQ[(FORMA, f.iloc[0].metodo)]
    parte = DPOS[MV] / DPOS[MV].sum(axis=1, keepdims=True)
    filas_mapa = []
    for g in sorted(set(lv)):
        s = lv == g
        niv = pd.Series(nivel[MV][s]).value_counts(normalize=True)
        tp = pd.Series(tipo[MV][s]).value_counts(normalize=True)
        top = np.argsort(-parte[s].mean(axis=0))[:3]
        filas_p.append({'poblacion': 'marcadas (forma del exceso)', 'grupo': 'ruido' if g < 0 else int(g),
                        'ventanas_val': int(s.sum()),
                        **{f'pct_{TEXTO_NIVEL[n]}': round(100 * niv.get(n, 0), 1) for n in TEXTO_NIVEL},
                        'tipo_principal': f'{tp.index[0]} ({100 * tp.iloc[0]:.0f} %)' if len(tp) else '',
                        'variables': ', '.join(f'{VARIABLES[j]} {100 * parte[s][:, j].mean():.0f} %' for j in top)})
        filas_mapa.append(100 * parte[s].mean(axis=0))
    mapas.append(('Marcadas: % del exceso de cada variable', np.array(filas_mapa),
                  [f"grupo {r['grupo']} ({r['ventanas_val']})" for r in filas_p], '%'))
lv, lt = mejor['lv'], mejor['lt']
filas_mapa, etiquetas_b = [], []
for g in range(mejor['k']):
    s, st = lv == g, lt == g
    med = np.median(REL[va][s], axis=0) if s.any() else np.full(len(VARIABLES), np.nan)
    top = np.argsort(-np.nan_to_num(med))[:3]
    fila = {'poblacion': 'todas (exceso relativo)', 'grupo': g, 'ventanas_val': int(s.sum()),
            'pct_anomalas_val': round(100 * y_v[s].mean(), 1) if s.any() else np.nan,
            'pct_anomalas_test': round(100 * y_t[st].mean(), 1) if st.any() else np.nan,
            'variables': ', '.join(f'{VARIABLES[j]} {med[j]:.1f} veces' for j in top)}
    filas_p.append(fila)
    filas_mapa.append(med)
    etiquetas_b.append(f"grupo {g} ({fila['pct_anomalas_val']:.0f} % anómalas)")
mapas.append(('Todas: veces sobre el umbral (mediana, escala log)', np.clip(np.array(filas_mapa), 1, None),
              etiquetas_b, 'veces'))
perfil = pd.DataFrame(filas_p)
guardar(perfil, '05e_perfil_grupos.csv')
for pob, s in perfil.groupby('poblacion', sort=False):
    print(f'\n{pob}')
    print(s.dropna(axis=1, how='all').drop(columns='poblacion').to_string(index=False))

fig, ejes = plt.subplots(1, len(mapas), figsize=(7.5 * len(mapas), 0.45 * max(len(m[2]) for m in mapas) + 2.4),
                         squeeze=False)
for ax, (titulo, mat, etq, unidad) in zip(ejes[0], mapas):
    norma = LogNorm(vmin=1, vmax=max(float(np.nanmax(mat)), 1.01)) if unidad == 'veces' else None
    im = ax.imshow(mat, aspect='auto', cmap='Reds', norm=norma)
    ax.set_xticks(range(len(VARIABLES)))
    ax.set_xticklabels(VARIABLES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(etq)))
    ax.set_yticklabels(etq, fontsize=8)
    ax.set_title(titulo, fontsize=10)
    fig.colorbar(im, ax=ax, label=unidad)
fig.suptitle(f'{EQUIPO}: perfil de los grupos hechos con el delta por variable (validación)')
fig.tight_layout()
fig.savefig(salida('05e_perfil.png'), dpi=150)
plt.close(fig)
print(f'\nGuardado en {RUTA_EQ}: 05e_agrupamiento.csv, 05e_comparacion.csv, 05e_deteccion.csv, '
      f'05e_perfil_grupos.csv, 05e_perfil.png')
