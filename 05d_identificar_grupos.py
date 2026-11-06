# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 05d — IDENTIFICACIÓN DE LOS GRUPOS DEL LATENTE
=============================================================================
Entrada:  05c_grupos_ventanas.csv, 05c_eleccion.json, 05c_prototipos.csv, {EQUIPO}_limpio.csv
Salida:   05d_nombres_grupos.csv, 05d_lift_familias.csv, 05d_lift_variables.csv, 05d_grupo_nivel.csv,
          05d_arbol.txt/png, 05d_medoides.csv/png, 05d_firma.csv, 05d_etiquetas_ventanas.csv
Nombres, árbol y firma se deciden en validación; prueba solo se informa.
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
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# =============================================================================
# 0. CONSTANTES
# =============================================================================
SEED = 42
W = 60
PRES_MIN = 0.30            # familia presente en al menos el 30 % del grupo
LIFT_MIN = 1.10            # sobrerrepresentada: 10 % más que en el total
APORTE_MIN = 0.05          # variable con al menos el 5 % del error del grupo
DESV_MIN = 0.5             # alta o baja: al menos 0,5 desv. de la normal típica
PROF_MAX = 3               # profundidad máxima del árbol explicativo
FIDELIDAD_MIN = 0.90
K_FIRMA = range(2, 11)
MIN_PRUEBA = 10            # ventanas mínimas para revisar un grupo en prueba
MAX_SILUETA = 5000
NIVELES = ['1_leve', '2_media', '3_grave']
TEXTO_NIVEL = {'0_normal': 'sin alarma', '1_leve': 'leve', '2_media': 'media', '3_grave': 'grave'}
TEXTO_FAM = {'lubricacion': 'lubricación', 'corriente': 'corriente', 'atollo': 'atollo',
             'filtro': 'filtro', 'temperatura': 'temperatura', 'vibracion': 'vibración'}

PUNTO_PARADA = os.environ.get('PUNTO_PARADA', 'perdida')   # 'perdida' o 'separacion'
ARCHIVO_GRUPOS = f'{RUTA_EQ}/05c_grupos_ventanas.csv'
ARCHIVO_ELECCION = f'{RUTA_EQ}/05c_eleccion.json'
ARCHIVO_PROTO = f'{RUTA_EQ}/05c_prototipos.csv'
ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
for _f in [ARCHIVO_GRUPOS, ARCHIVO_ELECCION, ARCHIVO_PROTO, ARCHIVO_LIMPIO, ARCHIVO_CORTE]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste la versión nueva del paso 05c?')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


def guardar(df, nombre):
    df.to_csv(salida(nombre), sep=';', decimal=',', index=False, encoding='utf-8-sig')


eleccion = json.load(open(ARCHIVO_ELECCION, encoding='utf-8'))
if eleccion['punto_parada'] != PUNTO_PARADA:
    raise SystemExit(f"El 05c se corrió con PUNTO_PARADA={eleccion['punto_parada']!r}: corre el 05d igual "
                     f"(o vuelve a correr el 05c con {PUNTO_PARADA!r}).")
print('=' * 72)
print(f'PASO 05d — IDENTIFICACIÓN DE LOS GRUPOS ({EQUIPO})')
print('=' * 72)
print(f"Agrupamiento del 05c: {eleccion['metodo']} · {eleccion['espacio']} · {eleccion['grupos']} grupos · "
      f"silueta val {eleccion['silueta_val_latente']:.3f} / prueba {eleccion['silueta_test_latente']:.3f}")


# =============================================================================
# 1. VENTANAS MARCADAS, SU GRUPO Y SUS VALORES EN UNIDADES DEL SENSOR
# =============================================================================
g = pd.read_csv(ARCHIVO_GRUPOS, sep=';', decimal=',', parse_dates=['timestamp_inicio'])
VARIABLES = [c[5:] for c in g.columns if c.startswith('evar_')]
FAMILIAS = [c[4:] for c in g.columns if c.startswith('fam_')]

_cab = pd.read_csv(ARCHIVO_LIMPIO, nrows=0)
_col = 'timestamp' if 'timestamp' in _cab.columns else 'Time'
df = (pd.read_csv(ARCHIVO_LIMPIO, usecols=[_col] + VARIABLES, parse_dates=[_col], low_memory=False)
      .set_index(_col).sort_index())
pos = df.index.searchsorted(g.timestamp_inicio)
if (pos + W > len(df)).any() or not (df.index[np.minimum(pos, len(df) - 1)].to_numpy()
                                        == g.timestamp_inicio.to_numpy()).all():
    raise SystemExit('Hay ventanas que no están en el limpio: ¿es el mismo archivo que usó el 05c?')
_dur = (df.index[pos + W - 1] - df.index[pos]).total_seconds()
if (_dur > (W - 1) * 10 + 1).any():
    print(f'OJO: {int((_dur > (W - 1) * 10 + 1).sum())} ventanas duran más de 10 min en el limpio.')
VALS = df[VARIABLES].to_numpy(dtype=float)
media_v = np.stack([np.nanmean(VALS[p:p + W], axis=0) for p in pos])     # media de la ventana

corte = json.load(open(ARCHIVO_CORTE))
std_v = df.loc[df.index <= pd.Timestamp(corte['T_train']), VARIABLES].std().replace(0, 1.0)
proto = pd.read_csv(ARCHIVO_PROTO, sep=';', decimal=',')
ref = proto[proto.prototipo == 'normal'].groupby('variable').valor.mean()[VARIABLES]   # normal típica

ev = g[[f'evar_{v}' for v in VARIABLES]].to_numpy()
aporte = ev / ev.sum(axis=1, keepdims=True)                 # parte del error de cada variable
presente = g[[f'fam_{f}' for f in FAMILIAS]].to_numpy() > 0
va, te = (g.conjunto == 'val').to_numpy(), (g.conjunto == 'test').to_numpy()
grupo = g.grupo.to_numpy()
GRUPOS = sorted(int(x) for x in set(grupo[va]) if x >= 0)
print(f'Ventanas marcadas: validación {va.sum():,} | prueba {te.sum():,} | grupos {GRUPOS} '
      f'(ruido: {int((va & (grupo < 0)).sum())} en validación)')


def silueta(X, lab):
    ok = lab >= 0
    if len(set(lab[ok])) < 2:
        return np.nan
    i = np.where(ok)[0]
    if len(i) > MAX_SILUETA:
        i = np.random.RandomState(SEED).choice(i, MAX_SILUETA, replace=False)
    return float(silhouette_score(X[i], lab[i]))


def v_cramer(a, b):
    t = pd.crosstab(np.asarray(a), np.asarray(b))
    if min(t.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.to_numpy().sum() * (min(t.shape) - 1))))


# =============================================================================
# 2. LIFT DE FAMILIAS DE REGLAS Y DE VARIABLES, Y NOMBRE DE CADA GRUPO
# =============================================================================
# lift = % en el grupo / % en todas las ventanas marcadas del mismo conjunto
print('\n' + '=' * 72)
print('2. ¿QUÉ TIENE DE PROPIO CADA GRUPO? (lift; el nombre se decide en validación)')
print('=' * 72)
filas_f, filas_v = [], []
for conj, m in [('val', va), ('test', te)]:
    base_f, base_v = presente[m].mean(axis=0), aporte[m].mean(axis=0)
    for gr in sorted(set(grupo[m])):
        s = m & (grupo == gr)
        pf, pv = presente[s].mean(axis=0), aporte[s].mean(axis=0)
        for j, f in enumerate(FAMILIAS):
            filas_f.append({'conjunto': conj, 'grupo': int(gr), 'ventanas': int(s.sum()), 'familia': f,
                            'pct_presente': round(100 * pf[j], 1), 'pct_total': round(100 * base_f[j], 1),
                            'lift': round(pf[j] / base_f[j], 2) if base_f[j] > 0 else np.nan})
        for j, v in enumerate(VARIABLES):
            med = float(np.median(media_v[s, j]))
            filas_v.append({'conjunto': conj, 'grupo': int(gr), 'ventanas': int(s.sum()), 'variable': v,
                            'pct_error': round(100 * pv[j], 1), 'pct_error_total': round(100 * base_v[j], 1),
                            'lift': round(pv[j] / base_v[j], 2), 'mediana_sensor': round(med, 2),
                            'normal_tipica': round(float(ref[v]), 2), 'diferencia': round(med - ref[v], 2),
                            'diferencia_desv': round((med - ref[v]) / std_v[v], 2)})
lf, lv = pd.DataFrame(filas_f), pd.DataFrame(filas_v)
guardar(lf, '05d_lift_familias.csv')
guardar(lv, '05d_lift_variables.csv')

def sentido(desv):
    return '≈ normal' if abs(desv) < DESV_MIN else ('alta' if desv > 0 else 'baja')


nombre, filas_n, firma_vars = {-1: 'ruido'}, [], {}
for gr in GRUPOS:
    f = lf[(lf.conjunto == 'val') & (lf.grupo == gr)]
    cand = f[f.pct_presente >= 100 * PRES_MIN]
    distintiva = bool(len(cand)) and cand.lift.max() >= LIFT_MIN
    ff = cand.loc[cand.lift.idxmax()] if distintiva else f.loc[f.pct_presente.idxmax()]
    fam = TEXTO_FAM.get(ff.familia, ff.familia)
    if ff.pct_presente == 0:
        fam = 'sin regla activa'
    elif not distintiva and ff.pct_presente < 50:
        fam = 'mixto'                                       # ninguna familia propia ni mayoritaria
    v = lv[(lv.conjunto == 'val') & (lv.grupo == gr)]
    firma = v[v.pct_error >= 100 * APORTE_MIN].nlargest(2, 'lift')
    if firma.empty:
        firma = v.nlargest(2, 'pct_error')
    firma_vars[gr] = list(firma.variable)
    nombre[gr] = f'G{gr} · {fam}: ' + ', '.join(
        f"{r.variable} {sentido(r.diferencia_desv)}" for r in firma.itertuples())

    # ¿se sostiene en prueba? (solo se informa)
    n_te = int((te & (grupo == gr)).sum())
    ft = lf[(lf.conjunto == 'test') & (lf.grupo == gr) & (lf.familia == ff.familia)]
    vt = lv[(lv.conjunto == 'test') & (lv.grupo == gr) & lv.variable.isin(firma.variable)].set_index('variable')
    if n_te < MIN_PRUEBA:
        estado = f'sin ventanas suficientes en prueba ({n_te})'
    else:
        igual = (ft.pct_presente.iloc[0] >= 100 * PRES_MIN and (vt.lift >= 1).all()
                 and all(np.sign(vt.loc[r.variable, 'diferencia']) == np.sign(r.diferencia)
                         for r in firma.itertuples()))
        estado = 'se mantiene' if igual else 'no se mantiene'
    filas_n.append({
        'grupo': gr, 'nombre': nombre[gr], 'ventanas_val': int(ff.ventanas), 'ventanas_test': n_te,
        'familia': TEXTO_FAM.get(ff.familia, ff.familia), 'familia_pct_val': ff.pct_presente, 'familia_lift_val': ff.lift,
        'familia_distintiva': distintiva,
        'firma': ', '.join(f'{r.variable} {r.diferencia:+.1f} ({r.diferencia_desv:+.1f} desv.; '
                           f'{r.pct_error:.0f} % del error, lift {r.lift:.2f})' for r in firma.itertuples()),
        'familia_pct_test': ft.pct_presente.iloc[0] if len(ft) else np.nan,
        'familia_lift_test': ft.lift.iloc[0] if len(ft) else np.nan,
        'lift_firma_test': ', '.join(f'{x} {vt.loc[x, "lift"]:.2f}' for x in firma.variable if x in vt.index),
        'prueba': estado})
nombres = pd.DataFrame(filas_n)
guardar(nombres, '05d_nombres_grupos.csv')
pd.set_option('display.width', 240)
pd.set_option('display.max_colwidth', 90)
for r in nombres.itertuples():
    print(f'\n{r.nombre}   ({r.ventanas_val} ventanas en validación, {r.ventanas_test} en prueba)')
    print(f"  familia: {r.familia} en el {r.familia_pct_val:.0f} % del grupo, lift {r.familia_lift_val:.2f}"
          f"{'' if r.familia_distintiva else ' (no es propia del grupo: es común a todas las marcadas)'}")
    print(f'  firma:   {r.firma}')
    print(f'  prueba:  {r.prueba} (familia {r.familia_pct_test:.0f} %, lift {r.familia_lift_test:.2f}; '
          f'lift de la firma {r.lift_firma_test})')
print('\nlift > 1: más frecuente en el grupo que en todas las marcadas. Alta o baja: mediana del')
print(f'grupo contra la ventana normal típica decodificada, si se aparta al menos {DESV_MIN} desv.;')
print('mixto: ninguna familia de reglas es propia del grupo ni está en la mitad de sus ventanas.')


# =============================================================================
# 3. GRUPO x NIVEL Y ASOCIACIÓN CON EL TIPO Y CON EL NIVEL
# =============================================================================
print('\n' + '=' * 72)
print('3. ¿EL GRUPO DICE EL TIPO O EL NIVEL? (V de Cramér: 0 = nada, 1 = total)')
print('=' * 72)
filas_c = []
for conj, m in [('val', va), ('test', te)]:
    t = pd.crosstab(pd.Series(grupo[m]).map(nombre).to_numpy(), g.nivel[m].to_numpy(),
                    rownames=['grupo'], colnames=['nivel'])
    t = t[[c for c in ['0_normal'] + NIVELES if c in t.columns]]
    print(f'\n{"Validación" if conj == "val" else "Prueba (solo se informa)"}: % de cada nivel dentro del grupo')
    print((100 * t.div(t.sum(axis=1), axis=0)).round(1).assign(ventanas=t.sum(axis=1)).to_string())
    for gn, fila in t.iterrows():
        for nv, c in fila.items():
            filas_c.append({'conjunto': conj, 'grupo': gn, 'nivel': nv, 'ventanas': int(c),
                            'pct_del_grupo': round(100 * c / fila.sum(), 1)})
    s = m & (grupo >= 0) & np.isin(g.nivel, NIVELES)
    print(f'V de Cramér del grupo con el tipo de alarma: {v_cramer(grupo[s], g.tipo_dominante[s]):.3f} | '
          f'con el nivel: {v_cramer(grupo[s], g.nivel[s]):.3f}')
guardar(pd.DataFrame(filas_c), '05d_grupo_nivel.csv')
print('\nSi la V con el tipo supera a la V con el nivel, el grupo dice qué pasa y el nivel cuánto.')


# =============================================================================
# 4. ÁRBOL EXPLICATIVO: REGLAS LEGIBLES QUE REPRODUCEN LOS GRUPOS
# =============================================================================
# Árbol corto que predice el grupo desde valores del sensor y parte del error
# (Moshkovitz et al., 2020). Fidelidad = % de ventanas en el mismo grupo que el 05c.
print('\n' + '=' * 72)
print('4. ÁRBOL EXPLICATIVO (se ajusta en validación)')
print('=' * 72)
FEAT = [f'{v} media' for v in VARIABLES] + [f'{v} % del error' for v in VARIABLES]
X_arbol = np.hstack([media_v, 100 * aporte])
mv_, mt_ = va & (grupo >= 0), te & (grupo >= 0)
yv, yt = grupo[mv_], grupo[mt_]
hoja = max(5, min(int(0.02 * mv_.sum()), int(pd.Series(yv).value_counts().min()) // 2))


def fidelidad(arbol, X, y):
    if len(y) == 0:
        return np.nan, np.nan
    p = arbol.predict(X)
    por_grupo = [np.mean(p[y == c] == c) for c in np.unique(y)]
    return float(np.mean(p == y)), float(np.mean(por_grupo))


arboles, filas_a = {}, []
for prof in range(1, PROF_MAX + 1):
    a = DecisionTreeClassifier(max_depth=prof, min_samples_leaf=hoja, class_weight='balanced',
                               random_state=SEED).fit(X_arbol[mv_], yv)
    fv, fbv = fidelidad(a, X_arbol[mv_], yv)
    ft_, fbt = fidelidad(a, X_arbol[mt_], yt)
    arboles[prof] = a
    filas_a.append({'profundidad': prof, 'fidelidad_val': round(fv, 3), 'fidelidad_balanceada_val': round(fbv, 3),
                    'fidelidad_test': round(ft_, 3), 'fidelidad_balanceada_test': round(fbt, 3)})
tab_a = pd.DataFrame(filas_a)
_ok = tab_a[tab_a.fidelidad_balanceada_val >= FIDELIDAD_MIN]
PROF = int(_ok.profundidad.min() if len(_ok) else tab_a.loc[tab_a.fidelidad_balanceada_val.idxmax(), 'profundidad'])
arbol = arboles[PROF]
print(tab_a.to_string(index=False))
print(f'Profundidad elegida: {PROF} (la menor con fidelidad balanceada ≥ {FIDELIDAD_MIN:.0%} en validación;'
      f' si ninguna llega, la de mayor fidelidad)')


def condicion(j, signo, umbral):
    if j < len(VARIABLES):
        v = VARIABLES[j]
        return f'{v} media {signo} {umbral:.1f} (normal típica {ref[v]:.1f})'
    return f'{VARIABLES[j - len(VARIABLES)]} {signo} {umbral:.0f} % del error'


def caminos(a):
    t, out = a.tree_, {}

    def rec(nodo, conds):
        if t.children_left[nodo] == -1:
            out[nodo] = conds
            return
        j, u = t.feature[nodo], t.threshold[nodo]
        rec(t.children_left[nodo], conds + [condicion(j, '≤', u)])
        rec(t.children_right[nodo], conds + [condicion(j, '>', u)])
    rec(0, [])
    return out


hojas = arbol.apply(X_arbol[mv_])
pred = arbol.predict(X_arbol[mv_])
lineas = []
for nodo, conds in caminos(arbol).items():
    s = hojas == nodo
    if not s.any():
        continue
    c = int(pd.Series(pred[s]).mode()[0])
    lineas.append((c, -int(s.sum()), f'{nombre[c]}  ←  ' + ' y '.join(conds or ['(todas)'])
                   + f'   [{int(s.sum())} ventanas; el {100 * np.mean(yv[s] == c):.0f} % es de ese grupo]'))
lineas = [x[2] for x in sorted(lineas)]
print('\nReglas (validación):')
for x in lineas:
    print('  ' + x)
with open(salida('05d_arbol.txt'), 'w', encoding='utf-8') as f:
    f.write(f'{EQUIPO} · árbol explicativo de profundidad {PROF}\n\n')
    f.write(tab_a.to_string(index=False) + '\n\nReglas (validación):\n')
    f.write('\n'.join(lineas) + '\n\n')
    f.write(export_text(arbol, feature_names=FEAT, decimals=1))
fig, ax = plt.subplots(figsize=(6 + 4 * PROF, 3 + 1.5 * PROF))
plot_tree(arbol, feature_names=FEAT, class_names=[nombre[int(c)] for c in arbol.classes_], filled=True,
          impurity=False, proportion=True, rounded=True, fontsize=8, ax=ax)
ax.set_title(f'{EQUIPO}: árbol que explica los grupos del latente (validación; value con grupos balanceados)')
fig.tight_layout()
fig.savefig(salida('05d_arbol.png'), dpi=150)
plt.close(fig)


# =============================================================================
# 5. MEDOIDE: LA VENTANA REAL MÁS TÍPICA DE CADA GRUPO
# =============================================================================
print('\n' + '=' * 72)
print('5. VENTANA REAL MÁS TÍPICA DE CADA GRUPO (medoide, validación)')
print('=' * 72)
minutos = np.arange(W) * 10 / 60
filas_m = []
fig, ejes = plt.subplots(len(GRUPOS), 4, figsize=(15, 3.2 * len(GRUPOS)), squeeze=False)
for fila_ax, gr in zip(ejes, GRUPOS):
    i = np.where(va & (grupo == gr) & (g.medoide == 1).to_numpy())[0]
    if len(i) == 0:
        continue
    i = i[0]
    filas_m.append({'grupo': nombre[gr], 'timestamp_inicio': g.timestamp_inicio.iloc[i],
                    'nivel': g.nivel.iloc[i], 'tipo_dominante': g.tipo_dominante.iloc[i],
                    'error_reconstruccion': g.error_reconstruccion.iloc[i]})
    v = lv[(lv.conjunto == 'val') & (lv.grupo == gr)]
    vars_ = list(dict.fromkeys(firma_vars[gr] + list(
        v[v.pct_error >= 100 * APORTE_MIN].nlargest(4, 'lift').variable) + list(v.nlargest(4, 'pct_error').variable)))[:4]
    for ax, var in zip(fila_ax, vars_):
        j = VARIABLES.index(var)
        ax.plot(minutos, VALS[pos[i]:pos[i] + W, j], lw=1.6, label='ventana real (medoide)')
        pg = proto[(proto.prototipo == f'grupo {gr}') & (proto.variable == var)]
        pn = proto[(proto.prototipo == 'normal') & (proto.variable == var)]
        ax.plot(pg.minuto, pg.valor, ':', lw=1.6, label='típica del grupo (decodificada)')
        ax.plot(pn.minuto, pn.valor, 'k--', lw=1.2, label='normal típica (decodificada)')
        ax.set_title(f'{var}', fontsize=10)
        ax.grid(alpha=0.3)
    fila_ax[0].set_ylabel(nombre[gr], fontsize=9)
ejes[0][0].legend(fontsize=7)
for ax in ejes[-1]:
    ax.set_xlabel('Minuto de la ventana')
fig.suptitle(f'{EQUIPO}: ventana real más típica de cada grupo')
fig.tight_layout()
fig.savefig(salida('05d_medoides.png'), dpi=150)
plt.close(fig)
medoides = pd.DataFrame(filas_m)
guardar(medoides, '05d_medoides.csv')
print(medoides.to_string(index=False))


# =============================================================================
# 6. ETIQUETA DE DOS EJES: GRUPO (QUÉ) x NIVEL (CUÁNTO)
# =============================================================================
print('\n' + '=' * 72)
print('6. ETIQUETA DE CADA VENTANA: GRUPO · NIVEL')
print('=' * 72)
g['nombre_grupo'] = pd.Series(grupo).map(nombre).to_numpy()
g['etiqueta'] = np.where(grupo < 0, 'ruido',
                         g.nombre_grupo + ' · ' + g.nivel.map(TEXTO_NIVEL).fillna('sin nivel'))
guardar(g[['conjunto', 'timestamp_inicio', 'grupo', 'nombre_grupo', 'nivel', 'etiqueta', 'N',
           'tipo_dominante', 'error_reconstruccion']], '05d_etiquetas_ventanas.csv')
print(pd.crosstab(g.etiqueta, g.conjunto).rename(columns={'val': 'validación', 'test': 'prueba'}).to_string())


# =============================================================================
# 7. COMPLEMENTO: AGRUPAR LA FORMA DEL ERROR, SIN SU TAMAÑO
# =============================================================================
# Parte del error de cada variable (suma 1) con transformación de Hellinger: dice qué
# variables fallan, no cuánto. Si agrupa por tipo y no por nivel, el tamaño queda para el nivel.
print('\n' + '=' * 72)
print('7. COMPLEMENTO: GRUPOS DE LA FORMA DEL ERROR (sin su tamaño)')
print('=' * 72)
H = np.sqrt(aporte)
pv_, pt_ = va & np.isin(g.nivel, NIVELES), te & np.isin(g.nivel, NIVELES)


def nmi(a, b):
    return round(float(normalized_mutual_info_score(np.asarray(a), np.asarray(b))), 3) if len(a) > 1 else np.nan


filas_k = []
for k in K_FIRMA:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(H[va])
    lab = np.full(len(g), -1)
    lab[va], lab[te] = km.labels_, km.predict(H[te])
    filas_k.append({'k': k, 'silueta_val': round(silueta(H[va], lab[va]), 3),
                    'silueta_test': round(silueta(H[te], lab[te]), 3),
                    'NMI_tipo_val': nmi(g.tipo_dominante[pv_], lab[pv_]), 'NMI_nivel_val': nmi(g.nivel[pv_], lab[pv_]),
                    'NMI_tipo_test': nmi(g.tipo_dominante[pt_], lab[pt_]), 'NMI_nivel_test': nmi(g.nivel[pt_], lab[pt_]),
                    'NMI_con_grupos_latente_val': nmi(grupo[mv_], lab[mv_])})
firma_k = pd.DataFrame(filas_k)
guardar(firma_k, '05d_firma.csv')
mejor = firma_k.loc[firma_k.silueta_val.idxmax()]
s_l = pv_ & (grupo >= 0)
print(firma_k.to_string(index=False))
print(f'\nElegido por silueta en validación: k = {int(mejor.k)} (silueta {mejor.silueta_val:.3f}; '
      f'prueba {mejor.silueta_test:.3f})')
print(f'  forma del error -> NMI con el tipo {mejor.NMI_tipo_val:.3f}, con el nivel {mejor.NMI_nivel_val:.3f}')
print(f'  latente (05c)   -> NMI con el tipo {nmi(g.tipo_dominante[s_l], grupo[s_l]):.3f}, '
      f'con el nivel {nmi(g.nivel[s_l], grupo[s_l]):.3f}')
print('Si la forma del error tiene más NMI con el tipo que con el nivel, separa tipos de falla;')
print('el nivel se lee en el tamaño del error o en N.')

print(f'\nGuardado en {RUTA_EQ}: 05d_nombres_grupos.csv, 05d_lift_familias.csv, 05d_lift_variables.csv, '
      f'05d_grupo_nivel.csv, 05d_arbol.txt/png, 05d_medoides.csv/png, 05d_etiquetas_ventanas.csv, 05d_firma.csv')
