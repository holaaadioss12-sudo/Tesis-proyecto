# -*- coding: utf-8 -*-
"""
=============================================================================
diagnosticos.py  (v2)  -  las mediciones que deciden antes de programar
=============================================================================
Cada una contesta una pregunta que hoy se esta respondiendo por default. Todas
son baratas y todas pueden cambiar decisiones de diseno.

  1. distribucion_tramos()         cuanto cuesta de verdad el margen de guarda
  2. tiempo_establecimiento()      cuanto deberia valer tau, medido
  3. n_efectivo()                  que esta midiendo el indice W en cada variable
  4. degeneracion_criticidad()     si la Fase B tiene algo que clasificar
  5. complementariedad_w_vs_ae()   si el indice W aporta algo que el AE no ve

QUE CAMBIA EN LA v2
-----------------------------------------------------------------------------
A. tiempo_establecimiento() ya no es O(n x n_tramos).
   La v1 hacia `np.flatnonzero((bloque == b).to_numpy())` DENTRO del bucle de
   tramos: con 1,18 M filas y 1.929 tramos son ~2,3e9 comparaciones solo para
   marcar el regimen, y eso en Colab no termina. Ahora el regimen se marca
   vectorizado con el largo de bloque mapeado por fila: una pasada, O(n).

B. La banda de tolerancia se escala con el TRANSITORIO, no con el ruido.
   La v1 usaba k_banda * IQR_regimen. El IQR del regimen mide cuanto fluctua
   el sensor cuando ya esta estable -- no tiene nada que ver con el tamano del
   escalon de arranque. Resultado: tau salia 12 min donde el transitorio real
   era de ~71. Ahora la banda es una fraccion del ESCALON medido (|valor al
   arrancar - valor de regimen|): eso es t90 y t95, que es como se define un
   tiempo de establecimiento en control de procesos y como se puede citar.
   El tau viejo se sigue reportando en la columna tau_iqr, para que el
   contraste quede visible.

C. Una variable cuyo escalon es mas chico que su propio ruido no tiene
   transitorio que esperar. Antes se le calculaba un tau igual y empujaba el
   maximo. Ahora se marca sin_transitorio y se excluye de la decision: por eso
   el paso 3 del orden de ejecucion dice "el t95 de la variable de escalon mas
   grande", no "el maximo de los t95".

D. distribucion_tramos() agrega tramos_muertos. El costo de la guarda no es
   proporcional: un tau grande no solo le corta la cabeza a cada tramo, MATA
   enteros los que miden menos que tau + W. Esa es la columna que decide, con
   pct_ventanas -- no pct_filas, que subestima el dano.

E. complementariedad_w_vs_ae(), nueva. El indice W y el error del autoencoder
   pueden estar detectando exactamente las mismas ventanas, y en ese caso el
   indice es un adorno caro. Esto lo mide antes de escribir el capitulo.
=============================================================================
"""

import numpy as np
import pandas as pd

PASO_S = 10


# =============================================================================
# 1. ¿Cuanto cuesta el margen de guarda? La respuesta esta en los largos
# =============================================================================
def distribucion_tramos(df, W=60, guardas_min=(0, 5, 15, 30, 45, 60, 90)):
    """Lo primero que hay que mirar, antes de elegir tau.

    Con ~1,18 M filas a 10 s y ~1.929 tramos, el tramo promedio dura 1,7 h.
    Un margen de 60 min por arranque se come buena parte del material. Si la
    mediana de tramo sale en 40 min y tau sale 90, ya sabes la respuesta antes
    de leer el resto de la tabla.

    LA COLUMNA QUE DECIDE ES ventanas, NO filas. Un tau grande no solo le corta
    la cabeza a cada tramo: mata entero cualquier tramo mas corto que tau + W,
    y esa perdida es abrupta, no proporcional. pct_filas la suaviza y hace
    parecer barato un margen que en ventanas es carisimo.
    """
    salto = df.index.to_series().diff() > pd.Timedelta(seconds=PASO_S)
    tam = df.groupby(salto.cumsum()).size()

    print(f'Tramos: {len(tam):,}   filas: {tam.sum():,}')
    print(f'Horas de operacion: {tam.sum()*PASO_S/3600:,.0f}')
    print('\nLargo de tramo (minutos):')
    print((tam * PASO_S / 60).describe(
        percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    print(f'\nTramos mas cortos que W={W}: {int((tam < W).sum()):,} '
          f'(se pierden enteros)')

    filas = []
    n0 = tam.sum()
    v0 = int((tam // W).sum())
    n_tramos = len(tam)
    for g in guardas_min:
        t = int(g * 60 / PASO_S)
        util = (tam - t).clip(lower=0)
        viva = util >= W
        muertos = int(n_tramos - viva.sum())
        filas.append({'guarda_min': g,
                      'tramos_vivos': int(viva.sum()),
                      'tramos_muertos': muertos,
                      'pct_tramos_muertos': round(100 * muertos / max(n_tramos, 1), 1),
                      'filas': int(util[viva].sum()),
                      'pct_filas': round(100 * util[viva].sum() / n0, 1),
                      'ventanas': int((util[viva] // W).sum()),
                      'pct_ventanas': round(100 * (util[viva] // W).sum() / max(v0, 1), 1)})
    tabla = pd.DataFrame(filas)
    print('\nCosto de cada margen de guarda:')
    print(tabla.to_string(index=False))
    print('\nCompara pct_filas con pct_ventanas en la misma fila: la brecha')
    print('entre las dos es el dano que la cuenta por filas no ve. Y mira')
    print('tramos_muertos: son arranques que desaparecen completos del')
    print('entrenamiento, no recortados -- borrados.')
    return tabla


# =============================================================================
# 2. Tiempo de establecimiento, medido como corresponde
# =============================================================================
def tiempo_establecimiento(df, variables, W=60, fracciones=(0.10, 0.05),
                           k_banda=0.25, lag_max_min=360, min_tramos=20,
                           horas_regimen=6):
    """Tau por variable y por estrato de parada previa, con criterio t90 / t95.

    QUE MIDE. Para cada estrato y cada variable arma la trayectoria mediana
    desde el instante de arranque, la compara con el nivel de regimen, y busca
    el ultimo instante en que todavia esta fuera de una banda. Ese instante
    + 1 es tau: desde ahi en adelante la mediana ya no sale de la banda.

    LA BANDA. Fraccion del ESCALON, no del ruido. escalon = |valor al arrancar
    - nivel de regimen|. Con fraccion 0,10 eso es t90 (llego al 90 % del
    recorrido y se queda), con 0,05 es t95. Asi se define un tiempo de
    establecimiento en control de procesos, se puede citar, y no depende de
    cuanto ruido tenga el sensor. La v1 usaba k * IQR_regimen, que mide ruido y
    no transitorio: por eso devolvia 12 min donde el transitorio era de ~71.
    tau_iqr queda en la tabla para que el contraste se vea.

    SIN TRANSITORIO. Si el escalon es mas chico que el IQR del regimen, la
    variable arranca practicamente en su nivel normal y no hay nada que
    esperar. Se marca sin_transitorio=True y NO entra en la decision de tau.
    Es el caso de la variable que arranca a 0,4 C de su regimen: por lenta que
    sea, no contamina nada.

    DOS COSAS MAS, que una version simple se pierde y que cambian el resultado:

    HIPOS DE ADQUISICION. Un tramo precedido por un hueco de pocos segundos no
    empieza en un arranque: el equipo nunca se detuvo, se corto el dato. Esos
    tramos aportan trayectorias que YA estan en regimen y sesgan tau HACIA
    ABAJO. Se descartan con el umbral de 2 minutos.

    ESTRATIFICACION. Tras 15 minutos detenido el aceite sigue caliente; tras
    ocho horas el equipo arranca frio. Son transitorios distintos y
    promediarlos junta dos fenomenos. De aca sale ademas la politica que mas
    datos salva: tau condicional al estrato, en vez de pagar el tau largo
    despues de cada parada corta.

    Se usa la MEDIANA entre tramos, no el promedio, para que un arranque raro
    no mande.
    """
    salto = df.index.to_series().diff() > pd.Timedelta(seconds=PASO_S)
    bloque = salto.cumsum()
    pos = df.groupby(bloque).cumcount().to_numpy()

    idx = df.index.to_series()
    ini_pos = np.flatnonzero(pos == 0)
    huecos = np.full(len(ini_pos), np.inf)
    seg = (idx.to_numpy().astype('datetime64[s]').astype(np.int64))
    for k, p in enumerate(ini_pos):
        if p > 0:
            huecos[k] = float(seg[p] - seg[p - 1])

    def estrato(h):
        if h < 120:       return 'hipo'       # no hubo detencion
        if h < 3600:      return 'corta'
        if h < 8 * 3600:  return 'media'
        return 'larga'

    est = np.array([estrato(h) for h in huecos])
    print('Tramos por estrato de parada previa:')
    print(pd.Series(est).value_counts().to_string())
    print('  "hipo" queda FUERA: son cortes de adquisicion, no arranques.')

    lag_max = int(lag_max_min * 60 / PASO_S)

    # --- regimen, VECTORIZADO -------------------------------------------------
    # v1: np.flatnonzero((bloque == b).to_numpy()) dentro del bucle de tramos.
    # Con 1,18 M filas x 1.929 tramos son ~2,3e9 comparaciones y no termina.
    # Aca: largo de bloque mapeado por fila, una pasada.
    largo = df.groupby(bloque).size()
    n_fila = bloque.map(largo).to_numpy()
    n_min_reg = int(horas_regimen * 3600 / PASO_S)
    reg = (n_fila > n_min_reg) & (pos >= (3 * n_fila) // 4)
    print(f'\nRegimen: {int(reg.sum()):,} filas '
          f'({100*reg.mean():.1f} %), del ultimo cuarto de los tramos de mas '
          f'de {horas_regimen} h.')
    if reg.sum() < 1000:
        print('  >> Muy pocas filas de regimen. Baja horas_regimen o el nivel')
        print('     de referencia va a ser ruidoso y tau no va a significar nada.')

    f90, f95 = fracciones[0], fracciones[1]
    filas = []
    for e in ('corta', 'media', 'larga'):
        sel = ini_pos[est == e]
        if len(sel) < min_tramos:
            print(f'  estrato {e}: {len(sel)} tramos, insuficiente')
            continue
        for v in variables:
            x = df[v].to_numpy(dtype=np.float64)
            acc = np.full((len(sel), lag_max), np.nan)
            for j, p in enumerate(sel):
                fin = min(p + lag_max, len(x))
                # cortar en el siguiente arranque, si cae dentro del horizonte
                mismo = np.flatnonzero(pos[p:fin] == 0)
                if len(mismo) > 1:
                    fin = p + int(mismo[1])
                acc[j, :fin - p] = x[p:fin]

            n_ev = (~np.isnan(acc)).sum(0)
            with np.errstate(all='ignore'):
                med = np.nanmedian(acc, 0)
            med[n_ev < min_tramos] = np.nan
            valid = np.flatnonzero(np.isfinite(med))
            if len(valid) < 2:
                continue
            med = med[:valid[-1] + 1]

            ref = x[reg]
            ref = ref[np.isfinite(ref)]
            if len(ref) < 100:
                continue
            m = float(np.median(ref))
            iqr = float(np.subtract(*np.percentile(ref, [75, 25])))
            iqr = iqr if iqr > 0 else np.nan

            desv = np.abs(med - m)
            escalon = float(desv[np.isfinite(desv)][0]) if np.isfinite(desv).any() else np.nan
            sin_trans = bool(np.isfinite(iqr) and np.isfinite(escalon)
                             and escalon < iqr)

            def tau_con(banda):
                if not np.isfinite(banda) or banda <= 0:
                    return 0
                fuera = np.flatnonzero(desv > banda)
                return (int(fuera[-1]) + 1) if len(fuera) else 0

            t90 = tau_con(f90 * escalon)
            t95 = tau_con(f95 * escalon)
            t_iqr = tau_con(k_banda * iqr)

            filas.append({
                'estrato': e, 'variable': v,
                'escalon': round(escalon, 3) if np.isfinite(escalon) else np.nan,
                'escalon_en_iqr': round(escalon / iqr, 2) if np.isfinite(iqr) else np.nan,
                'sin_transitorio': sin_trans,
                't90_min': round(t90 * PASO_S / 60, 1),
                't95_min': round(t95 * PASO_S / 60, 1),
                'tau_iqr_min': round(t_iqr * PASO_S / 60, 1),
                # saturado = la curva seguia fuera de la banda cuando se acabo
                # el horizonte. Ese t95 no es una medicion, es un PISO.
                'saturado': bool(t95 >= lag_max - 1),
                'n_tramos': int(len(sel)),
            })

    t = pd.DataFrame(filas)
    if t.empty:
        print('\nNo hay material suficiente en ningun estrato.')
        return t

    print('\nTiempo de establecimiento por variable y estrato:')
    print(t.sort_values(['estrato', 't95_min'], ascending=[True, False])
          .to_string(index=False))

    sat = t[t.saturado]
    if len(sat):
        print('\n' + '!' * 70)
        print(f'SATURADAS: estas dieron t95 = {lag_max_min} min, que es el')
        print('horizonte de la medicion, no su tiempo de establecimiento. La')
        print('curva seguia fuera de la banda cuando se dejo de mirar, asi que')
        print('el valor verdadero es MAYOR. Es un piso.')
        print(sat[['estrato', 'variable', 'escalon', 't90_min', 't95_min']]
              .to_string(index=False))
        print(f'\nSube lag_max_min (hoy {lag_max_min}) y volve a correr, o')
        print('declara ese estrato como excluido. Lo que no se puede es poner')
        print(f'{lag_max_min} en la guarda y llamarlo "el t95 medido".')
        print('!' * 70)

    print('\n--- t95 en minutos, pivote ---')
    piv = t.pivot(index='variable', columns='estrato', values='t95_min')
    print(piv.to_string())

    utiles = t[~t.sin_transitorio]
    descartadas = sorted(set(t.loc[t.sin_transitorio, 'variable']))
    if descartadas:
        print(f'\nSin transitorio apreciable (escalon < IQR del regimen), '
              f'excluidas de la decision: {descartadas}')
        print('Arrancan practicamente en su nivel normal. Por lentas que sean,')
        print('no contaminan la ventana.')

    if len(utiles):
        print('\n--- La variable que manda, por estrato ---')
        for e, g in utiles.groupby('estrato'):
            f = g.loc[g.escalon_en_iqr.idxmax()]
            print(f'  {e:>6}: {f.variable} '
                  f'(escalon {f.escalon_en_iqr:.1f} IQR) -> '
                  f't95 = {f.t95_min:.1f} min, t90 = {f.t90_min:.1f} min')
        print('\nESTE es el numero que va al paso 01, no el maximo de la')
        print('columna t95: el maximo lo puede estar poniendo una variable con')
        print('un escalon insignificante que tarda en asentarse por lenta, no')
        print('por estar lejos del regimen.')
        print('\nY compara t95_min con tau_iqr_min en la misma fila: la brecha')
        print('entre las dos es exactamente el error de escala de la v1, con')
        print('tus propios datos. Ese contraste va a la memoria.')
    return t


# =============================================================================
# 3. ¿Que esta midiendo el indice W? Depende de la autocorrelacion
# =============================================================================
def n_efectivo(X_normal, variables, W=60):
    """Tamano efectivo de muestra dentro de una ventana, por variable.

    Esto no estaba en ningun lado y cambia como se interpreta el indice W.

    A 10 segundos de muestreo las temperaturas tienen rho cerca de 0,97. Para
    un AR(1), n_ef = n(1-rho)/(1+rho). Con rho = 0,97 y n = 60 eso da MENOS DE
    UNO: la ventana es practicamente un solo punto del proceso lento.

    Consecuencia: para las variables lentas el indice W a W=60 no mide un
    corrimiento de distribucion, mide DONDE ESTA EL NIVEL ahora mismo respecto
    del rango normal. Es interpretable y sirve, pero hay que decirlo asi en la
    memoria, y es el argumento para usar un W mas largo en el indice que en el
    autoencoder.

    X_normal: (n_ventanas, W, n_vars) estandarizado.
    """
    filas = []
    for i, v in enumerate(variables):
        A = X_normal[:, :, i]
        a = A - A.mean(axis=1, keepdims=True)
        num = (a[:, :-1] * a[:, 1:]).sum(axis=1)
        den = (a * a).sum(axis=1)
        rho = float(np.median(np.divide(num, den, out=np.zeros_like(num),
                                        where=den > 0)))
        rho = min(max(rho, -0.999), 0.999)
        filas.append({'variable': v, 'rho_lag1': round(rho, 4),
                      'n_efectivo': round(W * (1 - rho) / (1 + rho), 2)})
    t = pd.DataFrame(filas).sort_values('n_efectivo')
    print(t.to_string(index=False))
    print(f'\nn_efectivo << {W} significa que la ventana no tiene informacion')
    print('distribucional: el indice W ahi es un indicador de nivel, no de')
    print('corrimiento de forma. Declararlo, no esconderlo.')
    print('\nY ojo con el caso extremo: una variable constante DENTRO y ENTRE')
    print('ventanas -- un sensor pegado -- da IQR de la nula cerca de cero y')
    print('el indice estandarizado explota. Eso no es n_ef < 1, es division')
    print('por casi cero, y es otro problema. indice_w.validar_nula() lo')
    print('detecta por el piso del IQR.')
    return t


# =============================================================================
# 4. ¿Tiene la Fase B algo que clasificar?
# =============================================================================
def degeneracion_criticidad(ventanas, col='criticidad_min', min_por_clase=50):
    """Hay que correr esto ANTES de construir la Fase B.

    Por el anidamiento, la vibracion es la regla mas parlanchina y sus niveles
    altos (A19 > 34, A18 > 40) son criticidad 1. Si eso arrastra casi todas las
    ventanas anomalas a criticidad 1, la escala experta queda casi degenerada
    sobre estos datos y no hay nada que clasificar.

    Si una clase tiene una sola ventana, un R2 o una exactitud sobre ese
    objetivo no significan nada.
    """
    d = ventanas[col].value_counts().sort_index()
    print(f'Distribucion de {col} (1 = mas grave):')
    print(pd.DataFrame({'ventanas': d,
                        'pct': (100*d/len(ventanas)).round(1)}).to_string())

    anom = ventanas[ventanas[col] < 4]
    if len(anom):
        da = anom[col].value_counts().sort_index()
        print(f'\nSolo sobre las {len(anom):,} ventanas anomalas:')
        print(pd.DataFrame({'ventanas': da,
                            'pct': (100*da/len(anom)).round(1)}).to_string())
        if (da.max() / max(len(anom), 1)) > 0.90:
            print('\n  >> Mas del 90 % en una sola clase. La escala experta esta')
            print('     degenerada sobre estos datos. La Fase B no puede ser')
            print('     clasificacion de gravedad: hay que clasificar COMPONENTE')
            print('     o MODO DE FALLA, que si tienen varias clases pobladas.')
        else:
            print('\n  >> El reparto NO esta degenerado: la Fase B como')
            print('     clasificacion de gravedad es viable sobre estos datos.')
            print('     Si las clases 1 y 2 concentran casi todo, plantearla')
            print('     binaria grave/media es mas defendible que a tres clases.')
    flacas = d[d < min_por_clase]
    if len(flacas):
        print(f'\n  Clases con menos de {min_por_clase} ventanas: '
              f'{flacas.to_dict()}')
        print('  No reportes metricas por clase sobre esas.')

    if 'componente' in ventanas.columns:
        print('\n--- Por componente, solo ventanas anomalas ---')
        dc = anom.componente.value_counts()
        print(pd.DataFrame({'ventanas': dc,
                            'pct': (100*dc/max(len(anom), 1)).round(1)}).to_string())
        print('Esta es la alternativa si la gravedad sale degenerada: aca se ve')
        print('si hay clases pobladas suficientes para clasificar componente.')
    return d


# =============================================================================
# 5. ¿Aporta el indice W algo que el autoencoder no vea?
# =============================================================================
def complementariedad_w_vs_ae(idx_w, E_rel, variables, q=0.90, agregado='max'):
    """Correla el indice W con el error relativo del AE, ventana por ventana.

    POR QUE HAY QUE MEDIRLO. El indice W y el error de reconstruccion pueden
    estar marcando exactamente las mismas ventanas por las mismas razones. Si
    es asi, el indice es un adorno caro y la memoria tiene que decirlo; si no
    lo es, la complementariedad es un resultado y justifica tener los dos.
    Lo que no se puede hacer es presentar dos indicadores sin saber si son el
    mismo indicador dos veces.

    Y ADEMAS. n_efectivo() ya mostro que para las variables lentas el indice W
    a W=60 mide NIVEL, mientras el autoencoder reconstruye FORMA dentro de la
    ventana. Si eso es cierto, las correlaciones por variable tienen que salir
    bajas justo en las variables de n_ef chico -- y ese cruce entre las dos
    tablas es el argumento, no la correlacion sola.

    idx_w : (n_ventanas, n_vars) indice W estandarizado, de indice_w.transformar
    E_rel : (n_ventanas, n_vars) error de reconstruccion relativo, E / base
    """
    idx_w = np.asarray(idx_w, dtype=float)
    E_rel = np.asarray(E_rel, dtype=float)
    if idx_w.shape != E_rel.shape:
        raise ValueError(f'formas distintas: {idx_w.shape} vs {E_rel.shape}')
    if idx_w.shape[1] != len(variables):
        raise ValueError(f'{idx_w.shape[1]} columnas y {len(variables)} variables')

    def spearman(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            return np.nan
        ra = pd.Series(a[ok]).rank().to_numpy()
        rb = pd.Series(b[ok]).rank().to_numpy()
        if ra.std() == 0 or rb.std() == 0:
            return np.nan
        return float(np.corrcoef(ra, rb)[0, 1])

    filas = []
    for i, v in enumerate(variables):
        filas.append({'variable': v,
                      'rho_spearman': round(spearman(idx_w[:, i], E_rel[:, i]), 3)})
    t = pd.DataFrame(filas).sort_values('rho_spearman')
    print('Correlacion de rangos por variable, indice W contra error del AE:')
    print(t.to_string(index=False))
    print('\nCruza esta columna con n_efectivo: si las correlaciones bajas caen')
    print('en las variables de n_ef chico, queda demostrado que los dos')
    print('indicadores miden cosas distintas y por que.')

    red = (lambda M: M.max(axis=1)) if agregado == 'max' else (lambda M: M.mean(axis=1))
    s_w, s_ae = red(idx_w), red(E_rel)
    print(f'\nAgregado por ventana ({agregado}): rho = '
          f'{spearman(s_w, s_ae):.3f}')

    u_w = float(np.nanquantile(s_w, q))
    u_ae = float(np.nanquantile(s_ae, q))
    f_w, f_ae = s_w > u_w, s_ae > u_ae
    ambos = int((f_w & f_ae).sum())
    solo_w = int((f_w & ~f_ae).sum())
    solo_ae = int((~f_w & f_ae).sum())
    union = ambos + solo_w + solo_ae
    jac = ambos / union if union else np.nan

    print(f'\nCon el corte en el cuantil {q:.2f} de cada indicador '
          f'(u_W = {u_w:.3f}, u_AE = {u_ae:.3f}):')
    print(pd.DataFrame({
        'ventanas': [ambos, solo_w, solo_ae, union],
    }, index=['marcadas por los dos', 'solo por el indice W',
              'solo por el autoencoder', 'union']).to_string())
    print(f'\nJaccard (interseccion / union) = {jac:.3f}')
    print(f'Esperado si fueran independientes: '
          f'{(1-q)/(2-(1-q)):.3f}.  Si fueran el mismo indicador: 1,000.')

    if np.isfinite(jac):
        if jac > 0.70:
            print('\n  >> Mas del 70 % de solape. Los dos marcan casi las mismas')
            print('     ventanas: el indice W no esta aportando deteccion. Sigue')
            print('     sirviendo para ATRIBUIR -- dice en que sensor y en que')
            print('     direccion --, pero no lo presentes como un segundo')
            print('     detector, porque no lo es.')
        elif jac < 0.30:
            print('\n  >> Menos del 30 % de solape. Son complementarios, y eso')
            print('     es un resultado: la union detecta mas que cualquiera de')
            print('     los dos. Reportalo con las tres filas de arriba, que es')
            print('     la evidencia directa.')
        else:
            print('\n  >> Solape intermedio: se pisan en parte. Reporta las tres')
            print('     filas y no afirmes ni redundancia ni complementariedad')
            print('     sin mirar ADEMAS si las ventanas que cada uno marca solo')
            print('     caen en episodios distintos.')

    print('\nOJO con lo que esto NO prueba. "Marcada" aqui es estar sobre un')
    print('cuantil, no acertar: dos indicadores pueden solaparse poco y errar')
    print('los dos. El juicio de si sirven es del paso 05, por episodio.')

    return {'por_variable': t, 'jaccard': jac, 'ambos': ambos,
            'solo_w': solo_w, 'solo_ae': solo_ae,
            'umbral_w': u_w, 'umbral_ae': u_ae,
            'rho_agregado': spearman(s_w, s_ae)}


# =============================================================================
# Uso
# =============================================================================
#   from diagnosticos import *
#   df = pd.read_csv(ARCHIVO_LIMPIO, parse_dates=['timestamp']).set_index('timestamp')
#
#   distribucion_tramos(df)                      # antes de elegir la guarda
#   tiempo_establecimiento(df, VARIABLES)        # cuanto deberia valer tau
#   n_efectivo(X_tr[n_tr], VARIABLES)            # que mide el indice W
#   degeneracion_criticidad(ventanas)            # si la Fase B es viable
#
#   # y despues del paso 04, con el indice y el error de las mismas ventanas:
#   from indice_w import IndiceW
#   iw   = IndiceW(VARIABLES, w=W).ajustar(X_tr[n_tr])
#   idx  = iw.transformar(X_te)
#   base = np.load(f'{RUTA_BASE}/04_base_por_variable.npy')
#   complementariedad_w_vs_ae(idx, error_por_variable(ld_te) / base, VARIABLES)
