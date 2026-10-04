# -*- coding: utf-8 -*-
"""
=============================================================================
indice_w.py  -  el indice de Wasserstein por sensor y por ventana  (v4)
=============================================================================
Guardar en /content/drive/MyDrive/tesis_chancadores/ junto a reglas_engine.py.

Es el aporte propio que el profesor nombro segundo -- las bandas de severidad
por quintiles -- y el que dijo que estaba mal planteado. Lo que sigue resuelve
los cuatro defectos de una vez.

1. LA FORMULA QUE SE ESCRIBE ES LA QUE SE CALCULA
   La memoria anterior -- Ayala Penailillo, H., "Monitoreo de condicion
   industrial usando Deep Learning", profesor guia Mario Ramos Maldonado,
   Concepcion, 15-05-2026 -- escribe en su ecuacion 15 la forma general de
   Kantorovich:

       Wemd(mu, v) = min_{gamma in P} integral ||x - y|| gamma(x,y) dx dy

   y despues calcula, de hecho, una distancia unidimensional. Las dos son
   correctas, pero no son la misma expresion, y la que corre el codigo es la
   segunda.

   TRES PRECISIONES, verificadas contra el texto, para no pasarse al citarlo:
     - Su ecuacion 15 esta escrita con "min", no con "infimo", y el trabajo
       nunca escribe "W1" ni habla de orden p.
     - Que el calculo sea unidimensional NO lo declara el texto: se DEDUCE, de
       que la variable comparada es un escalar por ventana (el conteo de
       alertas). En la memoria hay que escribir "se deduce", no "declara".
     - Usa la sigla "Wemd" sin definir el subindice. La lectura evidente es
       Earth Mover's Distance, pero tampoco esta escrito.
     - La portada no nombra universidad en el texto (solo el logo) y el correo
       del autor es @ing.ucsc.cl. CONFIRMAR CON EL PROFESOR como referenciarlo
       antes de ponerlo en la bibliografia.

   En 1-D hay forma cerrada via funciones cuantil, y es la que hay que escribir
   porque es la que se calcula:

       W1(mu,nu) = integral_0^1 |F_mu^-1(q) - F_nu^-1(q)| dq

   En la memoria conviene poner las dos: la general como definicion, esta como
   lo efectivamente implementado. Eso contesta de entrada la observacion del
   profesor de que el calculo estaba mal planteado.

   Y UNA DIFERENCIA DE FONDO que no hay que esconder al citarlo. Ayala calcula
   la distancia entre la distribucion del CONTEO DE ALERTAS de un grupo y la
   del conjunto completo de ventanas -- una distancia entre GRUPOS DE VENTANAS,
   para ordenar soluciones de agrupamiento. Su propia notacion es
   max_{C in A} Wemd(C, X), con X el conjunto completo. Aca es por SENSOR y por
   VENTANA, contra la referencia de operacion normal. No es la misma cantidad y
   los numeros no se comparan: los suyos van de 8 a 215 "alertas por ventana",
   los de aca son adimensionales y centrados en 0.

2. LA DISTRIBUCION DE REFERENCIA QUEDA DEFINIDA
   Para cada sensor, la referencia son los valores de ese sensor agrupando
   todas las ventanas NORMALES del tramo de ENTRENAMIENTO de ESE chancador.
   - Solo entrenamiento: si incluye val o test, filtra informacion.
   - Solo normales: misma razon que para entrenar el autoencoder.
   - Por chancador: cada equipo tiene su propio regimen normal.

3. LA ESCALA ES COMPARABLE ENTRE SENSORES
   W1 tiene las unidades de la variable: grados para T5, por ciento para CM,
   mm/s para V1. Asi no se pueden ordenar entre si ni construir quintiles
   comunes. Se calcula sobre valores ESTANDARIZADOS con la media y desviacion
   del tramo de entrenamiento -- las mismas del normalizador del autoencoder
   -- y queda adimensional: "la distribucion se corrio el equivalente a 0,8
   desviaciones estandar".

4. SE CORRIGE POR TAMANO DE MUESTRA Y AUTOCORRELACION
   Con W = 60 se estima una distribucion con 60 puntos. La distancia empirica
   entre una muestra de 60 puntos y la referencia NO es cero aunque las dos
   vengan de la misma distribucion: tiene valor esperado positivo. Y a 10 s de
   muestreo esos 60 puntos estan fuertemente autocorrelacionados, asi que el
   tamano efectivo es bastante menor que 60.

   La nula se construye con las propias ventanas normales de entrenamiento:
   cada una ES un bloque contiguo de W muestras, asi que hereda la
   autocorrelacion real sin necesidad de remuestrear. El indice reportado es

       W_tilde = (W1 - mediana(nula)) / IQR(nula)

   Cerca de 0 significa "este sensor se ve como se ve normalmente".

QUE CAMBIA RESPECTO DE LA v3
-----------------------------------------------------------------------------
LAS BANDAS DE VENTANA SE CORRIGEN POR MULTIPLICIDAD.  <- el cambio de fondo

La v3 define los cortes POR SENSOR y despues agrega con el MAXIMO sobre los
14 sensores. Eso infla la banda de ventana de forma grosera: si los sensores
fueran independientes, la probabilidad de que una ventana NORMAL tenga al
menos un sensor sobre su propio P99 es

    1 - 0,99^14 = 13,1 %

o sea que la tabla declararia grave a una de cada ocho ventanas normales. Con
el P80 es peor todavia: 1 - 0,80^14 = 95,6 %. Estan correlacionados, asi que
los numeros reales son menores, pero el orden de magnitud no cambia y el
resultado seria ilegible.

La correccion: los cortes de la BANDA DE VENTANA salen de la distribucion nula
del AGREGADO -- el maximo sobre sensores calculado en las ventanas normales de
entrenamiento -- que es el estadistico que de verdad se usa para decidir. Los
cortes por sensor se conservan intactos para el PERFIL por sensor, donde la
multiplicidad no molesta porque se lee sensor por sensor.

El metodo ajustar() imprime las dos cifras para que el efecto quede medido y no
supuesto, y ese numero va a la memoria: es la diferencia entre un criterio
calibrado y uno que marca el 13 % de lo normal.
=============================================================================
"""

import numpy as np
import pandas as pd

# Sub-puntos de grilla por cada escalon de la funcion cuantil de la muestra.
# Error maximo contra scipy, medido con W = 60 y referencia de 50.000 puntos:
#   M_SUB =   1  ->  9,1e-3     (equivale a no sub-dividir)
#   M_SUB =   5  ->  1,1e-3
#   M_SUB =  20  ->  2,6e-4
#   M_SUB =  50  ->  1,5e-4     <- default, 0,002 veces el IQR de la nula
#   M_SUB = 100  ->  8,1e-5
# Mas alla de 100 el error se estanca en el ruido de la propia referencia.
M_SUB = 50
TAM_TROZO = 2000        # ventanas por trozo, para no reventar la memoria

PERCENTILES = [20, 40, 60, 80, 95, 99]
ETIQUETAS_BANDA = ['0_normal', '1_leve', '2_media', '3_grave']


def _grilla(w, m_sub=M_SUB):
    """Grilla de cuantiles alineada a los W escalones de la muestra."""
    return (np.arange(w * m_sub) + 0.5) / (w * m_sub)


def _cuantiles_referencia(valores, q):
    """F^-1 de la referencia, precomputada una sola vez por sensor."""
    v = np.asarray(valores, dtype=np.float64)
    v = v[np.isfinite(v)]
    return np.quantile(v, q)


def _w1_lote(muestras, cuant_ref_2d, tam_trozo=TAM_TROZO):
    """W1 de cada fila de `muestras` contra la referencia. Exacta en la muestra.

    muestras      : (n, W) -- n ventanas de un mismo sensor
    cuant_ref_2d  : (W, M_SUB) -- cuantiles de la referencia, reshape de la
                    grilla alineada: la fila k son los sub-puntos del escalon k

    La funcion cuantil de la muestra es la escalera exacta: dentro del escalon
    k vale s[k] en todos sus M_SUB sub-puntos. Por eso basta restar.
    """
    X = np.asarray(muestras, dtype=np.float64)
    n = len(X)
    out = np.empty(n, dtype=np.float64)
    for a in range(0, n, tam_trozo):
        b = min(a + tam_trozo, n)
        s = np.sort(X[a:b], axis=1)                       # (t, W)
        d = np.abs(s[:, :, None] - cuant_ref_2d[None])    # (t, W, M_SUB)
        out[a:b] = d.mean(axis=(1, 2))
    return out


def _w1(muestra, cuant_ref_2d):
    """W1 de UNA ventana y UN sensor. Igual que _w1_lote con una sola fila."""
    return float(_w1_lote(np.asarray(muestra)[None, :], cuant_ref_2d)[0])


class IndiceW:
    """Ajusta referencia y nula con las ventanas normales de entrenamiento."""

    def __init__(self, variables, w=60, m_sub=M_SUB):
        self.variables = list(variables)
        self.w = int(w)
        self.m_sub = int(m_sub)
        self.q = _grilla(self.w, self.m_sub)
        self.cuant_ref = {}         # sensor -> (W, M_SUB) cuantiles de referencia
        self.nula = {}              # sensor -> W1 de cada ventana normal de train
        self.med = {}
        self.iqr = {}
        self.cortes = {}            # sensor -> cortes del indice, POR SENSOR
        self.cortes_agregado = None # cortes de la banda de VENTANA
        self.nula_max = None        # el agregado sobre las normales de train

    # --------------------------------------------------------------- ajuste
    def ajustar(self, X_train_normal, verbose=True):
        """X_train_normal: (n_ventanas, W, n_vars) YA ESTANDARIZADO.

        Son las mismas ventanas con las que se entrena el autoencoder.
        """
        if X_train_normal.shape[1] != self.w:
            raise ValueError(f'IndiceW se construyo con w={self.w} pero las '
                             f'ventanas tienen {X_train_normal.shape[1]}. La '
                             f'grilla depende de W: pasale el W correcto.')
        if X_train_normal.shape[2] != len(self.variables):
            raise ValueError(f'{X_train_normal.shape[2]} variables en X pero '
                             f'{len(self.variables)} en el constructor.')
        if verbose:
            print(f'Ajustando indice W con {len(X_train_normal):,} ventanas '
                  f'normales de entrenamiento (W={self.w}, M_SUB={self.m_sub})')
            print('NOTA: estas ventanas son a la vez la referencia y la muestra')
            print('de la nula, asi que cada una se compara contra una')
            print('referencia que la incluye. Con miles el sesgo es')
            print('despreciable, pero hay que declararlo en la memoria.')

        for i, v in enumerate(self.variables):
            plano = X_train_normal[:, :, i].ravel()
            cr = _cuantiles_referencia(plano, self.q)
            self.cuant_ref[v] = cr.reshape(self.w, self.m_sub)

            # la nula: cada ventana normal contra la referencia. Cada ventana
            # es un bloque contiguo de W muestras, asi que hereda la
            # autocorrelacion real -- por eso no hace falta un bootstrap por
            # bloques aparte.
            d = _w1_lote(X_train_normal[:, :, i], self.cuant_ref[v])
            self.nula[v] = d
            self.med[v] = float(np.median(d))
            q75, q25 = np.percentile(d, [75, 25])
            iqr = float(q75 - q25)
            # "or 1e-12" protegia contra el cero exacto pero no contra un IQR
            # diminuto. Si la nula se concentra, el indice -- que divide por
            # este numero -- se dispara sin aviso.
            #
            # OJO CON EL DIAGNOSTICO, porque es facil equivocarse: esto NO le
            # pasa a las variables que n_efectivo() marca con n_ef < 1. Una
            # variable plana DENTRO de la ventana pero que deriva ENTRE
            # ventanas (una temperatura, el caso tipico) tiene W1 muy distinto
            # de ventana en ventana, o sea IQR GRANDE. El caso peligroso es el
            # otro: una variable casi constante dentro Y entre ventanas --
            # sensor trabado, o variable muy controlada -- donde todas las
            # ventanas se parecen y la nula se colapsa.
            if iqr <= 0 or iqr < 1e-3 * max(self.med[v], 1e-12):
                print(f'  AVISO {v}: IQR de la nula muy chico ({iqr:.2e}) '
                      f'frente a la mediana ({self.med[v]:.4f}). Todas las '
                      f'ventanas normales se parecen demasiado en esta '
                      f'variable: revisa si el sensor esta trabado antes de '
                      f'leer sus bandas. El indice se acota para que no '
                      f'explote, pero el aviso es el dato.')
                iqr = max(iqr, 1e-3 * max(self.med[v], 1e-12))
            self.iqr[v] = iqr

            corr = (d - self.med[v]) / self.iqr[v]
            # cortes POR SENSOR: solo para el perfil, donde se lee sensor por
            # sensor y la multiplicidad no molesta.
            self.cortes[v] = np.percentile(corr, PERCENTILES)

            if verbose:
                print(f'  {v:4s} mediana_nula={self.med[v]:.4f} '
                      f'IQR={self.iqr[v]:.4f} '
                      f'P80={self.cortes[v][3]:+.2f} '
                      f'P99={self.cortes[v][5]:+.2f}')

        # --- cortes de la BANDA DE VENTANA, corregidos por multiplicidad ----
        # Exigir el P99 a cada uno de 14 sensores por separado y despues tomar
        # el maximo NO es un criterio al 1 %: es 1 - 0,99^14 = 13,1 % si fueran
        # independientes. La banda de ventana se calibra sobre la nula del
        # estadistico que de verdad se usa -- el maximo.
        nula_idx = np.stack([(self.nula[v] - self.med[v]) / self.iqr[v]
                             for v in self.variables], axis=1)
        self.nula_max = nula_idx.max(axis=1)
        self.cortes_agregado = np.percentile(self.nula_max, PERCENTILES)

        if verbose:
            # el efecto, medido sobre las propias normales de entrenamiento
            niveles_sensor = np.stack(
                [np.digitize(nula_idx[:, i], self.cortes[v])
                 for i, v in enumerate(self.variables)], axis=1).max(axis=1)
            niveles_agg = np.digitize(self.nula_max, self.cortes_agregado)
            print('\n--- Correccion por multiplicidad (14 sensores) ---')
            print(f'{"banda":<12}{"sin corregir":>14}{"corregida":>12}')
            for k, etq in enumerate(ETIQUETAS_BANDA):
                lo = [-1, 3, 4, 5][k]
                hi = [3, 4, 5, 9][k]
                a = float(((niveles_sensor > lo) & (niveles_sensor <= hi)).mean())
                b = float(((niveles_agg > lo) & (niveles_agg <= hi)).mean())
                print(f'{etq:<12}{100*a:>13.1f}%{100*b:>11.1f}%')
            print('Son ventanas NORMALES de entrenamiento: la columna corregida')
            print('tiene que dar 80 / 15 / 4 / 1 por construccion. La otra es lo')
            print('que pasaba antes, y ese contraste va a la memoria.')
        return self

    # ------------------------------------------------------------ aplicacion
    def transformar(self, X, verbose=False):
        """X: (n, W, n_vars) estandarizado -> indice corregido (n, n_vars)."""
        if X.shape[1] != self.w:
            raise ValueError(f'ventanas de {X.shape[1]} contra un IndiceW '
                             f'ajustado en w={self.w}')
        out = np.empty((len(X), len(self.variables)), dtype=np.float32)
        for i, v in enumerate(self.variables):
            d = _w1_lote(X[:, :, i], self.cuant_ref[v])
            out[:, i] = (d - self.med[v]) / self.iqr[v]
            if verbose:
                print(f'  {v}: listo')
        return out

    # ----------------------------------------------------------------- bandas
    def bandas(self, indice):
        """Banda ordinal POR SENSOR. 0..6 segun los percentiles de su nula.

        Esto es para el PERFIL. Para la banda de la ventana usar
        banda_de_ventana(), que corrige por multiplicidad.
        """
        niveles = np.zeros(indice.shape, dtype=np.int8)
        for i, v in enumerate(self.variables):
            niveles[:, i] = np.digitize(indice[:, i], self.cortes[v])
        return niveles

    def banda_de_ventana(self, indice):
        """Nivel ordinal de la VENTANA, calibrado sobre la nula del agregado."""
        if self.cortes_agregado is None:
            raise RuntimeError('hay que llamar a ajustar() primero')
        return np.digitize(indice.max(axis=1), self.cortes_agregado)

    def resumen_por_ventana(self, indice, timestamps=None):
        """El perfil por sensor es la salida valiosa; el agregado es el MAXIMO.

        Maximo y no promedio: la gravedad la manda el sensor peor, no el
        conjunto. Y el sensor que alcanza ese maximo es la respuesta directa a
        lo que pidio el profesor -- "faltaria determinar cual es el que esta
        originando la condicion".

        Es el mismo criterio que resulto mejor en la escala experta: el 02c
        midio que criticidad_max le gana a peso_criticidad_total como
        predictor del error. Dos escalas distintas, misma conclusion -- el
        peor componente manda, no la suma.

        OJO: 'banda_w' sale de los cortes del AGREGADO, no del maximo de las
        bandas por sensor. Ver el encabezado del archivo.
        """
        niveles = self.bandas(indice)
        arg = indice.argmax(axis=1)
        nivel_ventana = self.banda_de_ventana(indice)
        df = pd.DataFrame({
            'w_max': indice.max(axis=1),
            'w_medio': indice.mean(axis=1),
            'sensor_origen': np.array(self.variables)[arg],
            'nivel_ventana': nivel_ventana,
            'nivel_sensor_max': niveles.max(axis=1),   # informativo, sin corregir
            'n_sensores_sobre_P80': (niveles >= 4).sum(axis=1),
        })
        df['banda_w'] = pd.cut(nivel_ventana, bins=[-1, 3, 4, 5, 9],
                               labels=ETIQUETAS_BANDA)
        for i, v in enumerate(self.variables):
            df[f'w_{v}'] = indice[:, i]
        if timestamps is not None:
            df.insert(0, 'timestamp', timestamps)
        return df


# =============================================================================
# VALIDACION -- correr una vez despues de pegar el archivo
# =============================================================================
def validar_contra_scipy(n_pruebas=300, w=60, m_sub=M_SUB, semilla=0,
                         verbose=True):
    """Mide la diferencia contra scipy.stats.wasserstein_distance.

    Lo que importa no es el error absoluto sino el error FRENTE AL IQR DE LA
    NULA, porque el indice se reporta dividido por ese IQR. El umbral de 0,05
    es arbitrario pero conservador: significa que el error numerico es menor
    que la veinteava parte de la unidad en que se reporta el indice.

    Esta funcion es la que destapo que el estimador de la v1 estaba mal.
    Dejarla y correrla: cuesta dos segundos.
    """
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(semilla)
    ref = rng.normal(0, 1, 50_000)
    cr = _cuantiles_referencia(ref, _grilla(w, m_sub)).reshape(w, m_sub)

    muestras = np.stack([rng.normal(rng.uniform(-0.5, 0.5),
                                    rng.uniform(0.7, 1.5), w)
                         for _ in range(n_pruebas)])
    mios = _w1_lote(muestras, cr)
    suyos = np.array([wasserstein_distance(m, ref) for m in muestras])
    err = np.abs(mios - suyos)

    nula = _w1_lote(rng.normal(0, 1, (2000, w)), cr)
    iqr_nula = float(np.subtract(*np.percentile(nula, [75, 25])))
    razon = float(err.max() / iqr_nula)

    if verbose:
        print(f'W = {w}, M_SUB = {m_sub}, grilla = {w*m_sub}, '
              f'{n_pruebas} muestras')
        print(f'  error medio vs scipy : {err.mean():.3e}')
        print(f'  error maximo         : {err.max():.3e}')
        print(f'  IQR de la nula       : {iqr_nula:.6f}')
        print(f'  error maximo / IQR   : {razon:.5f}')
        if razon < 0.05:
            print('  --> OK. El error numerico es despreciable frente a la')
            print('      escala en que se reporta el indice.')
        else:
            print('  --> ATENCION: subir M_SUB hasta que esta razon quede')
            print('      bajo 0,05. Si no baja al subirlo, el problema no es')
            print('      la grilla sino el estimador.')
    return {'err_medio': float(err.mean()), 'err_max': float(err.max()),
            'iqr_nula': iqr_nula, 'razon': razon, 'ok': razon < 0.05}


def validar_nula(w=60, m_sub=M_SUB, n=3000, rho=0.97, semilla=0, verbose=True):
    """Comprueba que la correccion por la nula hace lo que dice hacer.

    Genera ventanas de la MISMA distribucion que la referencia, pero con
    autocorrelacion parecida a la real (AR(1) con rho alto, que es lo que pasa
    a 10 s de muestreo). Sin la correccion, el W1 de esas ventanas da muy
    distinto de cero. Con la correccion, el indice tiene que quedar centrado
    en 0 y con IQR 1 por construccion.

    Si el indice de ventanas sanas no queda centrado en 0, la nula esta mal
    ajustada y las bandas van a marcar como anomalo lo que es ruido de
    muestreo.
    """
    rng = np.random.default_rng(semilla)

    def ar1(n_filas):
        e = rng.normal(0, 1, (n_filas, w))
        x = np.empty_like(e)
        x[:, 0] = e[:, 0]
        for k in range(1, w):
            x[:, k] = rho * x[:, k-1] + np.sqrt(1 - rho**2) * e[:, k]
        return x

    train = ar1(n)
    iw = IndiceW(['X'], w=w, m_sub=m_sub)
    iw.ajustar(train[:, :, None], verbose=False)
    idx = iw.transformar(ar1(n)[:, :, None])[:, 0]

    w1_crudo = iw.nula['X']
    if verbose:
        print(f'AR(1) con rho = {rho}, W = {w}')
        print(f'  W1 crudo de ventanas SANAS: mediana {np.median(w1_crudo):.4f}'
              f'  -- muy lejos de 0, que es justo el problema que la')
        print(f'     correccion resuelve')
        print(f'  indice corregido en ventanas sanas nuevas:')
        print(f'     mediana {np.median(idx):+.3f}  (deberia estar cerca de 0)')
        print(f'     IQR     {np.subtract(*np.percentile(idx, [75,25])):.3f}'
              f'  (deberia estar cerca de 1)')
        print(f'     P99     {np.percentile(idx, 99):+.3f}')
        frac = float((idx > iw.cortes['X'][5]).mean())
        print(f'  fraccion sobre el corte P99 de la nula: {100*frac:.2f} %'
              f'  (deberia ser cerca de 1 %)')
    return {'mediana': float(np.median(idx)),
            'iqr': float(np.subtract(*np.percentile(idx, [75, 25])))}


def validar_multiplicidad(n_vars=14, n=5000, semilla=0, verbose=True):
    """Muestra el efecto de la multiplicidad con sensores independientes.

    Es el caso extremo -- los sensores reales estan correlacionados, asi que
    el efecto sobre tus datos va a ser menor. Pero el orden de magnitud es
    este, y la cifra teorica 1 - 0,99^14 = 13,1 % es la que conviene citar en
    la memoria junto a la medida.
    """
    rng = np.random.default_rng(semilla)
    idx = rng.normal(0, 1, (n, n_vars))
    cortes_sensor = [np.percentile(idx[:, i], PERCENTILES)
                     for i in range(n_vars)]
    niv_sensor = np.stack([np.digitize(idx[:, i], cortes_sensor[i])
                           for i in range(n_vars)], axis=1).max(axis=1)
    cortes_agg = np.percentile(idx.max(axis=1), PERCENTILES)
    niv_agg = np.digitize(idx.max(axis=1), cortes_agg)
    if verbose:
        print(f'{n_vars} sensores INDEPENDIENTES, {n:,} ventanas normales')
        print(f'  grave (sobre P99) sin corregir : '
              f'{100*(niv_sensor > 5).mean():.1f} %  '
              f'(teorico 1-0,99^{n_vars} = {100*(1-0.99**n_vars):.1f} %)')
        print(f'  grave (sobre P99) corregido    : '
              f'{100*(niv_agg > 5).mean():.1f} %  (teorico 1,0 %)')
    return {'sin_corregir': float((niv_sensor > 5).mean()),
            'corregido': float((niv_agg > 5).mean())}


# =============================================================================
# Como se usa, dentro del paso 04 justo despues de construir las ventanas
# =============================================================================
#   from indice_w import (IndiceW, validar_contra_scipy, validar_nula,
#                         validar_multiplicidad)
#
#   validar_contra_scipy()        # una sola vez, dos segundos
#   validar_nula()                # una sola vez
#   validar_multiplicidad()       # una sola vez, para la cifra de la memoria
#
#   iw = IndiceW(VARIABLES, w=W).ajustar(X_tr[n_tr])   # normales de train
#   idx_te = iw.transformar(X_te)
#   res = iw.resumen_por_ventana(idx_te, ts_te)
#   res.to_csv(f'{RUTA_BASE}/indice_w_test.csv', index=False)
#
#   # el contraste que compensa la limitacion de la tabla de criticidad:
#   # escala empirica (bandas del indice W) contra escala experta
#   print(pd.crosstab(res.banda_w, ventanas_test.banda_criticidad))
#
# EL INDICE W Y EL ERROR DE RECONSTRUCCION: COMPLEMENTARIOS, PERO MEDILO.
# W se calcula sobre la distribucion de valores de la ventana, asi que es
# invariante a permutar el orden de las muestras: no ve dinamica, ve nivel y
# dispersion. El error del autoencoder si depende del orden: ve forma.
# PERO ese argumento se debilita en las variables lentas: si n_efectivo() da
# menos de 2, dentro de la ventana esa variable es casi un solo punto y W ahi
# mide NIVEL -- y el autoencoder, reconstruyendo una ventana casi plana en un
# nivel inusual, tambien responde al nivel. La complementariedad depende de la
# variable: medila con diagnosticos.complementariedad_w_vs_ae() y escribi el
# argumento matizado, con el numero propio al lado.
