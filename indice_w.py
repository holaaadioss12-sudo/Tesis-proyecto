# -*- coding: utf-8 -*-
"""
=============================================================================
indice_w.py  -  el indice de Wasserstein por sensor y por ventana  (v4)
=============================================================================
"""

import numpy as np
import pandas as pd

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
    """W1 de cada fila de `muestras` contra la referencia."""
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
    """W1 de UNA ventana y UN sensor."""
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
        """X_train_normal: (n_ventanas, W, n_vars) YA ESTANDARIZADO."""
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

            d = _w1_lote(X_train_normal[:, :, i], self.cuant_ref[v])
            self.nula[v] = d
            self.med[v] = float(np.median(d))
            q75, q25 = np.percentile(d, [75, 25])
            iqr = float(q75 - q25)
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
            self.cortes[v] = np.percentile(corr, PERCENTILES)

            if verbose:
                print(f'  {v:4s} mediana_nula={self.med[v]:.4f} '
                      f'IQR={self.iqr[v]:.4f} '
                      f'P80={self.cortes[v][3]:+.2f} '
                      f'P99={self.cortes[v][5]:+.2f}')

        # --- cortes de la BANDA DE VENTANA, corregidos por multiplicidad ----
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

    def bandas(self, indice):
        """Banda ordinal POR SENSOR."""
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
        """El perfil por sensor es la salida valiosa; el agregado es el MAXIMO."""
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
    """Mide la diferencia contra scipy.stats.wasserstein_distance."""
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
    """Comprueba que la correccion por la nula hace lo que dice hacer."""
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
    """Muestra el efecto de la multiplicidad con sensores independientes."""
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
