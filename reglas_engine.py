# -*- coding: utf-8 -*-
"""
=============================================================================
reglas_engine.py  -  el motor de las 32 reglas de alerta
=============================================================================

                  *** LEE ESTO ANTES DE CORRERLO ***

ESTE ARCHIVO ES UNA RECONSTRUCCION, NO TU ORIGINAL.

Tu reglas_engine.py se perdio. Lo que sigue esta armado juntando los umbrales
que aparecen desperdigados en tus otros scripts, que son los que si
sobrevivieron:

  - 00_diagnostico_previo.py, diccionario REGLAS_POR_VARIABLE y la lista
    DIFERENCIAS (T7-T1 y T8-T1 con sus tres y dos cortes)
  - 01_preprocesamiento.py, diccionario UMBRAL_REGLA_MAX
  - criticidad.py, FAMILIAS_ANIDADAS, donde cada familia lleva sus umbrales
    escritos en el comentario de la linea
  - 02_parches.py, VARIABLES_DE_REGLA (que variables mira cada regla) y
    REGLAS_ROLLING (A4 y A17 con ventana de 1 h y min_count 4 y 3)

Las 31 reglas evaluables cierran y los umbrales son consistentes entre las
cuatro fuentes. Pero CONSISTENTE NO ES IGUAL A CORRECTO: si tu original tenia
un operador >= donde aca hay >, una condicion extra en alguna regla, o una
definicion distinta de "sostenido", los conteos de activacion van a dar
distinto y TODO lo que viene despues cambia con ellos -- que ventana es
normal, con que entrena el autoencoder, y los numeros de la memoria.

QUE TENES QUE HACER, Y NO ES OPCIONAL:
  1. Corre resumen_umbrales() -- imprime las 32 filas completas.
  2. Compara esa tabla contra el DICCIONARIO DE ALERTAS de la planta, que es
     el documento del que salieron estas reglas.
  3. Corregi aca lo que no coincida, y recien ahi corre el 02.

Si el diccionario no lo tenes a mano, pediselo al profesor antes de seguir.
Es media hora de verificacion que evita rehacer el pipeline entero.

-----------------------------------------------------------------------------
EL CONTRATO, que es lo que el 02, el 04 y el 05 esperan:

    evaluar_reglas(df, verbose=False) -> DataFrame con el MISMO indice que df,
        una columna 0/1 por cada ID de regla evaluable, mas 'n_reglas_activadas'
        y 'estado'.
    REGLAS            -> lista de dicts, cada uno con al menos 'id'
    _tags_de_regla()  -> {id: [variables que mira]}, que es contra lo que el
                         02 verifica su VARIABLES_DE_REGLA

A23 NO sale como columna salvo que ZI266 este en el dataframe. Como ZI266 no
esta en el registro, quedan 31 de 32, y eso se declara en la memoria.

SOBRE LOS NaN: una comparacion con NaN da False, asi que una muestra con el
sensor caido no activa la regla. Eso es lo correcto AQUI, y es justamente el
motivo de que evaluabilidad() viva aparte en el 02: una regla que no se activa
porque el dato no estaba no es lo mismo que una que no se activa porque el
equipo estaba sano, y sin esa distincion la afirmacion "esta regla nunca se
activo" es indecidible.
=============================================================================
"""

import numpy as np
import pandas as pd

# Tolerancia para la comparacion con cero de A3.
#
# EN 0.0 REPRODUCE EL ORIGINAL EXACTO: |x| <= 0 es lo mismo que x == 0, que es
# lo que hacia tu motor. Se deja como constante y no como literal porque '== 0'
# sobre un flotante es una trampa conocida: un sensor que lee 1e-9 con la
# camara vacia no activa la regla nunca, y A3 es justamente la que necesita
# detectar "camara vacia". El bloque 2 del 00 mide las dos versiones ('== 0' y
# '<= 1e-6') sobre tus datos. Si dan lo mismo, dejalo en 0.0 y decilo en la
# memoria; si dan distinto, subilo a 1e-6 y declara el cambio.
TOL_CERO = 0.0

# Las cuatro vibraciones se agregan con el MAXIMO: la condicion del diccionario
# es "alguna de las cuatro supera el umbral", no el promedio.
VARS_VIBRACION = ['V1', 'V2', 'V3', 'V4']


# =============================================================================
# LAS 32 REGLAS
# =============================================================================
# tipo:
#   'umbral'         una variable contra un valor
#   'umbral_max'     el maximo de varias variables contra un valor (vibracion)
#   'diferencia'     a - b contra un valor
#   'y'              conjuncion de dos condiciones de umbral
#   'rolling_count'  la condicion base se cumple min_count veces dentro de una
#                    ventana temporal. Con ventana de tiempo (no de filas), los
#                    huecos reales del registro se respetan solos.
#
# La criticidad NO esta aca a proposito: vive en criticidad.py, que es la
# transcripcion de criticidadCH8.xlsx. Si estuviera en los dos lados se iban a
# desincronizar. verificar_contra_criticidad() comprueba que los dos coincidan.
REGLAS = [
    # --- corriente del motor (CM = AI) ---------------------------------------
    {'id': 'A',   'tipo': 'umbral', 'var': 'CM', 'op': '>', 'valor': 100,
     'desc': 'Sobrecarga de corriente'},
    {'id': 'A1',  'tipo': 'umbral', 'var': 'CM', 'op': '>', 'valor': 110,
     'desc': 'Sobrecarga alta'},
    {'id': 'A4',  'tipo': 'rolling_count', 'var': 'CM', 'op': '>', 'valor': 100,
     'ventana': '1h', 'min_count': 4,
     'desc': 'Sobrecarga permanente: >100 mas de 4 veces en 1 h'},

    # --- nivel de camara (PI = LIT254) ---------------------------------------
    # B ES COMPUESTA. Lo tuve mal en la reconstruccion: la puse como umbral
    # puro sobre PI porque es como la trata 00_diagnostico_previo.py. El motor
    # original la tiene como AND -- nivel alto Y corriente alta -- y tiene
    # sentido fisico: un nivel alto con el equipo descargado no es un atollo.
    # La tabla de 02_parches ('B': ['PI','CM']) era la correcta.
    {'id': 'B',   'tipo': 'y', 'var': 'PI', 'op': '>', 'valor': 60,
     'var2': 'CM', 'op2': '>', 'valor2': 100,
     'desc': 'Atollo / material excesivo o compactacion'},
    {'id': 'A3',  'tipo': 'y', 'var': 'PI', 'op': '==0', 'valor': 0,
     'var2': 'CM', 'op2': '>', 'valor2': 35,
     'desc': 'Camara vacia con el motor andando (contacto metal-metal)'},

    # --- filtro (PDF = PDIT263) ----------------------------------------------
    {'id': 'C',   'tipo': 'umbral', 'var': 'PDF', 'op': '>', 'valor': 400,
     'desc': 'Filtro saturado / obstruccion alta'},

    # --- presion de linea de lubricacion (PEL = PIT260) ----------------------
    {'id': 'D',   'tipo': 'umbral', 'var': 'PEL', 'op': '>', 'valor': 350,
     'desc': 'Presion de linea alta: bomba o linea obstruida'},
    {'id': 'A5',  'tipo': 'umbral', 'var': 'PEL', 'op': '>', 'valor': 450,
     'desc': 'Linea muy obstruida'},
    {'id': 'E',   'tipo': 'umbral', 'var': 'PEL', 'op': '<', 'valor': 150,
     'desc': 'Presion de linea baja: caudal insuficiente'},

    # --- temperatura T7 (TIT257), absoluta ------------------------------------
    {'id': 'F',   'tipo': 'umbral', 'var': 'T7', 'op': '>', 'valor': 69,
     'desc': 'T7 alta: riesgo de quema por lubricacion o sobrecarga'},
    {'id': 'A7',  'tipo': 'umbral', 'var': 'T7', 'op': '>', 'valor': 70,
     'desc': 'T7 muy alta'},
    {'id': 'A6',  'tipo': 'umbral', 'var': 'T7', 'op': '>', 'valor': 75,
     'desc': 'T7 critica: desgaste prematuro / lubricacion deficiente'},

    # --- T7 contra T1 (diferencia) --------------------------------------------
    {'id': 'A10', 'tipo': 'diferencia', 'a': 'T7', 'b': 'T1', 'op': '>',
     'valor': 30, 'desc': 'Salto termico T7-T1 sobre 30'},
    {'id': 'A9',  'tipo': 'diferencia', 'a': 'T7', 'b': 'T1', 'op': '>',
     'valor': 32, 'desc': 'Salto termico T7-T1 sobre 32'},
    {'id': 'A8',  'tipo': 'diferencia', 'a': 'T7', 'b': 'T1', 'op': '>',
     'valor': 56, 'desc': 'Salto termico T7-T1 sobre 56 (TRIP)',
     'trip': True},

    # --- T8 contra T1 (diferencia) --------------------------------------------
    {'id': 'I',   'tipo': 'diferencia', 'a': 'T8', 'b': 'T1', 'op': '>',
     'valor': 32, 'desc': 'Socket liner: friccion anormal'},
    {'id': 'A11', 'tipo': 'diferencia', 'a': 'T8', 'b': 'T1', 'op': '>',
     'valor': 56, 'desc': 'Socket liner: riesgo de gripado (TRIP)',
     'trip': True},

    # --- temperatura T9 (TIT259), contraeje ------------------------------------
    {'id': 'A13', 'tipo': 'umbral', 'var': 'T9', 'op': '>', 'valor': 53,
     'desc': 'Contraeje: desgaste o desalineacion'},
    {'id': 'A12', 'tipo': 'umbral', 'var': 'T9', 'op': '>', 'valor': 60,
     'desc': 'Contraeje: desgaste grave'},

    # --- temperatura T1 (TIT261), retorno de aceite ----------------------------
    {'id': 'K',   'tipo': 'umbral', 'var': 'T1', 'op': '>', 'valor': 50,
     'desc': 'Retorno de aceite caliente (alarma)'},
    {'id': 'A15', 'tipo': 'umbral', 'var': 'T1', 'op': '>', 'valor': 53,
     'desc': 'Retorno de aceite caliente (alta)'},
    {'id': 'A14', 'tipo': 'umbral', 'var': 'T1', 'op': '>', 'valor': 60,
     'desc': 'Retorno de aceite caliente (critica)'},
    {'id': 'A16', 'tipo': 'umbral', 'var': 'T1', 'op': '<', 'valor': 18,
     'desc': 'Retorno de aceite frio (baja)'},
    {'id': 'L',   'tipo': 'umbral', 'var': 'T1', 'op': '<', 'valor': 15,
     'desc': 'Retorno de aceite frio (TRIP)', 'trip': True},

    # --- temperaturas de aceite de entrada y tanque ----------------------------
    {'id': 'A21', 'tipo': 'umbral', 'var': 'T2', 'op': '>', 'valor': 50,
     'desc': 'Temperatura de aceite de entrada alta'},
    {'id': 'A22', 'tipo': 'umbral', 'var': 'T5', 'op': '>', 'valor': 50,
     'desc': 'Alta temperatura del tanque de aceite'},

    # --- vibracion (VIT255 A/B/C/D) -------------------------------------------
    {'id': 'O',   'tipo': 'umbral_max', 'vars': VARS_VIBRACION, 'op': '>',
     'valor': 16, 'desc': 'Vibracion: fijacion suelta / desbalance'},
    {'id': 'A20', 'tipo': 'umbral_max', 'vars': VARS_VIBRACION, 'op': '>',
     'valor': 20, 'desc': 'Vibracion elevada'},
    {'id': 'A19', 'tipo': 'umbral_max', 'vars': VARS_VIBRACION, 'op': '>',
     'valor': 34, 'desc': 'Vibracion alta'},
    {'id': 'A18', 'tipo': 'umbral_max', 'vars': VARS_VIBRACION, 'op': '>',
     'valor': 40, 'desc': 'Desbalance grave'},
    {'id': 'A17', 'tipo': 'rolling_count', 'vars': VARS_VIBRACION, 'op': '>',
     'valor': 20, 'ventana': '1h', 'min_count': 3,
     'desc': 'Vibracion sostenida: >20 mas de 3 veces en 1 h'},

    # --- setting (ZI266) -- NO EVALUABLE: el tag no esta en el registro --------
    {'id': 'A23', 'tipo': 'umbral', 'var': 'ZI266', 'op': '>', 'valor': 0,
     'desc': 'Setting fuera de rango',
     'nota': 'ZI266 no esta en el dataset. Queda inevaluable: 31 de 32.'},
]

IDS = [r['id'] for r in REGLAS]
REGLAS_TRIP = [r['id'] for r in REGLAS if r.get('trip')]
POR_ID = {r['id']: r for r in REGLAS}

# Las de conteo movil, con su ventana y su minimo. El 02 las necesita para
# decidir si una imputacion dentro de la hora compromete de verdad el veredicto.
ROLLING = {r['id']: {'ventana': r['ventana'], 'min_count': r['min_count']}
           for r in REGLAS if r['tipo'] == 'rolling_count'}


def _tags_de_regla():
    """{id: [variables que mira la regla]}. El 02 verifica su tabla contra esto."""
    out = {}
    for r in REGLAS:
        if r['tipo'] == 'diferencia':
            out[r['id']] = [r['a'], r['b']]
        elif r['tipo'] == 'y':
            out[r['id']] = [r['var'], r['var2']]
        elif 'vars' in r:
            out[r['id']] = list(r['vars'])
        else:
            out[r['id']] = [r['var']]
    return out


def _serie(df, r):
    """La magnitud que la regla compara, o None si falta alguna columna."""
    if r['tipo'] == 'diferencia':
        if r['a'] not in df.columns or r['b'] not in df.columns:
            return None
        return df[r['a']].to_numpy(dtype=float) - df[r['b']].to_numpy(dtype=float)
    if 'vars' in r:
        presentes = [v for v in r['vars'] if v in df.columns]
        if not presentes:
            return None
        # maximo sobre las cuatro: la condicion es "alguna supera el umbral"
        return np.nanmax(np.stack([df[v].to_numpy(dtype=float)
                                   for v in presentes]), axis=0)
    if r['var'] not in df.columns:
        return None
    return df[r['var']].to_numpy(dtype=float)


def condicion_base(df, regla_id):
    """La condicion por MUESTRA de una regla de conteo movil, antes de contar.

    A4 es "CM > 100 al menos 4 veces en 1 h": esto devuelve el "CM > 100" de
    cada muestra. El 02 lo usa para medir cuanto margen tiene el conteo antes
    de que una imputacion pueda cambiar el veredicto.
    """
    r = POR_ID.get(regla_id)
    if r is None or r['tipo'] != 'rolling_count':
        raise ValueError(f'{regla_id} no es una regla de conteo movil')
    x = _serie(df, r)
    if x is None:
        return None
    return np.nan_to_num(_comparar(x, r['op'], r['valor']), nan=False)


def _comparar(x, op, valor):
    with np.errstate(invalid='ignore'):
        if op == '>':
            return x > valor
        if op == '<':
            return x < valor
        if op == '>=':
            return x >= valor
        if op == '<=':
            return x <= valor
        if op == '==0':
            return np.abs(x) <= TOL_CERO
    raise ValueError(f'operador no soportado: {op}')


def evaluar_reglas(df, verbose=False):
    """Evalua las reglas muestra a muestra.

    df : DataFrame con indice datetime y las columnas canonicas (CM, PI, PDF,
         PEL, T1, T2, T5, T7, T8, T9, V1..V4). Columnas de mas se ignoran.

    Devuelve un DataFrame con el mismo indice: una columna 0/1 por regla
    evaluable, 'n_reglas_activadas' y 'estado'.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError('evaluar_reglas necesita indice datetime: las reglas '
                        'A4 y A17 usan una ventana movil de 1 hora.')

    cols, omitidas = {}, []
    for r in REGLAS:
        x = _serie(df, r)
        if x is None:
            omitidas.append(r['id'])
            continue

        if r['tipo'] == 'rolling_count':
            # La condicion base se tiene que cumplir min_count veces dentro de
            # la ventana. rolling con ventana TEMPORAL respeta los huecos del
            # registro: si el equipo estuvo parado, esas muestras no existen y
            # no entran en el conteo.
            # EL CONTEO ES ESTRICTAMENTE MAYOR QUE min_count, no mayor o
            # igual. Lo tuve mal: con >= , A4 disparaba con 4 ocurrencias y el
            # motor original pide 5. Un corrimiento de uno en el umbral de la
            # regla que mas ventanas consume de todas.
            #
            # Y sin min_periods: la ventana movil exige una hora completa de
            # historia, asi que A4 y A17 no pueden dispararse en los primeros
            # 60 min de cada tramo. Es lo que hace el motor original y se
            # respeta, pero conviene saberlo: con tramos de ~1,7 h de promedio,
            # eso es una parte apreciable del registro donde estas dos reglas
            # estan apagadas por construccion, no por el estado del equipo.
            base = _comparar(x, r['op'], r['valor'])
            s = pd.Series(base.astype(np.float32), index=df.index)
            act = s.rolling(r['ventana']).sum().to_numpy() > r['min_count']
        elif r['tipo'] == 'y':
            y = _serie(df, {'tipo': 'umbral', 'var': r['var2']})
            if y is None:
                omitidas.append(r['id'])
                continue
            act = _comparar(x, r['op'], r['valor']) & _comparar(y, r['op2'],
                                                               r['valor2'])
        else:
            act = _comparar(x, r['op'], r['valor'])

        cols[r['id']] = np.nan_to_num(act, nan=False).astype(np.int8)

    res = pd.DataFrame(cols, index=df.index)
    res['n_reglas_activadas'] = res.sum(axis=1).astype(np.int16)
    res['estado'] = np.where(res.n_reglas_activadas > 0, 'alarma', 'normal')

    if verbose:
        print(f'Reglas evaluadas: {len(cols)} de {len(REGLAS)}')
        if omitidas:
            print(f'  omitidas por falta de tag: {omitidas}')
        act_tot = int((res.n_reglas_activadas > 0).sum())
        print(f'  muestras en alarma: {act_tot:,} de {len(res):,} '
              f'({100*act_tot/max(len(res),1):.2f} %)')
    return res


# =============================================================================
# VERIFICACION -- correr las dos ANTES de usar el motor
# =============================================================================
def resumen_umbrales():
    """LA TABLA QUE HAY QUE COTEJAR CONTRA EL DICCIONARIO DE ALERTAS.

    Imprimila, ponela al lado del documento de la planta, y revisa fila por
    fila. Es la unica forma de saber si esta reconstruccion es tu motor.
    """
    filas = []
    for r in REGLAS:
        if r['tipo'] == 'diferencia':
            cond = f"{r['a']} - {r['b']} {r['op']} {r['valor']}"
        elif r['tipo'] == 'umbral_max':
            cond = f"max({','.join(r['vars'])}) {r['op']} {r['valor']}"
        elif r['tipo'] == 'rolling_count':
            donde = (f"max({','.join(r['vars'])})" if 'vars' in r else r['var'])
            cond = (f"{donde} {r['op']} {r['valor']}, > {r['min_count']} "
                    f"veces en {r['ventana']}")
        elif r['tipo'] == 'y':
            if r['op'] == '==0':
                _izq = (f"{r['var']} == 0" if TOL_CERO == 0
                        else f"|{r['var']}| <= {TOL_CERO:g}")
            else:
                _izq = f"{r['var']} {r['op']} {r['valor']}"
            cond = f"{_izq}  Y  {r['var2']} {r['op2']} {r['valor2']}"
        else:
            cond = f"{r['var']} {r['op']} {r['valor']}"
        filas.append({'id': r['id'], 'tipo': r['tipo'], 'condicion': cond,
                      'trip': bool(r.get('trip')),
                      'descripcion': r['desc']})
    t = pd.DataFrame(filas)
    print('LAS 32 REGLAS, COMO LAS EVALUA ESTE MOTOR')
    print('Cotejar contra el diccionario de alertas de la planta.\n')
    print(t.to_string(index=False))
    print(f'\nTRIP: {REGLAS_TRIP}')
    print('Nota: A23 queda fuera del calculo porque ZI266 no esta en el')
    print('registro. Son 31 de 32, y eso se declara en la memoria.')
    return t


def diagnosticar_entorno(verbose=True):
    """Que archivo esta usando Python para cada modulo del proyecto.

    CORRELA ESTO ANTES DE CULPAR AL CODIGO. En Colab el kernel sobrevive a que
    reemplaces los archivos en Drive: Python cachea los modulos por nombre, y
    si importaste algo antes de subir la version nueva, sigue usando la vieja
    aunque el archivo ya no exista. El sintoma es un error que no tiene nada
    que ver con la causa.

    Si alguna fila sale en rojo, la solucion no es tocar el codigo:
    Entorno de ejecucion -> Reiniciar sesion. El contador de celdas vuelve a
    [1], y ESA es la senal de que se reinicio de verdad.
    """
    import sys
    import os
    aqui = os.path.dirname(os.path.abspath(__file__))
    esperados = ['reglas_engine', 'criticidad', 'indice_w', 'diagnosticos']
    filas, malas = [], []
    for nombre in esperados:
        ruta_ok = os.path.join(aqui, f'{nombre}.py')
        existe = os.path.exists(ruta_ok)
        mod = sys.modules.get(nombre)
        cargado = getattr(mod, '__file__', None) if mod is not None else None
        coincide = (cargado is not None
                    and os.path.abspath(cargado) == os.path.abspath(ruta_ok))
        estado = ('no importado todavia' if mod is None
                  else 'OK' if coincide else '>>> APUNTA A OTRO ARCHIVO <<<')
        if mod is not None and not coincide:
            malas.append(nombre)
        if not existe:
            estado = '>>> NO ESTA EN LA CARPETA <<<'
            malas.append(nombre)
        filas.append({'modulo': nombre, 'archivo_en_carpeta': existe,
                      'cargado_desde': cargado or '-', 'estado': estado})
    t = pd.DataFrame(filas)
    if verbose:
        print(f'Carpeta del proyecto: {aqui}\n')
        print(t.to_string(index=False))
        if malas:
            print('\n' + '!' * 70)
            print(f'Modulos en mal estado: {sorted(set(malas))}')
            print('NO toques el codigo: el kernel tiene basura cacheada.')
            print('  Entorno de ejecucion -> Reiniciar sesion')
            print('  y volve a correr desde la primera celda.')
            print('Si el contador de celdas no vuelve a [1], no se reinicio.')
            print('!' * 70)
        else:
            print('\nTodo apunta a donde tiene que apuntar.')
    return t


def verificar_contra_criticidad(verbose=True):
    """Los IDs del motor y los de criticidad.py tienen que ser los mismos.

    Si no coinciden, una de las dos tablas quedo desactualizada y las ventanas
    van a recibir criticidad de reglas que no existen, o al reves.
    """
    # SE CARGA POR RUTA, A PROPOSITO, SIN PASAR POR sys.modules.
    #
    # Un 'import criticidad' normal confia en el cache de modulos del kernel, y
    # ese cache se corrompe con una facilidad que sorprende: basta haber
    # importado algo con ese nombre antes en la misma sesion para que Python no
    # vuelva a mirar el archivo nunca mas. Ya paso una vez en este proyecto, y
    # de la forma mas confusa posible: el nombre 'criticidad' habia quedado
    # apuntando a 02_parches.py, y el error era "cannot import name
    # 'CRITICIDAD' from '02_parches'" -- un mensaje que no se parece en nada a
    # su causa.
    #
    # Un validador no puede depender de eso. Carga el archivo que esta al lado
    # de este, lo ejecuta aparte, y punto. Lo que diga sys.modules no importa.
    import importlib.util
    import os
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'criticidad.py')
    if not os.path.exists(ruta):
        raise SystemExit(f'No existe {ruta}. criticidad.py tiene que estar en '
                         f'la misma carpeta que reglas_engine.py.')
    _spec = importlib.util.spec_from_file_location('_criticidad_verif', ruta)
    criticidad = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(criticidad)

    NECESARIOS = ['CRITICIDAD', 'FAMILIAS_ANIDADAS', 'REGLAS_TRIP',
                  'test_equivalencia', 'COLAPSAR_B_CON_A']
    faltan = [n for n in NECESARIOS if not hasattr(criticidad, n)]
    if faltan:
        print('\n' + '!' * 70)
        print(f'El criticidad.py que esta cargado NO es la v2: le faltan '
              f'{faltan}.')
        print(f'Archivo que Python esta usando:\n  {criticidad.__file__}')
        print('\nCasi seguro es tu criticidad.py original (el que tenia')
        print('diagnostico_solapamiento), no el que te pase. Dos cosas:')
        print('  1. Confirma que el criticidad.py de Drive sea la v2 --')
        print('     buscale COLAPSAR_B_CON_A adentro.')
        print('  2. Reinicia el entorno de ejecucion de Colab. Python cachea')
        print('     los modulos: si lo importaste antes de reemplazar el')
        print('     archivo, el viejo se queda en memoria y el nuevo no se')
        print('     lee nunca. Entorno de ejecucion -> Reiniciar sesion.')
        print('!' * 70)
        raise SystemExit('criticidad.py desactualizado, ver arriba.')

    CRITICIDAD = criticidad.CRITICIDAD
    FAMILIAS_ANIDADAS = criticidad.FAMILIAS_ANIDADAS
    TRIP_C = criticidad.REGLAS_TRIP
    if verbose:
        print(f'criticidad.py cargado desde:\n  {criticidad.__file__}')
    motor, tabla = set(IDS), set(CRITICIDAD)
    solo_motor, solo_tabla = sorted(motor - tabla), sorted(tabla - motor)
    ok = not solo_motor and not solo_tabla

    # y los umbrales de las familias anidadas tienen que estar en orden
    por_id = {r['id']: r for r in REGLAS}
    desorden = []
    for fam, miembros in FAMILIAS_ANIDADAS.items():
        vals = [por_id[m].get('valor') for m in miembros if m in por_id]
        ops = {por_id[m].get('op') for m in miembros if m in por_id}
        if len(vals) != len(miembros) or None in vals:
            continue
        creciente = all(a < b for a, b in zip(vals, vals[1:]))
        decreciente = all(a > b for a, b in zip(vals, vals[1:]))
        if ops <= {'>'} and not creciente:
            desorden.append((fam, miembros, vals, 'deberia ser creciente'))
        if ops <= {'<'} and not decreciente:
            desorden.append((fam, miembros, vals, 'deberia ser decreciente'))

    if verbose:
        print(f'IDs en el motor: {len(motor)} | en criticidad.py: {len(tabla)}')
        if solo_motor:
            print(f'  SOLO en el motor   : {solo_motor}')
        if solo_tabla:
            print(f'  SOLO en la tabla   : {solo_tabla}')
        if ok:
            print('  --> coinciden las 32.')
        print(f'\nTRIP en el motor: {sorted(REGLAS_TRIP)} | '
              f'en criticidad.py: {sorted(TRIP_C)}')
        if sorted(REGLAS_TRIP) != sorted(TRIP_C):
            print('  --> NO coinciden. Arreglalo antes de construir R2 en el 05.')

        print('\nOrden de umbrales dentro de cada familia anidada:')
        if desorden:
            print('  >> HAY FAMILIAS DESORDENADAS:')
            for fam, m, v, msg in desorden:
                print(f'     {fam}: {list(zip(m, v))} -- {msg}')
            print('  El colapso de criticidad.py supone que la regla mas grave')
            print('  implica a las anteriores. Con los umbrales desordenados')
            print('  ese supuesto es falso y peso_agregado da mal.')
        else:
            print('  Todas crecen (o decrecen) como corresponde.')
    return {'ok': ok and not desorden, 'solo_motor': solo_motor,
            'solo_tabla': solo_tabla, 'familias_desordenadas': desorden}


def test_motor(verbose=True):
    """Casos sinteticos de activacion conocida. Milisegundos, assert duros."""
    idx = pd.date_range('2025-01-01', periods=6, freq='10s')
    df = pd.DataFrame({
        'CM':  [85, 105, 115, 85, 120, 85],
        'PI':  [30, 30, 30, 0.0, 70, 30],
        'PDF': [200] * 6,
        'PEL': [250, 250, 250, 250, 100, 500],
        'T7':  [60, 60, 60, 60, 60, 60],
        'T8':  [42] * 6,
        'T9':  [47] * 6,
        'T1':  [43, 43, 61, 43, 43, 43],
        'T2':  [37] * 6,
        'T5':  [43] * 6,
        'V1':  [8, 8, 8, 8, 8, 45],
        'V2':  [8] * 6, 'V3': [8] * 6, 'V4': [8] * 6,
    }, index=idx)
    r = evaluar_reglas(df, verbose=False)

    # fila 1: CM = 105 -> A (>100) si, A1 (>110) no
    assert r.A.iloc[1] == 1 and r.A1.iloc[1] == 0, 'CM=105'
    # fila 2: CM = 115 -> A y A1; T1 = 61 -> K, A15 y A14 a la vez
    assert r.A.iloc[2] == 1 and r.A1.iloc[2] == 1, 'CM=115'
    assert r.K.iloc[2] == r.A15.iloc[2] == r.A14.iloc[2] == 1, 'T1=61 anidada'
    # fila 3: PI = 0 pero CM = 85 (< 35 no, pero A3 pide CM>35) -> A3 SI
    assert r.A3.iloc[3] == 1, 'A3 camara vacia con motor andando'
    # fila 4: PEL = 100 -> E (<150). B pide PI > 60 Y CM > 100: con PI = 70 y
    # CM = 120 dispara. Con CM = 85 NO disparaba, y esa es exactamente la
    # diferencia que la reconstruccion tenia mal.
    assert r.E.iloc[4] == 1 and r.B.iloc[4] == 1, 'PEL bajo, y B compuesta'
    assert r.B.iloc[5] == 0, 'B no puede dispararse con PI = 30'
    # fila 5: V1 = 45 -> O, A20, A19 y A18 a la vez; PEL = 500 -> D y A5
    assert r.O.iloc[5] == r.A20.iloc[5] == r.A19.iloc[5] == r.A18.iloc[5] == 1, \
        'vibracion 45 anidada'
    assert r.D.iloc[5] == 1 and r.A5.iloc[5] == 1, 'PEL=500'
    # fila 0: todo normal
    assert r.n_reglas_activadas.iloc[0] == 0, 'fila 0 deberia estar limpia'
    # A23 no sale: ZI266 no esta
    assert 'A23' not in r.columns, 'A23 no deberia evaluarse sin ZI266'
    assert len([c for c in r.columns
                if c not in ('n_reglas_activadas', 'estado')]) == 31, \
        'tienen que quedar 31 reglas evaluables'

    if verbose:
        print('test_motor -- OK')
        print('  31 reglas evaluables, A23 fuera por falta de ZI266')
        print('  anidamiento verificado: T1=61 activa K, A15 y A14 a la vez,')
        print('  y vibracion=45 activa O, A20, A19 y A18. Eso es lo que hace')
        print('  que peso_agregado tenga que colapsar familias.')
    return True


if __name__ == '__main__':
    test_motor()
    print()
    verificar_contra_criticidad()
    print()
    resumen_umbrales()
