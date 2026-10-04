# -*- coding: utf-8 -*-
"""
=============================================================================
criticidad.py  (v2)  -  la tabla experta, transcrita de criticidadCH8.xlsx
=============================================================================
Guardar en /content/drive/MyDrive/tesis_chancadores/ junto a reglas_engine.py.

El paso 02 tiene hoy CRITICIDAD_REGLAS = {} con el comentario de que la tabla
"no existe todavia (pendiente con el profesor)". Si existe: es criticidadCH8.xlsx,
en la misma carpeta de Drive que los CSV. Trae criticidad, componente y modo de
falla para los 32 IDs.

OJO CON LA ESCALA: va al reves de lo intuitivo.
    criticidad 1 = MAS grave     peso 3
    criticidad 2 = media         peso 2
    criticidad 3 = MENOS grave   peso 1
El peso es 4 - criticidad, para poder agregar.

Y OJO CON EL ANIDAMIENTO: las reglas no son excluyentes. Una muestra de T1 a
61 grados activa K (>50), A15 (>53) y A14 (>60) a la vez. La criticidad de la
ventana es la de su regla MAS grave -- el MINIMO numero -- nunca la suma: si se
suman los pesos de reglas anidadas, una sola condicion fisica se cuenta tres
veces y la ventana aparece artificialmente grave.

QUE CAMBIA EN LA v2
-----------------------------------------------------------------------------
1. test_equivalencia(). El colapso por familias se verifica con casos
   sinteticos de numero conocido, no de palabra. Tres assert duros. Son
   milisegundos y los numeros que imprime van a la memoria.

2. COLAPSAR_B_CON_A, y verificar_anidamiento() para decidirlo CON DATOS.
   FAMILIAS_ANIDADAS esta escrito a mano leyendo los umbrales del diccionario
   de alertas. Eso funciona para las familias obvias -- tres cortes sobre la
   misma T1 -- pero A y B describen el mismo evento fisico visto por dos tags
   ("Bloqueo mecanico / Sobrecarga" y "Atollo / Material excesivo"), y si en
   los datos B nunca se activa sin A, tratarlas como independientes duplica el
   peso de un solo atollo. NO lo doy por cierto: el flag queda en False y
   verificar_anidamiento() mide la co-ocurrencia real sobre tus ventanas y dice
   si corresponde prenderlo. Decision explicita y medida, no implicita.

3. REGLAS_COMPUESTAS. La tabla del profesor tiene 35 filas: 32 reglas base mas
   tres combinaciones (A+A6, A15+E, A+O). El motor no las evalua. Quedan aqui
   declaradas para que el trabajo futuro sepa exactamente que falta, y
   cobertura_compuestas() te dice cuantas ventanas las cumplirian hoy.
=============================================================================
"""

import numpy as np
import pandas as pd

# --- criticidad por ID de regla ---------------------------------------------
CRITICIDAD = {
    'A': 2, 'B': 1, 'C': 1, 'D': 2, 'E': 3, 'F': 2,
    'I': 2, 'K': 3, 'L': 1, 'O': 3,
    'A1': 1, 'A3': 2, 'A4': 2, 'A5': 1, 'A6': 1, 'A7': 1,
    'A8': 1, 'A9': 2, 'A10': 3, 'A11': 1, 'A12': 1, 'A13': 2,
    'A14': 1, 'A15': 2, 'A16': 2, 'A17': 1, 'A18': 1, 'A19': 1,
    'A20': 2, 'A21': 2, 'A22': 3, 'A23': 2,   # A23 no se evalua: falta ZI266
}
PESO = {r: 4 - c for r, c in CRITICIDAD.items()}

# --- componente y modo de falla, para la Fase B ------------------------------
COMPONENTE = {
    'A': 'motor', 'A1': 'motor', 'A4': 'motor',
    'B': 'alimentacion', 'A3': 'alimentacion',
    'C': 'filtro',
    'D': 'lubricacion', 'E': 'lubricacion', 'A5': 'lubricacion',
    'K': 'lubricacion', 'A14': 'lubricacion', 'A15': 'lubricacion',
    'A16': 'lubricacion', 'L': 'lubricacion',
    'A8': 'lubricacion', 'A9': 'lubricacion', 'A10': 'lubricacion',
    'A21': 'lubricacion', 'A22': 'tanque_aceite',
    'F': 'eje_principal', 'A6': 'eje_principal', 'A7': 'eje_principal',
    'I': 'socket_liner', 'A11': 'socket_liner',
    'A12': 'contraeje', 'A13': 'contraeje',
    'O': 'estructura', 'A17': 'estructura', 'A18': 'estructura',
    'A19': 'estructura', 'A20': 'estructura',
    'A23': 'setting',
}

FALLA = {
    'A': 'Bloqueo mecanico / Sobrecarga / Material muy duro',
    'A1': 'Sobrecarga alta',
    'A4': 'Sobrecarga permanente',
    'B': 'Atollo / Material excesivo o compactacion',
    'A3': 'Material atrapado o friccion en sistemas mecanicos',
    'C': 'Filtro saturado / obstruccion alta',
    'D': 'Falla de bomba / linea obstruida',
    'E': 'Falla de bomba / linea obstruida',
    'A5': 'Linea muy obstruida',
    'F': 'Riesgo de quema por lubricacion inadecuada o sobrecarga',
    'A7': 'Riesgo de quema por lubricacion inadecuada o sobrecarga',
    'A6': 'Desgaste prematuro / Lubricacion deficiente',
    'I': 'Lubricacion deficiente / friccion anormal en socket liner',
    'A11': 'Sobrecalentamiento severo del socket liner / riesgo de gripado',
    'A8': 'Problema termico / posible contaminacion',
    'A9': 'Problema termico / posible contaminacion',
    'A10': 'Problema termico / posible contaminacion',
    'A12': 'Desgaste grave',
    'A13': 'Desgaste o desalineacion de componentes',
    'K': 'Calentamiento excesivo / perdida de eficiencia (alarma)',
    'A15': 'Calentamiento excesivo / perdida de eficiencia (alta)',
    'A14': 'Calentamiento excesivo / perdida de eficiencia (critica)',
    'A16': 'Calentamiento excesivo / perdida de eficiencia (baja)',
    'L': 'Calentamiento excesivo / perdida de eficiencia (trip)',
    'A21': 'Temperatura de aceite de entrada alta',
    'A22': 'Alta temperatura del tanque de aceite',
    'O': 'Fijacion suelta / Desbalance estructural / Eje descentrado',
    'A20': 'Fijacion suelta / Desbalance estructural / Eje descentrado',
    'A19': 'Fijacion suelta / Desbalance estructural / Eje descentrado',
    'A18': 'Fijacion suelta / Desbalance estructural / Eje descentrado',
    'A17': 'Desbalance grave',
    'A23': 'Setting fuera de rango',
}

# --- familias anidadas -------------------------------------------------------
# Dentro de cada familia las reglas se disparan en cascada: una muestra que
# cumple la mas estricta cumple tambien todas las anteriores. Para agregar por
# peso hay que colapsar cada familia a su regla mas grave primero.
#
# Estas nueve salen de leer los umbrales: son cortes sucesivos sobre LA MISMA
# variable, asi que el anidamiento es matematico y no hace falta medirlo.
FAMILIAS_ANIDADAS = {
    'T7_absoluta':   ['F', 'A7', 'A6'],            # >69, >70, >75
    'T7_diferencia': ['A10', 'A9', 'A8'],          # >30, >32, >56 sobre T1
    'T8_diferencia': ['I', 'A11'],                 # >32, >56 sobre T1
    'T9_absoluta':   ['A13', 'A12'],               # >53, >60
    'T1_alta':       ['K', 'A15', 'A14'],          # >50, >53, >60
    'T1_baja':       ['A16', 'L'],                 # <18, <15
    'vibracion':     ['O', 'A20', 'A19', 'A18'],   # >16, >20, >34, >40
    'CM':            ['A', 'A1'],                  # >100, >110
    'PEL_alta':      ['D', 'A5'],                  # >350, >450
}

# --- la decima familia, la que NO es matematica -------------------------------
# A  = "Bloqueo mecanico / Sobrecarga / Material muy duro"   (criticidad 2)
# B  = "Atollo / Material excesivo o compactacion"           (criticidad 1)
#
# Son dos tags distintos describiendo el mismo evento de proceso: el chancador
# atorado. Si en los datos B practicamente no aparece sin A, sumar los dos
# pesos cuenta un solo atollo dos veces y la ventana queda mas grave de lo que
# es. Pero eso es una AFIRMACION SOBRE LOS DATOS, no sobre los umbrales, asi
# que no se decide leyendo el diccionario de alertas: se mide.
#
# Queda en False a proposito. Corre verificar_anidamiento() sobre tus ventanas,
# mira la columna P(B | A) y su reciproca, y prendelo solo si la medicion lo
# sostiene. Si lo prendes, el numero medido va a la memoria junto al cambio.
#
# Nota: esto afecta UNICAMENTE a peso_agregado(). criticidad_min es un minimo,
# y un minimo no se altera por contar dos veces -- la banda de severidad que
# usa el autoencoder no cambia en ningun caso.
COLAPSAR_B_CON_A = False
FAMILIA_B_A = ['B', 'A', 'A1']     # orden irrelevante: colapsa por criticidad


def familias_activas():
    """Las familias que se van a usar, segun el flag."""
    fam = dict(FAMILIAS_ANIDADAS)
    if COLAPSAR_B_CON_A:
        fam.pop('CM', None)
        fam['atollo_sobrecarga'] = list(FAMILIA_B_A)
    return fam


# --- reglas marcadas TRIP en el diccionario de alertas -----------------------
# Sirven como referencia R2 para medir anticipacion: son la condicion mas grave
# que la planta reconoce, y las tres tienen criticidad 1.
REGLAS_TRIP = ['L', 'A8', 'A11']

# --- las tres filas de la tabla que el motor no evalua -----------------------
# criticidadCH8.xlsx tiene 35 filas: 32 reglas base + 3 combinaciones. El motor
# de reglas evalua condiciones de una variable, asi que estas tres quedan
# fuera. Declaradas aqui para que el capitulo de trabajo futuro sea concreto y
# para poder medir cuanto se esta dejando sobre la mesa.
REGLAS_COMPUESTAS = {
    'A+A6':  {'miembros': ['A', 'A6'],  'criticidad': 1,
              'componente': 'eje_principal',
              'falla': 'Sobrecarga con lubricacion deficiente'},
    'A15+E': {'miembros': ['A15', 'E'], 'criticidad': 1,
              'componente': 'lubricacion',
              'falla': 'Temperatura alta con caudal insuficiente'},
    'A+O':   {'miembros': ['A', 'O'],   'criticidad': 1,
              'componente': 'estructura',
              'falla': 'Sobrecarga con desbalance estructural'},
}

BANDA = {1: '3_grave', 2: '2_media', 3: '1_leve', 4: '0_normal'}


def criticidad_de_ventana(fila, ids_reglas):
    """La criticidad de la regla mas grave activa. 4 = ninguna activa."""
    act = [CRITICIDAD[r] for r in ids_reglas
           if r in CRITICIDAD and fila[r] > 0]
    return min(act) if act else 4


def peso_agregado(fila, ids_reglas, colapsar=True):
    """Suma de pesos colapsando antes cada familia anidada.

    Sin el colapso, T1 a 61 grados aporta 1+2+3 = 6 por una sola condicion.
    Con el colapso aporta 3, que es el peso de A14.

    colapsar=False existe para poder reportar las dos columnas lado a lado:
    el contraste es la evidencia de por que el colapso hace falta.
    """
    activas = {r for r in ids_reglas if r in CRITICIDAD and fila[r] > 0}
    if colapsar:
        for miembros in familias_activas().values():
            presentes = [r for r in miembros if r in activas]
            if len(presentes) > 1:
                peor = min(presentes, key=lambda r: CRITICIDAD[r])
                activas -= set(presentes) - {peor}
    return sum(PESO[r] for r in activas)


def anotar_ventanas(ventanas, ids_reglas):
    """Agrega banda_criticidad, criticidad_min, peso_criticidad y componente."""
    v = ventanas.copy()
    crit = v[ids_reglas].apply(lambda f: criticidad_de_ventana(f, ids_reglas), axis=1)
    v['criticidad_min'] = crit
    v['banda_criticidad'] = crit.map(BANDA)
    v['peso_criticidad'] = v[ids_reglas].apply(
        lambda f: peso_agregado(f, ids_reglas), axis=1)
    v['peso_sin_colapsar'] = v[ids_reglas].apply(
        lambda f: peso_agregado(f, ids_reglas, colapsar=False), axis=1)

    def comp(fila):
        act = [r for r in ids_reglas if r in CRITICIDAD and fila[r] > 0]
        if not act:
            return 'normal'
        peor = min(act, key=lambda r: CRITICIDAD[r])
        return COMPONENTE.get(peor, 'desconocido')

    def dominante(fila):
        act = [r for r in ids_reglas if r in CRITICIDAD and fila[r] > 0]
        if not act:
            return 'ninguna'
        return min(act, key=lambda r: CRITICIDAD[r])

    v['componente'] = v[ids_reglas].apply(comp, axis=1)
    v['regla_dominante'] = v[ids_reglas].apply(dominante, axis=1)

    infl = 100 * (v.peso_sin_colapsar - v.peso_criticidad).sum() / \
        max(v.peso_criticidad.sum(), 1)
    print(f'Inflacion del peso por anidamiento, si no se colapsara: '
          f'{infl:.1f} %')
    print('Ese numero es la razon de ser del colapso. Va a la memoria.')
    return v


def comparar_escalas(ventanas):
    """Matriz entre la escala experta y la de numero de reglas activas.

    El acuerdo entre las dos es lo que compensa la limitacion declarada de la
    tabla -- elicitacion de un solo experto, no conversada con operadores.
    El desacuerdo tambien es un hallazgo: significa que contar subsistemas
    comprometidos y pesar por consecuencia no son lo mismo.
    """
    return pd.crosstab(ventanas.banda_severidad, ventanas.banda_criticidad)


def sensibilidad_por_regla(ventanas, ids_reglas):
    """Cuantas ventanas volverian a ser normales si se excluyera cada regla.

    Si una sola regla esta descalificando varios miles de ventanas, el material
    de entrenamiento del autoencoder lo esta decidiendo esa regla y no la fisica.
    """
    M = ventanas[ids_reglas].to_numpy() > 0
    base = int((M.sum(axis=1) == 0).sum())
    filas = []
    for i, r in enumerate(ids_reglas):
        sin_r = int((np.delete(M, i, axis=1).sum(axis=1) == 0).sum())
        filas.append({'regla': r,
                      'criticidad': CRITICIDAD.get(r),
                      'normales_si_se_excluye': sin_r,
                      'ganancia': sin_r - base,
                      'ganancia_pct': round(100 * (sin_r - base) / max(base, 1), 1)})
    return (pd.DataFrame(filas).sort_values('ganancia', ascending=False)
            .reset_index(drop=True))


# =============================================================================
# verificar_anidamiento  --  para decidir COLAPSAR_B_CON_A con datos
# =============================================================================
def verificar_anidamiento(ventanas, ids_reglas, umbral=0.95):
    """Mide la co-ocurrencia de cada par de una familia declarada, y de B/A.

    COMO LEERLO. Para un par (x, y) con y mas grave que x, el anidamiento
    matematico implica P(x | y) = 1: si se cumple el corte estricto, el laxo
    tambien. Las nueve familias de FAMILIAS_ANIDADAS tienen que dar 1,000 en
    esa columna; si alguna no da 1, el umbral que tengo escrito en el
    comentario no es el que usa reglas_engine.py y hay que revisarlo ANTES de
    seguir -- ese es el verdadero valor de esta funcion.

    Para B y A no hay anidamiento matematico, asi que la pregunta es empirica:
    si P(A | B) sale por encima del umbral, B casi nunca se activa sin A y
    sumar los dos pesos cuenta el mismo atollo dos veces. Ahi corresponde
    prender COLAPSAR_B_CON_A, dejando escrito el numero medido.
    """
    pres = {r: (ventanas[r].to_numpy() > 0) for r in ids_reglas
            if r in CRITICIDAD}
    filas = []

    def par(fam, x, y):
        if x not in pres or y not in pres:
            return
        nx, ny = int(pres[x].sum()), int(pres[y].sum())
        amb = int((pres[x] & pres[y]).sum())
        filas.append({
            'familia': fam, 'laxa': x, 'estricta': y,
            'n_laxa': nx, 'n_estricta': ny, 'ambas': amb,
            'P(laxa|estricta)': round(amb / ny, 3) if ny else np.nan,
            'P(estricta|laxa)': round(amb / nx, 3) if nx else np.nan,
        })

    for fam, miembros in FAMILIAS_ANIDADAS.items():
        orden = sorted([r for r in miembros if r in pres],
                       key=lambda r: -CRITICIDAD[r])      # de laxa a estricta
        for i in range(len(orden) - 1):
            par(fam, orden[i], orden[i + 1])

    t = pd.DataFrame(filas)
    if not t.empty:
        print('Familias declaradas -- P(laxa|estricta) tiene que valer 1,000:')
        print(t.to_string(index=False))
        malas = t[t['P(laxa|estricta)'] < 0.999]
        if len(malas):
            print('\n  >> ATENCION: estos pares NO estan anidados en los datos.')
            print('     Los umbrales que tengo comentados no coinciden con')
            print('     reglas_engine.py. Revisalo antes de usar peso_agregado.')
            print(malas.to_string(index=False))
        else:
            print('\n  Las nueve familias verifican. El colapso esta bien puesto.')

    print('\n--- El caso que NO es matematico: B (atollo) y A (sobrecarga) ---')
    if 'B' in pres and 'A' in pres:
        nb, na = int(pres['B'].sum()), int(pres['A'].sum())
        amb = int((pres['B'] & pres['A']).sum())
        p_a_dado_b = amb / nb if nb else np.nan
        p_b_dado_a = amb / na if na else np.nan
        print(f'  ventanas con B: {nb:,}   con A: {na:,}   con ambas: {amb:,}')
        print(f'  P(A | B) = {p_a_dado_b:.3f}      <-- el que decide')
        print(f'  P(B | A) = {p_b_dado_a:.3f}')
        if nb == 0:
            print('  B no se activa nunca en estos datos. El flag es irrelevante;')
            print('  dejalo en False y no gastes un parrafo en el.')
        elif p_a_dado_b >= umbral:
            print(f'  >> B casi nunca aparece sin A ({100*p_a_dado_b:.1f} %).')
            print(f'     Corresponde poner COLAPSAR_B_CON_A = True y escribir en')
            print(f'     la memoria que la decision se tomo con este numero.')
        else:
            print(f'  >> B aparece sin A en el {100*(1-p_a_dado_b):.1f} % de sus')
            print('     ventanas: son eventos distinguibles. Dejalo en False.')
            print('     Esto tambien es un resultado: los dos tags no son')
            print('     redundantes sobre estos datos.')
    else:
        print('  B o A no estan entre las reglas evaluadas.')
    return t


# =============================================================================
# cobertura_compuestas  --  cuanto cuestan las tres filas que el motor no ve
# =============================================================================
def cobertura_compuestas(ventanas, ids_reglas):
    """Cuantas ventanas cumplirian cada regla compuesta, y si cambiarian de banda.

    Sirve para que el capitulo de trabajo futuro diga un numero en vez de "no
    se implementaron". Si las tres juntas afectan a 40 ventanas de 20.000, la
    omision es intrascendente y se puede escribir asi, con respaldo.
    """
    filas = []
    for nombre, d in REGLAS_COMPUESTAS.items():
        miembros = [r for r in d['miembros'] if r in ids_reglas]
        if len(miembros) < len(d['miembros']):
            filas.append({'compuesta': nombre, 'evaluable': False,
                          'ventanas': 0, 'pct': 0.0, 'subirian_de_banda': 0})
            continue
        m = np.ones(len(ventanas), bool)
        for r in miembros:
            m &= ventanas[r].to_numpy() > 0
        n = int(m.sum())
        suben = 0
        if n and 'criticidad_min' in ventanas.columns:
            suben = int((ventanas.loc[m, 'criticidad_min'].to_numpy()
                         > d['criticidad']).sum())
        filas.append({'compuesta': nombre, 'evaluable': True,
                      'ventanas': n,
                      'pct': round(100 * n / max(len(ventanas), 1), 2),
                      'subirian_de_banda': suben})
    t = pd.DataFrame(filas)
    print('Reglas compuestas de la tabla que el motor no evalua:')
    print(t.to_string(index=False))
    print('\n"subirian_de_banda" son las ventanas cuya criticidad empeoraria si')
    print('la compuesta se implementara. Si es 0, la omision no cambia ninguna')
    print('etiqueta y el trabajo futuro es prolijidad, no una brecha.')
    return t


# =============================================================================
# test_equivalencia  --  tres assert, milisegundos, numeros para la memoria
# =============================================================================
def test_equivalencia(verbose=True):
    """Verifica el colapso por familias con casos de valor conocido a mano.

    No comprueba que la tabla del profesor sea correcta -- eso no es
    verificable desde aqui. Comprueba que el CODIGO hace lo que dice el
    parrafo del anidamiento, que es lo que se puede comprobar.
    """
    ids = list(CRITICIDAD.keys())

    def fila(activas):
        return pd.Series({r: (1 if r in activas else 0) for r in ids})

    casos = []

    # CASO 1 -- T1 a 61 grados: K (>50), A15 (>53) y A14 (>60) a la vez.
    f = fila({'K', 'A15', 'A14'})
    crit = criticidad_de_ventana(f, ids)
    pc = peso_agregado(f, ids, colapsar=True)
    ps = peso_agregado(f, ids, colapsar=False)
    assert crit == 1, f'T1=61: criticidad {crit}, esperaba 1 (A14)'
    assert pc == 3, f'T1=61: peso colapsado {pc}, esperaba 3 (solo A14)'
    assert ps == 6, f'T1=61: peso sin colapsar {ps}, esperaba 6 (1+2+3)'
    casos.append(('T1 = 61 C  (K, A15, A14)', crit, pc, ps))

    # CASO 2 -- vibracion a 41 mm/s: O (>16), A20 (>20), A19 (>34), A18 (>40).
    f = fila({'O', 'A20', 'A19', 'A18'})
    crit = criticidad_de_ventana(f, ids)
    pc = peso_agregado(f, ids, colapsar=True)
    ps = peso_agregado(f, ids, colapsar=False)
    assert crit == 1, f'vib=41: criticidad {crit}, esperaba 1'
    assert pc == 3, f'vib=41: peso colapsado {pc}, esperaba 3'
    assert ps == 9, f'vib=41: peso sin colapsar {ps}, esperaba 9 (1+2+3+3)'
    casos.append(('vib = 41 mm/s  (O, A20, A19, A18)', crit, pc, ps))

    # CASO 3 -- dos familias distintas activas: NO se colapsan entre si.
    # T1 a 61 y vibracion a 41 al mismo tiempo son dos condiciones fisicas
    # independientes y tienen que sumar 3 + 3 = 6.
    f = fila({'K', 'A15', 'A14', 'O', 'A20', 'A19', 'A18'})
    pc = peso_agregado(f, ids, colapsar=True)
    assert pc == 6, f'dos familias: peso colapsado {pc}, esperaba 6 (3+3)'
    casos.append(('las dos juntas', criticidad_de_ventana(f, ids), pc,
                  peso_agregado(f, ids, colapsar=False)))

    # CASO 4 -- ninguna regla: criticidad 4 (normal), peso 0.
    f = fila(set())
    assert criticidad_de_ventana(f, ids) == 4
    assert peso_agregado(f, ids) == 0
    casos.append(('ninguna activa', 4, 0, 0))

    if verbose:
        print('test_equivalencia -- OK')
        print(f'{"caso":<36}{"crit_min":>9}{"peso_col":>10}{"peso_sin":>10}')
        for nom, c, pc_, ps_ in casos:
            print(f'{nom:<36}{c:>9}{pc_:>10}{ps_:>10}')
        print('\nLa columna peso_sin es lo que daria sin colapsar familias.')
        print('La diferencia entre las dos ultimas columnas es exactamente el')
        print('error que el parrafo del anidamiento evita, cuantificado.')

    # coherencia de la tabla, por si se edita a mano
    assert len(CRITICIDAD) == 32, f'{len(CRITICIDAD)} reglas, esperaba 32'
    assert set(CRITICIDAD) >= set(COMPONENTE), 'COMPONENTE tiene IDs que no estan en CRITICIDAD'
    assert set(CRITICIDAD) >= set(FALLA), 'FALLA tiene IDs que no estan en CRITICIDAD'
    sin_comp = sorted(set(CRITICIDAD) - set(COMPONENTE))
    sin_falla = sorted(set(CRITICIDAD) - set(FALLA))
    if verbose and (sin_comp or sin_falla):
        print(f'\nSin componente asignado: {sin_comp}')
        print(f'Sin modo de falla asignado: {sin_falla}')
    for fam, miembros in FAMILIAS_ANIDADAS.items():
        faltan = [r for r in miembros if r not in CRITICIDAD]
        assert not faltan, f'familia {fam} cita reglas inexistentes: {faltan}'
    return True


# =============================================================================
# Como usarlo en el paso 02, reemplazando el bloque 5b
# =============================================================================
#   from criticidad import (anotar_ventanas, comparar_escalas,
#                           sensibilidad_por_regla, verificar_anidamiento,
#                           cobertura_compuestas, test_equivalencia)
#
#   test_equivalencia()                                  # primero, son ms
#
#   ventanas['n_reglas_activas'] = (ventanas[ids_reglas] > 0).sum(axis=1)
#   ventanas['banda_severidad']  = ventanas.n_reglas_activas.apply(asignar_banda)
#   ventanas = anotar_ventanas(ventanas, ids_reglas)
#
#   verificar_anidamiento(ventanas, ids_reglas)          # decide el flag
#   cobertura_compuestas(ventanas, ids_reglas)           # trabajo futuro
#   print(comparar_escalas(ventanas).to_string())
#   print(sensibilidad_por_regla(ventanas, ids_reglas).head(10).to_string(index=False))
#
# Y en el paso 03, la ablacion que si prueba lo que dice probar:
#   FEATURES_SIN_EXTREMOS = [f for f in FEATURES
#                            if not f.endswith(('_max', '_min'))]   # 42 de 70
# Sacar solo CM_max y PEL_max deja dentro T1_max, T7_max, V1_max... que son
# literalmente las condiciones de la veintena de reglas restantes.

if __name__ == '__main__':
    test_equivalencia()
