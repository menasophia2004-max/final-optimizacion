"""
streamlit_app.py — Modelo de optimización e interfaz Streamlit de producción y distribución
InduManufactura S.A.S.

Estructura (para facilitar modificaciones en la sustentación):
    1. DATOS BASE DEL CASO
    2. ESCENARIOS
    3. CONSTRUCCIÓN DEL MODELO
         3.1 Variables de decisión
         3.2 Función objetivo
         3.3 Restricciones
    4. SOLUCIÓN Y EXTRACCIÓN DE RESULTADOS
    5. VALIDACIÓN DE LA SOLUCIÓN

Unidades: costos en miles de COP, tiempos en horas por unidad, cantidades en unidades.
"""

import copy
import math

import pandas as pd
import pulp

# =============================================================================
# 1. DATOS BASE DEL CASO
# =============================================================================
PLANTAS = ["Norte", "Sur"]
PRODUCTOS = ["P1", "P2", "P3"]
CENTROS = ["Medellín", "Rionegro", "Bello"]

DATOS_BASE = {
    # Tabla 1 — horas disponibles por planta
    "horas": {"Norte": 1200, "Sur": 1000},
    # Tabla 2 — tiempo de fabricación (h/unidad) [producto][planta]
    "tiempo": {
        "P1": {"Norte": 2.0, "Sur": 2.5},
        "P2": {"Norte": 3.0, "Sur": 2.0},
        "P3": {"Norte": 1.5, "Sur": 2.0},
    },
    # Tabla 3 — costo de producción (miles COP/unidad) [producto][planta]
    "costo_prod": {
        "P1": {"Norte": 45, "Sur": 48},
        "P2": {"Norte": 60, "Sur": 55},
        "P3": {"Norte": 35, "Sur": 38},
    },
    # Tabla 4 — demanda (unidades) [centro][producto]
    "demanda": {
        "Medellín": {"P1": 120, "P2": 100, "P3": 150},
        "Rionegro": {"P1": 80, "P2": 90, "P3": 100},
        "Bello": {"P1": 100, "P2": 70, "P3": 130},
    },
    # Tabla 5 — costo de transporte (miles COP/unidad) [planta][centro]
    "costo_trans": {
        "Norte": {"Medellín": 8, "Rionegro": 12, "Bello": 10},
        "Sur": {"Medellín": 11, "Rionegro": 7, "Bello": 9},
    },
}


def datos_base():
    """Copia independiente de los datos originales (para no alterarlos)."""
    return copy.deepcopy(DATOS_BASE)


# =============================================================================
# 2. ESCENARIOS
# =============================================================================
def escenario_demanda(datos, producto="P2", pct=15):
    """Escenario 1: aumenta pct % la demanda de un producto en todos los centros (redondeo hacia arriba)."""
    d = copy.deepcopy(datos)
    for c in d["demanda"]:
        # round(...,6) evita errores de punto flotante antes del ceil (p. ej. 100*1.15 = 114.99999)
        d["demanda"][c][producto] = math.ceil(round(d["demanda"][c][producto] * (1 + pct / 100), 6))
    return d


def escenario_capacidad(datos, planta="Sur", pct=-20):
    """Escenario 2: cambia pct % las horas disponibles de una planta."""
    d = copy.deepcopy(datos)
    d["horas"][planta] = round(d["horas"][planta] * (1 + pct / 100), 6)
    return d


def escenario_costos(datos, planta="Sur", pct=30):
    """Escenario 3: cambia pct % los costos de transporte desde una planta."""
    d = copy.deepcopy(datos)
    for c in d["costo_trans"][planta]:
        d["costo_trans"][planta][c] = round(d["costo_trans"][planta][c] * (1 + pct / 100), 6)
    return d


def escenario_adicional(datos):
    """
    Escenario adicional (propio): "Temporada alta con parada de mantenimiento".
    Demanda de todos los productos +10 % (redondeo hacia arriba) y la Planta Norte
    pierde 15 % de sus horas por mantenimiento preventivo.
    """
    d = copy.deepcopy(datos)
    for c in d["demanda"]:
        for p in d["demanda"][c]:
            d["demanda"][c][p] = math.ceil(round(d["demanda"][c][p] * 1.10, 6))
    d["horas"]["Norte"] = round(d["horas"]["Norte"] * 0.85, 6)
    return d


ESCENARIOS = {
    "Base": lambda d: copy.deepcopy(d),
    "1. Demanda (+15 % P2)": escenario_demanda,
    "2. Capacidad (Sur −20 %)": escenario_capacidad,
    "3. Costos (transporte Sur +30 %)": escenario_costos,
    "4. Adicional (demanda +10 %, Norte −15 %)": escenario_adicional,
}


# =============================================================================
# 3. CONSTRUCCIÓN DEL MODELO
# =============================================================================
def construir_modelo(datos, entero=True):
    """
    Construye el modelo de PuLP a partir de un diccionario de datos.
    Retorna (modelo, x, y).
    """
    plantas = list(datos["horas"].keys())
    productos = list(datos["tiempo"].keys())
    centros = list(datos["demanda"].keys())
    cat = "Integer" if entero else "Continuous"

    m = pulp.LpProblem("InduManufactura", pulp.LpMinimize)

    # ---- 3.1 Variables de decisión -------------------------------------------
    # x[p,i]   : unidades del producto p fabricadas en la planta i
    # y[p,i,j] : unidades del producto p enviadas de la planta i al centro j
    x = pulp.LpVariable.dicts("x", [(p, i) for p in productos for i in plantas], lowBound=0, cat=cat)
    y = pulp.LpVariable.dicts(
        "y", [(p, i, j) for p in productos for i in plantas for j in centros], lowBound=0, cat=cat
    )

    # ---- 3.2 Función objetivo -------------------------------------------------
    costo_produccion = pulp.lpSum(datos["costo_prod"][p][i] * x[(p, i)] for p in productos for i in plantas)
    costo_transporte = pulp.lpSum(
        datos["costo_trans"][i][j] * y[(p, i, j)] for p in productos for i in plantas for j in centros
    )
    m += costo_produccion + costo_transporte, "Costo_total"

    # ---- 3.3 Restricciones ----------------------------------------------------
    # (R1) Demanda: lo que llega a cada centro cubre su demanda de cada producto
    for p in productos:
        for j in centros:
            m += pulp.lpSum(y[(p, i, j)] for i in plantas) >= datos["demanda"][j][p], f"Demanda_{p}_{j}"

    # (R2) Capacidad: horas usadas en cada planta <= horas disponibles
    for i in plantas:
        m += pulp.lpSum(datos["tiempo"][p][i] * x[(p, i)] for p in productos) <= datos["horas"][i], f"Capacidad_{i}"

    # (R3) Conservación de flujo: lo despachado desde una planta = lo fabricado allí
    for p in productos:
        for i in plantas:
            m += pulp.lpSum(y[(p, i, j)] for j in centros) == x[(p, i)], f"Balance_{p}_{i}"

    # (R4) No negatividad e integralidad: definidas en lowBound=0 y cat="Integer"

    # --------------------------------------------------------------------------
    # ESPACIO PARA RESTRICCIONES ADICIONALES (sustentación)
    # Ejemplo — producción mínima de P1 en Sur:
    #   m += x[("P1", "Sur")] >= 50, "Min_P1_Sur"
    # Ejemplo — capacidad máxima de envío por ruta:
    #   for i in plantas:
    #       for j in centros:
    #           m += pulp.lpSum(y[(p, i, j)] for p in productos) <= 300, f"Ruta_{i}_{j}"
    # --------------------------------------------------------------------------

    return m, x, y


# =============================================================================
# 4. SOLUCIÓN Y EXTRACCIÓN DE RESULTADOS
# =============================================================================
def _solver(tiempo_limite=60):
    """
    Usa CBC (incluido en PuLP 2.x); si no está, HiGHS (paquete highspy).
    gapRel=0 obliga a demostrar optimalidad exacta (sin brecha de tolerancia).
    """
    disponibles = pulp.listSolvers(onlyAvailable=True)
    if "PULP_CBC_CMD" in disponibles:
        return pulp.PULP_CBC_CMD(msg=False, timeLimit=tiempo_limite, gapRel=0)
    if "HiGHS" in disponibles:
        return pulp.HiGHS(msg=False, timeLimit=tiempo_limite, gapRel=0)
    raise RuntimeError("No hay un solucionador disponible. Instale 'pulp==2.9.0' o 'highspy'.")


def resolver(datos, entero=True):
    """
    Resuelve el modelo y devuelve un diccionario con:
      estado, optimo (bool), costo_total, costo_produccion, costo_transporte,
      produccion (DataFrame), distribucion (DataFrame), capacidad (DataFrame), validacion (DataFrame)
    Si el modelo no es óptimo, solo se devuelve el estado (no se muestran soluciones inválidas).
    """
    m, x, y = construir_modelo(datos, entero)
    m.solve(_solver())

    estado = pulp.LpStatus[m.status]
    estado_sol = pulp.LpSolution.get(m.sol_status, str(m.sol_status))
    optimo = m.status == pulp.LpStatusOptimal and m.sol_status == pulp.LpSolutionOptimal

    res = {"estado": estado, "estado_solucion": estado_sol, "optimo": optimo, "modelo": m}
    if not optimo:
        return res

    plantas = list(datos["horas"].keys())
    productos = list(datos["tiempo"].keys())
    centros = list(datos["demanda"].keys())
    v = lambda var: round(var.value() or 0, 6)  # noqa: E731

    # Producción: filas = producto, columnas = planta
    produccion = pd.DataFrame({i: {p: v(x[(p, i)]) for p in productos} for i in plantas})

    # Distribución: una fila por (producto, planta, centro) con envío > 0
    filas = []
    for p in productos:
        for i in plantas:
            for j in centros:
                q = v(y[(p, i, j)])
                if q > 1e-9:
                    filas.append(
                        {
                            "Producto": p,
                            "Planta": i,
                            "Centro": j,
                            "Unidades": q,
                            "Costo transporte": q * datos["costo_trans"][i][j],
                        }
                    )
    distribucion = pd.DataFrame(filas)

    # Utilización de capacidad
    cap = []
    for i in plantas:
        usadas = sum(datos["tiempo"][p][i] * v(x[(p, i)]) for p in productos)
        disp = datos["horas"][i]
        cap.append(
            {
                "Planta": i,
                "Horas disponibles": disp,
                "Horas usadas": usadas,
                "Holgura": disp - usadas,
                "Utilización (%)": 100 * usadas / disp if disp > 0 else 0,
            }
        )
    capacidad = pd.DataFrame(cap)

    c_prod = sum(datos["costo_prod"][p][i] * v(x[(p, i)]) for p in productos for i in plantas)
    c_trans = sum(
        datos["costo_trans"][i][j] * v(y[(p, i, j)]) for p in productos for i in plantas for j in centros
    )

    res.update(
        {
            "costo_total": c_prod + c_trans,
            "valor_objetivo_solver": pulp.value(m.objective),
            "costo_produccion": c_prod,
            "costo_transporte": c_trans,
            "produccion": produccion,
            "distribucion": distribucion,
            "capacidad": capacidad,
        }
    )
    res["validacion"] = validar(datos, res, x, y)
    return res


# =============================================================================
# 5. VALIDACIÓN DE LA SOLUCIÓN
# =============================================================================
def validar(datos, res, x, y, tol=1e-6):
    """Verifica independientemente cada restricción y recalcula la función objetivo."""
    plantas = list(datos["horas"].keys())
    productos = list(datos["tiempo"].keys())
    centros = list(datos["demanda"].keys())
    v = lambda var: var.value() or 0  # noqa: E731
    chequeos = []

    for p in productos:
        for j in centros:
            llega = sum(v(y[(p, i, j)]) for i in plantas)
            dem = datos["demanda"][j][p]
            chequeos.append(("Demanda", f"{p} → {j}", f"{llega:,.0f} ≥ {dem:,.0f}", llega >= dem - tol))

    for i in plantas:
        usadas = sum(datos["tiempo"][p][i] * v(x[(p, i)]) for p in productos)
        disp = datos["horas"][i]
        chequeos.append(("Capacidad", i, f"{usadas:,.1f} ≤ {disp:,.1f} h", usadas <= disp + tol))

    for p in productos:
        for i in plantas:
            env = sum(v(y[(p, i, j)]) for j in centros)
            fab = v(x[(p, i)])
            chequeos.append(("Conservación", f"{p} en {i}", f"{env:,.0f} = {fab:,.0f}", abs(env - fab) <= tol))

    todas = list(x.values()) + list(y.values())
    neg = [var.name for var in todas if v(var) < -tol]
    no_ent = [var.name for var in todas if abs(v(var) - round(v(var))) > tol]
    chequeos.append(("No negatividad", "Todas las variables", f"{len(neg)} negativas", len(neg) == 0))
    chequeos.append(("Integralidad", "Todas las variables", f"{len(no_ent)} fraccionarias", len(no_ent) == 0))

    dif = abs(res["costo_total"] - res["valor_objetivo_solver"])
    chequeos.append(
        (
            "Función objetivo",
            "Recalculada vs. solver",
            f"{res['costo_total']:,.2f} vs {res['valor_objetivo_solver']:,.2f}",
            dif <= 1e-4,
        )
    )
    return pd.DataFrame(chequeos, columns=["Tipo", "Restricción", "Verificación", "Cumple"])


# #############################################################################
# INTERFAZ STREAMLIT
# #############################################################################
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="InduManufactura · Optimización", page_icon="🏭", layout="wide")

# Paleta categórica fija (planta/componente → color). El color sigue a la entidad.
COLORES = {"Norte": "#2a78d6", "Sur": "#eb6834", "Producción": "#2a78d6", "Transporte": "#eb6834"}
# Etiquetas cortas para los ejes de los gráficos
CORTO = {
    "Base": "Base",
    "1. Demanda (+15 % P2)": "E1 Demanda",
    "2. Capacidad (Sur −20 %)": "E2 Capacidad",
    "3. Costos (transporte Sur +30 %)": "E3 Costos",
    "4. Adicional (demanda +10 %, Norte −15 %)": "E4 Adicional",
    "Personalizado": "Personalizado",
}


# -----------------------------------------------------------------------------
# Conversión entre el diccionario de datos del modelo y tablas editables
# -----------------------------------------------------------------------------
def a_tablas(d):
    plantas = list(d["horas"].keys())
    return {
        "horas": pd.DataFrame({"Horas disponibles": d["horas"]}).rename_axis("Planta"),
        "tiempo": pd.DataFrame(d["tiempo"]).T[plantas].rename_axis("Producto"),
        "costo_prod": pd.DataFrame(d["costo_prod"]).T[plantas].rename_axis("Producto"),
        "demanda": pd.DataFrame(d["demanda"]).T.rename_axis("Centro"),
        "costo_trans": pd.DataFrame(d["costo_trans"]).T.rename_axis("Planta"),
    }


def a_datos(t):
    return {
        "horas": {k: float(v) for k, v in t["horas"]["Horas disponibles"].items()},
        "tiempo": {p: {i: float(t["tiempo"].loc[p, i]) for i in t["tiempo"].columns} for p in t["tiempo"].index},
        "costo_prod": {
            p: {i: float(t["costo_prod"].loc[p, i]) for i in t["costo_prod"].columns} for p in t["costo_prod"].index
        },
        "demanda": {c: {p: float(t["demanda"].loc[c, p]) for p in t["demanda"].columns} for c in t["demanda"].index},
        "costo_trans": {
            i: {c: float(t["costo_trans"].loc[i, c]) for c in t["costo_trans"].columns} for i in t["costo_trans"].index
        },
    }


def errores_datos(t):
    """Revisa valores vacíos o negativos antes de optimizar."""
    errs = []
    nombres = {
        "horas": "Horas disponibles",
        "tiempo": "Tiempos de fabricación",
        "costo_prod": "Costos de producción",
        "demanda": "Demanda",
        "costo_trans": "Costos de transporte",
    }
    for k, df in t.items():
        if df.isna().any().any():
            errs.append(f"{nombres[k]}: hay celdas vacías.")
        elif (df < 0).any().any():
            errs.append(f"{nombres[k]}: hay valores negativos.")
    if (t["tiempo"] == 0).any().any():
        errs.append("Tiempos de fabricación: un tiempo de 0 h permitiría producción ilimitada.")
    return errs


def cargar(datos, etiqueta):
    st.session_state.tablas = a_tablas(datos)
    st.session_state.origen = etiqueta
    st.session_state.version = st.session_state.get("version", 0) + 1  # reinicia los editores
    st.session_state.pop("resultado", None)


def miles(x):
    """Formato colombiano: 50.673,4 (decimal solo si existe)."""
    txt = f"{x:,.1f}" if abs(x - round(x)) > 1e-9 else f"{x:,.0f}"
    return "$" + txt.replace(",", "X").replace(".", ",").replace("X", ".")


if "tablas" not in st.session_state:
    cargar(datos_base(), "Base")

# -----------------------------------------------------------------------------
# Barra lateral
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    esc = st.selectbox("Cargar escenario predefinido", list(ESCENARIOS.keys()))
    if st.button("Cargar en el editor", width="stretch"):
        cargar(ESCENARIOS[esc](datos_base()), esc)
        st.rerun()
    if st.button("↺ Restaurar datos originales", width="stretch"):
        cargar(datos_base(), "Base")
        st.rerun()
    st.caption(f"Datos actuales: **{st.session_state.origen}**")
    st.divider()
    entero = st.toggle("Variables enteras", value=True, help="Integralidad de unidades producidas y enviadas.")
    st.divider()
    st.caption("Costos en miles de COP · tiempos en h/unidad · cantidades en unidades.")

st.title("🏭 InduManufactura S.A.S.")
st.markdown("**Optimización integrada de producción y distribución** — plantas Norte y Sur → Medellín, Rionegro y Bello")

tab_ctx, tab_datos, tab_res, tab_esc = st.tabs(
    ["📘 Contexto y modelo", "✏️ Datos y optimización", "📊 Resultados", "🔀 Comparación de escenarios"]
)

# -----------------------------------------------------------------------------
# Pestaña 1 — Contexto y formulación
# -----------------------------------------------------------------------------
with tab_ctx:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Contexto")
        st.markdown(
            """
InduManufactura S.A.S. fabrica tres referencias (**P1, P2, P3**) en dos plantas (**Norte** y **Sur**)
y abastece tres centros de distribución (**Medellín, Rionegro, Bello**). Cada planta tiene horas
limitadas y tiempos/costos de fabricación distintos; cada ruta tiene un costo de transporte.

**Pregunta de decisión:** ¿cuánto fabricar de cada producto en cada planta y cuánto enviar a cada centro
para cubrir toda la demanda **al menor costo total** (producción + transporte)?
"""
        )
        st.subheader("Conjuntos e índices")
        st.markdown(
            """
- $P=\\{P1,P2,P3\\}$ productos, índice $p$
- $I=\\{Norte, Sur\\}$ plantas, índice $i$
- $J=\\{Medellín, Rionegro, Bello\\}$ centros, índice $j$
"""
        )
        st.subheader("Parámetros")
        st.markdown(
            """
- $H_i$: horas disponibles en la planta $i$ (h)
- $t_{pi}$: tiempo de fabricación de $p$ en $i$ (h/unidad)
- $c_{pi}$: costo de producción de $p$ en $i$ (miles COP/unidad)
- $d_{pj}$: demanda de $p$ en el centro $j$ (unidades)
- $k_{ij}$: costo de transporte de $i$ a $j$ (miles COP/unidad)
"""
        )
    with c2:
        st.subheader("Variables de decisión")
        st.markdown(
            """
- $x_{pi}\\ge 0$, entera: unidades de $p$ fabricadas en la planta $i$
- $y_{pij}\\ge 0$, entera: unidades de $p$ enviadas de $i$ a $j$
"""
        )
        st.subheader("Función objetivo")
        st.latex(r"\min Z=\sum_{p\in P}\sum_{i\in I} c_{pi}\,x_{pi}+\sum_{p\in P}\sum_{i\in I}\sum_{j\in J} k_{ij}\,y_{pij}")
        st.subheader("Restricciones")
        st.markdown("**(R1) Demanda** — cada centro recibe al menos lo que demanda:")
        st.latex(r"\sum_{i\in I} y_{pij}\ \ge\ d_{pj}\qquad \forall p\in P,\ j\in J")
        st.markdown("**(R2) Capacidad** — no se superan las horas de cada planta:")
        st.latex(r"\sum_{p\in P} t_{pi}\,x_{pi}\ \le\ H_i\qquad \forall i\in I")
        st.markdown("**(R3) Conservación** — se despacha exactamente lo fabricado:")
        st.latex(r"\sum_{j\in J} y_{pij}\ =\ x_{pi}\qquad \forall p\in P,\ i\in I")
        st.markdown("**(R4) No negatividad e integralidad:**")
        st.latex(r"x_{pi},\ y_{pij}\in \mathbb{Z}_{\ge 0}")
        st.caption("Tamaño: 6 variables x + 18 variables y = 24 variables · 9 + 2 + 6 = 17 restricciones.")

# -----------------------------------------------------------------------------
# Pestaña 2 — Datos editables y botón de optimización
# -----------------------------------------------------------------------------
with tab_datos:
    t = st.session_state.tablas
    v = st.session_state.version
    st.info(
        "Edite cualquier celda y presione **Optimizar**. Los cambios no requieren tocar el código. "
        f"Datos cargados: **{st.session_state.origen}**."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Tabla 1. Horas disponibles por planta**")
        e_horas = st.data_editor(t["horas"], key=f"horas_{v}", width="stretch")
    with c2:
        st.markdown("**Tabla 2. Tiempo de fabricación (h/unidad)**")
        e_tiempo = st.data_editor(t["tiempo"], key=f"tiempo_{v}", width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tabla 3. Costo de producción (miles COP/unidad)**")
        e_cprod = st.data_editor(t["costo_prod"], key=f"cprod_{v}", width="stretch")
    with c2:
        st.markdown("**Tabla 5. Costo de transporte (miles COP/unidad)**")
        e_ctrans = st.data_editor(t["costo_trans"], key=f"ctrans_{v}", width="stretch")

    st.markdown("**Tabla 4. Demanda (unidades)**")
    e_dem = st.data_editor(t["demanda"], key=f"dem_{v}", width="stretch")
    tot = e_dem.sum()
    st.caption("Totales por producto: " + " · ".join(f"{p}: {tot[p]:,.0f}" for p in tot.index))

    editadas = {"horas": e_horas, "tiempo": e_tiempo, "costo_prod": e_cprod, "demanda": e_dem, "costo_trans": e_ctrans}

    if st.button("🚀 Optimizar", type="primary", width="stretch"):
        errs = errores_datos(editadas)
        if errs:
            st.session_state.pop("resultado", None)
            for e in errs:
                st.error(e)
        else:
            datos = a_datos(editadas)
            with st.spinner("Resolviendo el modelo..."):
                try:
                    st.session_state.resultado = resolver(datos, entero=entero)
                    st.session_state.datos_resueltos = datos
                except Exception as ex:  # error del solucionador u otro
                    st.session_state.resultado = {"estado": "Error", "estado_solucion": str(ex), "optimo": False}
            r = st.session_state.resultado
            if r["optimo"]:
                st.success(f"Solución óptima encontrada. Costo total: {miles(r['costo_total'])} miles de COP. "
                           "Vea la pestaña **📊 Resultados**.")
            else:
                st.error(f"El solucionador reporta: **{r['estado']}** ({r['estado_solucion']}). Vea **📊 Resultados**.")

# -----------------------------------------------------------------------------
# Pestaña 3 — Resultados
# -----------------------------------------------------------------------------
with tab_res:
    r = st.session_state.get("resultado")
    if r is None:
        st.info("Aún no hay resultados. Vaya a **✏️ Datos y optimización** y presione **Optimizar**.")
    elif not r["optimo"]:
        explicacion = {
            "Infeasible": "No existe ninguna producción que cubra toda la demanda con las horas disponibles. "
            "Revise: horas disponibles, tiempos de fabricación o demanda.",
            "Unbounded": "El modelo es no acotado: revise que los costos no sean negativos.",
            "Not Solved": "El solucionador no terminó (p. ej. límite de tiempo).",
            "Undefined": "El solucionador no pudo determinar el estado; suele indicar infactibilidad.",
        }
        st.error(f"### Estado del solucionador: {r['estado']}")
        st.write(explicacion.get(r["estado"], r["estado_solucion"]))
        st.warning("Por seguridad no se muestra ninguna solución, porque no sería válida.")
        if r["estado"] in ("Infeasible", "Undefined") and "datos_resueltos" in st.session_state:
            d = st.session_state.datos_resueltos
            # Diagnóstico simple: horas mínimas si cada producto se hiciera en su planta más rápida
            minimo = sum(
                sum(d["demanda"][c][p] for c in d["demanda"]) * min(d["tiempo"][p].values()) for p in d["tiempo"]
            )
            st.info(
                f"Diagnóstico: aun fabricando cada producto en su planta más rápida se requieren "
                f"≈ {miles(minimo)[1:]} h; hay {miles(sum(d['horas'].values()))[1:]} h disponibles en total "
                "(la capacidad individual de cada planta también limita)."
            )
    else:
        st.success(f"Estado del solucionador: **{r['estado']}** — solución óptima.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Costo total (miles COP)", miles(r["costo_total"]))
        m2.metric("Costo de producción", miles(r["costo_produccion"]))
        m2.caption(f"{100 * r['costo_produccion'] / r['costo_total']:.1f} % del costo total")
        m3.metric("Costo de transporte", miles(r["costo_transporte"]))
        m3.caption(f"{100 * r['costo_transporte'] / r['costo_total']:.1f} % del costo total")
        m4.metric("Unidades producidas", f"{r['produccion'].values.sum():,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Producción óptima (unidades)")
            prod = r["produccion"].copy()
            prod["Total"] = prod.sum(axis=1)
            st.dataframe(prod.style.format("{:,.0f}"), width="stretch")
            larga = r["produccion"].reset_index().melt(id_vars="index", var_name="Planta", value_name="Unidades")
            fig = px.bar(larga, x="index", y="Unidades", color="Planta", barmode="group",
                         color_discrete_map=COLORES, labels={"index": "Producto"}, text_auto=".0f")
            fig.update_layout(height=320, margin=dict(t=10, b=10), legend_title_text="")
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("#### Utilización de capacidad")
            cap = r["capacidad"]
            fig = go.Figure()
            fig.add_bar(y=cap["Planta"], x=cap["Horas usadas"], orientation="h", name="Horas usadas",
                        marker_color=[COLORES[p] for p in cap["Planta"]],
                        text=[f"{u:,.0f} h ({p:.1f} %)" for u, p in zip(cap["Horas usadas"], cap["Utilización (%)"])],
                        textposition="inside", hovertemplate="%{y}: %{x:,.1f} h usadas<extra></extra>")
            fig.add_bar(y=cap["Planta"], x=cap["Holgura"], orientation="h", name="Holgura",
                        marker_color="#d4d3cf", hovertemplate="%{y}: %{x:,.1f} h libres<extra></extra>")
            fig.update_layout(barmode="stack", height=230, margin=dict(t=10, b=10), xaxis_title="Horas",
                              legend=dict(orientation="h", y=-0.35))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(cap.set_index("Planta").style.format("{:,.1f}"), width="stretch")

        st.markdown("#### Distribución óptima (unidades enviadas)")
        dist = r["distribucion"]
        piv = dist.pivot_table(index=["Planta", "Producto"], columns="Centro", values="Unidades",
                               aggfunc="sum", fill_value=0)
        piv = piv.reindex(columns=[c for c in st.session_state.datos_resueltos["demanda"] if c in piv.columns])
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(piv.style.format("{:,.0f}"), width="stretch")
            st.caption("Detalle por ruta con su costo de transporte:")
            st.dataframe(dist.style.format({"Unidades": "{:,.0f}", "Costo transporte": "{:,.1f}"}),
                         width="stretch", hide_index=True)
        with c2:
            fig = px.bar(dist, x="Centro", y="Unidades", color="Planta", pattern_shape="Producto",
                         color_discrete_map=COLORES, hover_data=["Producto", "Costo transporte"],
                         category_orders={"Centro": list(st.session_state.datos_resueltos["demanda"]),
                                          "Producto": sorted(dist["Producto"].unique())})
            fig.update_layout(height=380, margin=dict(t=10, b=10), legend_title_text="")
            st.plotly_chart(fig, width="stretch")

        st.markdown("#### Validación de la solución")
        val = r["validacion"]
        if val["Cumple"].all():
            st.success(f"Las {len(val)} verificaciones se cumplen: restricciones satisfechas y función objetivo recalculada.")
        else:
            st.error("Alguna verificación falló. Revise la tabla.")
        with st.expander("Ver detalle de las verificaciones"):
            st.dataframe(val.assign(Cumple=val["Cumple"].map({True: "✅", False: "❌"})),
                         width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# Pestaña 4 — Comparación de escenarios
# -----------------------------------------------------------------------------
with tab_esc:
    st.markdown(
        "Se resuelven **de forma independiente** el escenario base, los tres escenarios obligatorios y el "
        "escenario adicional, todos a partir de los datos originales del caso."
    )
    with st.expander("Definición de los escenarios"):
        st.markdown(
            """
| Escenario | Modificación |
|---|---|
| Base | Datos originales del caso |
| 1. Demanda | +15 % la demanda de P2 en los tres centros, redondeando hacia arriba (115, 104, 81) |
| 2. Capacidad | −20 % las horas de la Planta Sur (1.000 → 800 h); Norte sin cambios |
| 3. Costos | +30 % los costos de transporte desde la Planta Sur (11→14,3; 7→9,1; 9→11,7) |
| 4. Adicional | Temporada alta con mantenimiento: +10 % toda la demanda (↑) y Planta Norte −15 % horas (1.200 → 1.020 h) |
"""
        )
    incluir_actual = st.checkbox("Incluir también los datos actuales del editor como escenario «Personalizado»")

    if st.button("▶️ Ejecutar comparación", type="primary"):
        filas, resultados = [], {}
        lista = [(n, f(datos_base())) for n, f in ESCENARIOS.items()]
        if incluir_actual:
            if errores_datos(editadas):
                st.error("Los datos del editor tienen errores; no se incluye el escenario personalizado.")
            else:
                lista.append(("Personalizado", a_datos(editadas)))
        for nombre, datos in lista:
            r = resolver(datos, entero=entero)
            resultados[nombre] = r
            fila = {"Escenario": nombre, "Estado": r["estado"]}
            if r["optimo"]:
                cap = r["capacidad"].set_index("Planta")["Utilización (%)"]
                fila.update({
                    "Costo total": r["costo_total"],
                    "Costo producción": r["costo_produccion"],
                    "Costo transporte": r["costo_transporte"],
                    "Util. Norte (%)": cap.get("Norte"),
                    "Util. Sur (%)": cap.get("Sur"),
                    "Validación": "✅" if r["validacion"]["Cumple"].all() else "❌",
                })
            filas.append(fila)
        st.session_state.comparacion = (pd.DataFrame(filas), resultados)

    if "comparacion" in st.session_state:
        tabla, resultados = st.session_state.comparacion
        base_total = tabla.loc[tabla["Escenario"] == "Base", "Costo total"].iloc[0]
        tabla["Δ vs. base"] = tabla["Costo total"] - base_total
        tabla["Δ vs. base (%)"] = 100 * tabla["Δ vs. base"] / base_total
        st.markdown("#### Tabla comparativa (miles de COP)")
        st.dataframe(
            tabla.style.format({
                "Costo total": "{:,.1f}", "Costo producción": "{:,.1f}", "Costo transporte": "{:,.1f}",
                "Util. Norte (%)": "{:.1f}", "Util. Sur (%)": "{:.1f}", "Δ vs. base": "{:+,.1f}",
                "Δ vs. base (%)": "{:+.2f} %",
            }, na_rep="—"),
            width="stretch", hide_index=True,
        )
        ok = tabla.dropna(subset=["Costo total"]).copy()
        ok["Escenario"] = ok["Escenario"].map(CORTO)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Composición del costo")
            larga = ok.melt(id_vars="Escenario", value_vars=["Costo producción", "Costo transporte"],
                            var_name="Componente", value_name="Miles COP")
            larga["Componente"] = larga["Componente"].str.replace("Costo p", "P").str.replace("Costo t", "T")
            fig = px.bar(larga, x="Escenario", y="Miles COP", color="Componente", color_discrete_map=COLORES)
            for _, f in ok.iterrows():
                fig.add_annotation(x=f["Escenario"], y=f["Costo total"], text=f"{f['Costo total']:,.0f}",
                                   showarrow=False, yshift=10, font=dict(size=11))
            fig.update_layout(height=420, margin=dict(t=20, b=10), legend_title_text="", xaxis_title="")
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("#### Utilización de capacidad (%)")
            larga = ok.melt(id_vars="Escenario", value_vars=["Util. Norte (%)", "Util. Sur (%)"],
                            var_name="Planta", value_name="Utilización (%)")
            larga["Planta"] = larga["Planta"].str.extract(r"(Norte|Sur)")[0]
            fig = px.bar(larga, x="Escenario", y="Utilización (%)", color="Planta", barmode="group",
                         color_discrete_map=COLORES, text_auto=".0f")
            fig.update_traces(textangle=0, textposition="outside")
            fig.add_hline(y=100, line_dash="dot", line_color="#8a8984")
            fig.update_layout(height=420, margin=dict(t=20, b=10), legend_title_text="", xaxis_title="",
                              yaxis_range=[0, 115])
            st.plotly_chart(fig, width="stretch")

        st.markdown("#### Producción por planta y producto en cada escenario (unidades)")
        filas = []
        for n, r in resultados.items():
            if r["optimo"]:
                for p, fila in r["produccion"].iterrows():
                    for i, q in fila.items():
                        filas.append({"Escenario": CORTO[n], "Producto": p, "Planta": i, "Unidades": q})
        prod = pd.DataFrame(filas)
        piv = prod.pivot_table(index="Escenario", columns=["Planta", "Producto"], values="Unidades", sort=False)
        st.dataframe(piv.style.format("{:,.0f}"), width="stretch")
        fig = px.bar(prod, x="Producto", y="Unidades", color="Planta", facet_col="Escenario", facet_col_wrap=5,
                     color_discrete_map=COLORES, barmode="stack")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(height=360, margin=dict(t=40, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")

        infact = tabla[tabla["Estado"] != "Optimal"]
        for _, f in infact.iterrows():
            st.error(f"Escenario «{f['Escenario']}»: estado **{f['Estado']}** — sin solución válida que mostrar.")
