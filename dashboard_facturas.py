# -*- coding: utf-8 -*-
"""
Dashboard analitico — Validacion de Facturas Electronicas.

Ejecutar:  python -m streamlit run dashboard_facturas.py
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# -------------------------------------------------------------------------
# Config y paleta
# -------------------------------------------------------------------------
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_JSON = os.path.join(DIR_BASE, "resultados_facturas_dashboard.json")
RUTA_CSV = os.path.join(DIR_BASE, "db_facturas.csv")

C_LILT = "#2E86AB"      # LiLT
C_LMV3 = "#E4572E"      # LayoutLMv3
C_ENS = "#7B2CBF"       # ensamble focal (modelo final)
C_OK = "#2A9D8F"        # valores altos / correctos
C_ALERTA = "#E76F51"    # valores bajos / a reforzar
C_NEUTRO = "#457B9D"

st.set_page_config(page_title="Sistema Analítico de Facturas", layout="wide",
                   initial_sidebar_state="expanded")


def kpi_card(label, value, delta=None):
    """Indicador del resumen ejecutivo con recuadro de fondo gris claro para resaltarlo."""
    delta_html = (
        f'<div style="color:#2A9D8F;font-size:0.85rem;font-weight:600;margin-top:4px;">{delta}</div>'
        if delta else ""
    )
    return (
        '<div style="background:#F1F3F5;border:1px solid #D8DEE4;border-radius:10px;'
        'padding:16px 18px;height:100%;">'
        f'<div style="color:#5A6B7B;font-size:0.78rem;letter-spacing:.04em;'
        f'text-transform:uppercase;margin-bottom:6px;">{label}</div>'
        f'<div style="color:#1E2A38;font-size:1.7rem;font-weight:700;line-height:1.1;">{value}</div>'
        f'{delta_html}</div>'
    )


def analisis(texto):
    """Nota breve que interpreta los resultados de la sección en lenguaje natural."""
    st.markdown(
        '<div style="background:#EEF4F8;border-left:4px solid #2E86AB;border-radius:6px;'
        'padding:12px 16px;margin:4px 0 16px 0;">'
        f'<div style="color:#25333F;font-size:0.92rem;line-height:1.55;">{texto}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cargar_json(ruta):
    if not os.path.exists(ruta):
        return None
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def cargar_csv(ruta):
    if not os.path.exists(ruta):
        return None
    try:
        return pd.read_csv(ruta, sep=";", encoding="utf-8-sig", dtype=str)
    except Exception as e:
        st.warning(f"No se pudo leer db_facturas.csv: {e}")
        return None


def pendiente(clave, notebook="generar_datos_dashboard.ipynb"):
    """Aviso para graficos cuyos datos aun no estan en el JSON."""
    st.info(
        f"Gráfico nativo pendiente de datos (`{clave}`). "
        f"Ejecuta **{notebook}** en el VPS y fusiona su salida en `resultados_facturas_dashboard.json`."
    )


DATA = cargar_json(RUTA_JSON)
DB = cargar_csv(RUTA_CSV)
if DATA is None:
    st.error("No se encontró 'resultados_facturas_dashboard.json' junto a este script.")
    st.stop()
if not PLOTLY_OK:
    st.error("Falta plotly. Instala: pip install plotly")
    st.stop()

proyecto = DATA.get("proyecto", {})
ds = DATA.get("dataset", {})
ef = DATA.get("ensamble_focal", {})

# -------------------------------------------------------------------------
# Sidebar — indice de la pagina unica
# -------------------------------------------------------------------------
st.sidebar.title("Índice")
st.sidebar.markdown("""
- [Resumen ejecutivo](#resumen-ejecutivo)
- [EDA 1 · Base de referencia](#eda-1-base-de-referencia)
- [EDA 2 · OCR](#eda-2-ocr)
- [EDA 3 · Balance del dataset](#eda-3-balance-del-dataset)
- [Comparativa de modelos](#comparativa-de-modelos)
- [Modelo elegido · ensamble focal](#modelo-elegido-ensamble-focal)
- [Validación en 2 capas](#validacion-en-2-capas)
""")

# =========================================================================
# ENCABEZADO
# =========================================================================
st.title("Tablero del Sistema Analítico de Facturas")
st.markdown(f"#### {proyecto.get('titulo','')}")
st.markdown(proyecto.get("descripcion", ""))
st.markdown(f"**Modelo final:** {proyecto.get('modelo_final','N/D')}")

# =========================================================================
# 1) RESUMEN EJECUTIVO
# =========================================================================
st.divider()
st.header("Resumen ejecutivo")
analisis(
    "En las facturas del presente trabajo, LiLT, que se apoya solo en el texto y su disposición "
    "generaliza mejor que LayoutLMv3, que además usa la imagen, lo supera en F1 macro (0,726 frente a "
    "0,671). La diferencia se nota sobre todo en los campos que dependen del texto, "
    "como los valores, la fecha y el adquiriente. El sistema "
    "que se lleva a producción es el ensamble focal, el de mejor desempeño de todo el estudio (F1 micro "
    "0,755 y 69,1% de campos completos por documento)."
    "<br><br>"
    "<b>Síntesis del estudio: </b>"
    "Se compararon los modelos LayoutLMv3 (texto+posición+imagen) y LiLT (texto+layout), en emisores no vistos "
    "LiLT ganó (F1 macro 0,726 vs 0,671). Se aplicaron las mejoras: focal loss y ensamble por promedio "
    "de probabilidades, se selecciona al ensamble focal (LiLT focal + LayoutLMv3 focal) como modelo final "
    "(F1 micro 0,755), con ventaja estadísticamente significativa sobre el mejor modelo único. "
    "La salida se valida en dos capas (reglas + cruce por CUFE con una base de referencia)."
)

k = st.columns(4)
k[0].markdown(kpi_card("Modelo final", "Ensamble focal"), unsafe_allow_html=True)
k[1].markdown(kpi_card("F1 micro", f"{ef.get('f1_micro',0):.3f}"), unsafe_allow_html=True)
k[2].markdown(kpi_card("F1 macro", f"{ef.get('f1_macro',0):.3f}"), unsafe_allow_html=True)
k[3].markdown(kpi_card("Completitud / doc", f"{ef.get('completitud_doc_pct',0):.1f}%"), unsafe_allow_html=True)
st.write("")
k2 = st.columns(4)
k2[0].markdown(kpi_card("Campos KIE", f"{len(proyecto.get('campos_kie',[]))}"), unsafe_allow_html=True)
k2[1].markdown(kpi_card("Emisores", f"{ds.get('emisores',0)}"), unsafe_allow_html=True)
k2[2].markdown(kpi_card("Train / Test", f"{ds.get('train_facturas',0)} / {ds.get('test_facturas',0)}"), unsafe_allow_html=True)
k2[3].markdown(kpi_card("Páginas OCR", f"{ds.get('paginas_ocr',0)}"), unsafe_allow_html=True)


# =========================================================================
# 2) EDA 1 — db_facturas.csv (nativo, desde el CSV)
# =========================================================================
st.divider()
st.header("EDA 1 · Base de referencia")
st.caption("Facturas reales enmascaradas (data masking), la validación es sintáctica/coherencia.")
analisis(
    "Se cuenta con una base de 600 facturas reales enmascaradas, de la cual tenemos los datos Gold, la cual es la fuente de verdad contra la que se "
    "contrastan los campos extraídos. El cruce se hace por CUFE, que es único en las 600 filas y "
    "coincide con el que viaja dentro del propio código QR, así que funciona como llave sin ambigüedad. "
    "La completitud varía bastante de una columna a otra, y eso es factor importante porque toda columna vacía se "
    "marca como no comparable en el cruce."
)

if DB is None:
    st.warning("No se encontró `db_facturas.csv` junto al script.")
else:
    comp = (DB.replace("", np.nan).notna().mean() * 100).round(1).sort_values()
    dfc = pd.DataFrame({"columna": comp.index, "pct": comp.values})
    fig = px.bar(dfc, x="pct", y="columna", orientation="h",
                 text=dfc["pct"].map(lambda v: f"{v:.0f}%"),
                 color="pct", color_continuous_scale="Blues", range_x=[0, 108])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(title="Completitud por columna (% poblado)",
                      height=460, margin=dict(t=40, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    c = st.columns(3)
    n = len(DB)
    c[0].metric("Filas", f"{n}")
    for col, box in (("NUMERO_FACTURA", c[1]), ("CODIGO_CUFE", c[2])):
        if col in DB.columns:
            u = DB[col].nunique(dropna=True)
            box.metric(f"{col} únicos", f"{u} / {n}", f"{u/n*100:.1f}%" if n else "")

    cciu = next((c for c in DB.columns if "CIUDAD" in c.upper()), None)
    if cciu:
        ciudades = DB[cciu].replace("", np.nan).dropna()
        top = ciudades.value_counts().head(15).sort_values()
        n_vacias = len(DB) - len(ciudades)
        fig = px.bar(x=top.values, y=top.index, orientation="h", text=top.values,
                     color_discrete_sequence=[C_NEUTRO])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(title="Top-15 ciudades del adquiriente", height=430,
                          xaxis_title="facturas", yaxis_title="",
                          margin=dict(t=40, b=10), xaxis_range=[0, max(top.values) * 1.15])
        st.plotly_chart(fig, use_container_width=True)
        if n_vacias:
            st.caption(f"Excluye {n_vacias} facturas sin ciudad registrada "
                       f"({n_vacias/len(DB)*100:.0f}% de la base).")
    with st.expander("Ver muestra de datos (primeras filas)"):
        st.dataframe(DB.head(12), use_container_width=True)

# =========================================================================
# 3) EDA 2 — OCR (nativo, desde eda_ocr)
# =========================================================================
st.divider()
st.header("EDA 2 · OCR")
st.caption("El texto que produce PaddleOCR, cuántos tokens por página, con qué confianza y qué ancho de caja.")
analisis(
    "Sobre el OCR de las 620 páginas, el OCR deja una mediana de unos 100 tokens por página y una "
    "confianza alta (con mediana cercana a 0,99), lo que conviene mirar con atención son los tokens "
    "anchos, de 25 caracteres o más, debido a que suelen ser aquellos en los que el reconocedor pega el rótulo y el "
    "valor en una sola pieza. No es un caso aislado sino un patrón frecuente, y es justamente lo que "
    "obligó a afinar el emparejamiento entre las cajas anotadas y los tokens."
)
ocr = DATA.get("eda_ocr")
if not ocr:
    pendiente("eda_ocr")
else:
    r = ocr.get("resumen", {})
    m = st.columns(3)
    m[0].metric("Páginas OCR", f"{ocr.get('n_paginas',0)}")
    m[1].metric("Tokens (mediana/pág.)", f"{r.get('tokens_mediana',0):.0f}")
    m[2].metric("Confianza (mediana)", f"{r.get('score_mediana',0):.3f}")
    cols = st.columns(3)
    tpp = ocr.get("tokens_por_pagina")
    if tpp:
        fig = px.histogram(x=tpp, nbins=30, color_discrete_sequence=[C_NEUTRO], text_auto=True)
        fig.update_traces(textposition="outside", textfont_size=9, cliponaxis=False)
        fig.update_layout(title="Tokens por página", xaxis_title="tokens", yaxis_title="páginas",
                          height=340, margin=dict(t=40, b=10))
        cols[0].plotly_chart(fig, use_container_width=True)
    for box, key, titulo, xt in ((cols[1], "score_hist", "Confianza del OCR", "score"),
                                 (cols[2], "ancho_hist", "Ancho de caja (px)", "ancho")):
        h = ocr.get(key)
        if h:
            centros = [(h["bins"][i] + h["bins"][i+1]) / 2 for i in range(len(h["counts"]))]
            etiquetas = [f"{c:,}" if c else "" for c in h["counts"]]
            fig = go.Figure(go.Bar(x=centros, y=h["counts"], marker_color=C_NEUTRO,
                                   text=etiquetas, textposition="outside", textfont_size=9,
                                   cliponaxis=False))
            fig.update_layout(title=titulo, xaxis_title=xt, yaxis_title="tokens",
                              height=340, margin=dict(t=40, b=10))
            box.plotly_chart(fig, use_container_width=True)

# =========================================================================
# 4) EDA 3 — Balance del dataset (nativo, desde eda_balance)
# =========================================================================
st.divider()
st.header("EDA 3 · Balance del dataset")
st.caption("El split por emisor (train/prueba) debe ser representativo para que las conclusiones valgan.")
analisis(
    "El conjunto se parte por emisor, 24 para entrenar y 6 para probar, unas 480 y 140 facturas respectivamente, y las "
    "proporciones de cada campo se mantienen parejas a ambos lados, de modo que la partición no mete "
    "sesgo y la prueba sobre emisores nuevos mide generalización real. El 96% de los tokens son fondo (la etiqueta O), un desbalance fuerte "
    "que más adelante justifica usar focal loss. Antes de entrenar, el emparejamiento entre cajas y "
    "tokens se depuró hasta bajar los campos perdidos de 699 a 47."
)
bal = DATA.get("eda_balance")
if not bal:
    pendiente("eda_balance")
else:
    fpe = bal.get("facturas_por_emisor", {})
    if fpe:
        em = list(fpe.keys())
        y_tr = [fpe[e].get("train", 0) for e in em]
        y_te = [fpe[e].get("test", 0) for e in em]
        fig = go.Figure()
        fig.add_bar(x=em, y=y_tr, name="train", marker_color=C_LILT,
                    text=[v or "" for v in y_tr], textposition="inside", textfont_size=9)
        fig.add_bar(x=em, y=y_te, name="test", marker_color=C_LMV3,
                    text=[v or "" for v in y_te], textposition="inside", textfont_size=9)
        fig.update_layout(barmode="stack", title="Facturas por emisor (train / test)",
                          height=380, margin=dict(t=40, b=10), xaxis_tickangle=-60)
        st.plotly_chart(fig, use_container_width=True)
    c = st.columns(2)
    pc = bal.get("proporcion_campo", {})
    if pc:
        campos = list(pc.get("train", {}).keys())
        y_tr = [pc["train"][c_] for c_ in campos]
        y_te = [pc["test"][c_] for c_ in campos]
        fig = go.Figure()
        fig.add_bar(x=campos, y=y_tr, name="train", marker_color=C_LILT,
                    text=[f"{v:.2f}" for v in y_tr], textposition="outside", textfont_size=9)
        fig.add_bar(x=campos, y=y_te, name="test", marker_color=C_LMV3,
                    text=[f"{v:.2f}" for v in y_te], textposition="outside", textfont_size=9)
        fig.update_traces(cliponaxis=False)
        fig.update_layout(barmode="group", title="Proporción de facturas con cada campo (train vs test)",
                          height=400, margin=dict(t=40, b=10), xaxis_tickangle=-40,
                          yaxis_title="proporción", yaxis_range=[0, 1.12])
        c[0].plotly_chart(fig, use_container_width=True)
    comp_t = bal.get("composicion_tokens", {})
    if comp_t:
        fig = go.Figure()
        for split, col in (("train", C_LILT), ("test", C_LMV3)):
            d = comp_t.get(split, {})
            vals = [d.get("O", 0), d.get("entidad", 0)]
            fig.add_bar(x=["O (fondo)", "entidad"], y=vals, name=split, marker_color=col,
                        text=[f"{v:,}" for v in vals], textposition="outside", textfont_size=10)
        fig.update_traces(cliponaxis=False)
        fig.update_layout(barmode="group", title="Composición de tokens: fondo vs entidad",
                          height=400, margin=dict(t=40, b=10), yaxis_title="tokens")
        c[1].plotly_chart(fig, use_container_width=True)

st.subheader("Depuración del dataset: campos perdidos")
st.caption("Antes de entrenar, el matching token - caja se refinó en varias pasadas "
           "(simétrico + rescate por colisión), reduciendo los campos perdidos.")
tray = ds.get("campos_perdidos_trayectoria", [])
if tray:
    etapas = ["Inicial", "Simétrico", "+ Rescate", "Final"][: len(tray)]
    fig = go.Figure(go.Bar(x=etapas, y=tray, marker_color=C_NEUTRO,
                           text=tray, textposition="outside"))
    fig.update_layout(title="Campos perdidos por etapa del matching",
                      yaxis_title="campos perdidos", height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

analisis(
    "El problema de raíz son los tokens fundidos, el OCR pega en un solo token el rótulo y el "
    "valor (por ejemplo NIT: 900.319.753-3, FACTURA ELECTRÓNICA DE VENTA No. "
    "FE03-01676493, y ese token queda mucho más ancho que la caja anotada del campo. Para "
    "proyectar las cajas sobre los tokens sin perder campos, el emparejamiento se refinó en tres pasos:"
    "<br><br>"
    "<b>1. Inicial (699 perdidos).</b> Se medía solo qué fracción del <i>token</i> caía dentro de la "
    "caja, con un token ancho fundido esa fracción es baja y el campo se descartaba.<br>"
    "<b>2. Solapamiento simétrico (→ 132).</b> Se pasó a tomar el máximo entre «token dentro de la "
    "caja» y «caja dentro del token», este segundo rescata los tokens fundidos, donde la caja "
    "queda casi entera dentro del token, sin relajar el umbral general.<br>"
    "<b>3. Pasada de rescate (→ 47).</b> Cuando una caja pequeña (ej. el NIT) y una grande vecina "
    "(la razón social) se disputan el mismo token, la caja que quedó sin token recupera el de mayor "
    "solapamiento, siempre que la donante conserve al menos otro. Así el NIT vuelve a su sitio y se "
    "corrige su etiqueta."
    "<br><br>"
    "El resultado es una reducción del 93% (699 → 47), y lo importante es que los 47 restantes "
    "caen todos en emisores de entrenamiento, el conjunto de prueba quedó sin ninguna pérdida, así "
    "que la evaluación no se ve afectada. Los pocos casos que persisten son tokens genuinamente fundidos, "
    "rótulo y valor inseparables, que se delegan a la limpieza posterior por reglas (regex)."
)

# =========================================================================
# 5) COMPARATIVA DE MODELOS
# =========================================================================
st.divider()
st.header("Comparativa de modelos")
analisis(
    "LiLT termina por encima de LayoutLMv3 en F1 macro (0,726 frente a 0,671) y gana en 6 de las 8 "
    "entidades. LayoutLMv3, que suma la imagen, solo saca ventaja en los campos del encabezado del "
    "emisor, donde la apariencia (la cercanía al logo, una posición fija) ayuda a ubicarlos, en los demas campos "
    "domina LiLT, el caso más llamativo es el IVA, que pasa de 0,400 a 0,783. La diferencia no "
    "es casualidad, el intervalo de confianza de la resta entre ambos no llega a tocar el cero, así que "
    "es estadísticamente significativa. En resumen, la rama visual (para el idioma español) no alcanza a "
    "justificar su costo."
)

cb = DATA.get("comparativa_base", {})
pe = cb.get("por_entidad_f1", {})
if pe:
    ents = list(pe.keys())
    y_lm = [pe[e].get("LayoutLMv3", 0) for e in ents]
    y_li = [pe[e].get("LiLT", 0) for e in ents]
    fig = go.Figure()
    fig.add_bar(x=ents, y=y_lm, name="LayoutLMv3", marker_color=C_LMV3,
                text=[f"{v:.2f}" for v in y_lm], textposition="outside", textfont_size=9)
    fig.add_bar(x=ents, y=y_li, name="LiLT", marker_color=C_LILT,
                text=[f"{v:.2f}" for v in y_li], textposition="outside", textfont_size=9)
    fig.update_traces(cliponaxis=False)
    fig.update_layout(barmode="group", title="F1 por entidad — LayoutLMv3 vs LiLT (test: emisores no vistos)",
                      height=420, margin=dict(t=40, b=10), xaxis_tickangle=-40,
                      yaxis_title="F1", yaxis_range=[0, 1.08])
    st.plotly_chart(fig, use_container_width=True)
fm = cb.get("f1_macro", {})
if fm:
    c = st.columns(2)
    c[0].metric("F1 macro — LayoutLMv3", f"{fm.get('LayoutLMv3',0):.3f}")
    c[1].metric("F1 macro — LiLT (ganador base)", f"{fm.get('LiLT',0):.3f}",
                f"+{fm.get('LiLT',0)-fm.get('LayoutLMv3',0):.3f}")

st.subheader("Los sistemas comparados (F1 micro / precisión / recall)")
sis = DATA.get("mejora_b_ensamble", {}).get("sistemas", {})
if sis:
    dfs = pd.DataFrame(sis).T.reset_index().rename(columns={"index": "sistema"})
    st.dataframe(dfs.style.format({"f1_micro": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"}),
                 use_container_width=True)
    dsort = dfs.sort_values("f1_micro")
    fig = px.bar(dsort, x="f1_micro", y="sistema", orientation="h",
                 text=dsort["f1_micro"].map(lambda v: f"{v:.3f}"),
                 color="f1_micro", color_continuous_scale="Purples", range_x=[0.6, 0.82])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(title="F1 micro por sistema", height=360, margin=dict(t=40, b=10),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Ganador base (LiLT) vs modelo elegido (ensamble focal)")
st.caption("F1 por entidad: cuánto mejora el modelo elegido sobre el mejor modelo único (LiLT base).")
lilt_pe = cb.get("por_entidad_f1", {})
ens_pe = ef.get("por_entidad", {})
if lilt_pe and ens_pe:
    ents = list(ens_pe.keys())
    y_li = [lilt_pe.get(e, {}).get("LiLT", 0) for e in ents]
    y_en = [ens_pe[e]["f1"] for e in ents]
    fig = go.Figure()
    fig.add_bar(x=ents, y=y_li, name="LiLT (ganador base)", marker_color=C_LILT,
                text=[f"{v:.2f}" for v in y_li], textposition="outside", textfont_size=9)
    fig.add_bar(x=ents, y=y_en, name="Ensamble focal (elegido)", marker_color=C_ENS,
                text=[f"{v:.2f}" for v in y_en], textposition="outside", textfont_size=9)
    fig.update_traces(cliponaxis=False)
    fig.update_layout(barmode="group", title="F1 por entidad — LiLT base vs ensamble focal",
                      height=420, margin=dict(t=40, b=10), xaxis_tickangle=-40,
                      yaxis_title="F1", yaxis_range=[0, 1.08])
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Validación estadística del modelo elegido (ensamble focal)")
st.caption("Pruebas apropiadas para una diferencia de F1: bootstrap pareado + aleatorización aproximada.")
vsu = ef.get("validacion_estadistica_vs_unico", {})
if vsu:
    rivales = list(vsu.keys())
    deltas = [vsu[r]["delta_f1_micro"] for r in rivales]
    ep = [vsu[r]["ic95"][1] - vsu[r]["delta_f1_micro"] for r in rivales]
    em = [vsu[r]["delta_f1_micro"] - vsu[r]["ic95"][0] for r in rivales]
    fig = go.Figure(go.Bar(x=rivales, y=deltas, marker_color=C_ENS,
                           text=[f"+{d:.3f}" for d in deltas], textposition="outside",
                           error_y=dict(type="data", symmetric=False, array=ep, arrayminus=em)))
    fig.update_layout(title="Ventaja del ensamble focal en F1 micro (Δ con IC 95% bootstrap)",
                      height=380, margin=dict(t=40, b=10), yaxis_title="Δ F1 micro")
    st.plotly_chart(fig, use_container_width=True)
    lineas = [f"- **vs {r}:** Δ **+{vsu[r]['delta_f1_micro']:.3f}** · IC95 "
              f"[{vsu[r]['ic95'][0]:+.3f}, {vsu[r]['ic95'][1]:+.3f}] · "
              f"p(bootstrap)={vsu[r]['p_bootstrap']:g}, p(aleatorización)={vsu[r]['p_aleatorizacion']:g}"
              for r in rivales]
    st.success("La ventaja del ensamble focal es estadísticamente significativa "
               "(los IC no cruzan 0; p < 0,01):\n\n" + "\n".join(lineas))

# =========================================================================
# 6) MODELO ELEGIDO — ENSAMBLE FOCAL
# =========================================================================
st.divider()
st.header("Modelo elegido · ensamble focal")
st.caption("Elegido por máximo rendimiento. "
           "Su ventaja viene de la precisión (menos falsos positivos), a costa de algo de recall.")
analisis(
    "El ensamble focal promedia las probabilidades de LiLT y LayoutLMv3 (ambos con focal loss) y es el "
    "sistema que mejor rinde en el presente trabajo, con ventaja significativa sobre cualquiera de los dos por "
    "separado. Su fuerza está en la precisión, al combinar dos miradas se vuelve más prudente y comete "
    "menos falsos positivos. El error que domina no es confundir un campo con otro, sino no detectarlo, "
    "y se concentra en los montos y en la razón social del emisor, justo donde la capa de reglas puede "
    "reforzar. Vale la pena notar que la rama visual de LayoutLMv3, que por sí sola no compensaba, sí "
    "aporta cuando entra como complemento."
)

pent = ef.get("por_entidad", {})
if pent:
    filas = [{"entidad": e, "métrica": m, "valor": pent[e][k_]}
             for e in pent for m, k_ in (("precisión", "precision"), ("recall", "recall"), ("F1", "f1"))]
    dpe = pd.DataFrame(filas)
    fig = px.bar(dpe, x="entidad", y="valor", color="métrica", barmode="group",
                 color_discrete_map={"precisión": C_LILT, "recall": C_LMV3, "F1": C_ENS},
                 range_y=[0, 1.12], text_auto=".2f")
    fig.update_traces(textposition="outside", textfont_size=8, cliponaxis=False)
    fig.update_layout(title="Ensamble focal — precisión / recall / F1 por entidad",
                      height=440, margin=dict(t=40, b=10), xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Nota: VALOR_TOTAL con precisión 1,000 y recall 0,475 — cuando lo extrae nunca se equivoca, "
               "pero se le escapa la mitad (perfil ideal para reforzar con reglas).")

cm = DATA.get("confusion_ensamble_focal", {})
if cm.get("matriz"):
    clases = cm["_clases"]; M = np.array(cm["matriz"], float)
    Mn = M / M.sum(axis=1, keepdims=True).clip(min=1)
    fig = px.imshow(Mn, x=[f"P:{c}" for c in clases], y=[f"V:{c}" for c in clases],
                    color_continuous_scale="Blues", zmin=0, zmax=1, text_auto=".2f", aspect="auto")
    fig.update_layout(title="Matriz de confusión (normalizada por fila = recall)",
                      height=560, margin=dict(t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Casi todo el error fuera de la diagonal cae en la columna **O** → el problema es "
               "**no-detección**, no confusión entre campos.")
te = ef.get("tipos_error", {})
if te:
    cols_te = te.get("_cols", ["ok", "no_detectado", "confundido", "falso_positivo"])
    ents = [e for e in te if e != "_cols"]
    fig = go.Figure()
    paleta = {"no_detectado": C_ALERTA, "confundido": "#E8A33D", "falso_positivo": C_ENS}
    for j, cat in enumerate(cols_te):
        if cat == "ok":
            continue
        vals = [te[e][j] for e in ents]
        fig.add_bar(x=ents, y=vals, name=cat, marker_color=paleta.get(cat, C_NEUTRO),
                    text=[v or "" for v in vals], textposition="inside", textfont_size=9)
    fig.update_layout(barmode="stack", title="Tipos de error por entidad",
                      height=440, margin=dict(t=40, b=10), xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Completitud por documento")
cpm = DATA.get("completitud_por_modelo_pct", {})
if cpm:
    s = pd.Series(cpm).sort_values()
    colores = [C_ENS if "focal" in kk.lower() and "ensamble" in kk.lower() else C_NEUTRO for kk in s.index]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colores,
                           text=[f"{v:.1f}%" for v in s.values], textposition="outside"))
    fig.update_layout(title="Ranking de completitud por sistema", height=360,
                      margin=dict(t=40, b=10), xaxis_range=[0, 80])
    st.plotly_chart(fig, use_container_width=True)

    cens = cpm.get("Ensamble focal", ef.get("completitud_doc_pct", 0))
    st.markdown(
        '<div style="background:#F1F3F5;border:1px solid #D8DEE4;border-radius:10px;'
        'padding:18px 22px;margin-top:8px;">'
        '<div style="display:flex;align-items:center;gap:28px;flex-wrap:wrap;">'
        '<div style="text-align:center;">'
        '<div style="color:#5A6B7B;font-size:0.78rem;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:4px;">Completitud - Ensamble focal</div>'
        f'<div style="color:#7B2CBF;font-size:2.8rem;font-weight:800;line-height:1;">{cens:.1f}%</div>'
        '<div style="color:#5A6B7B;font-size:0.78rem;margin-top:4px;">'
        'promedio sobre las 140<br>facturas de prueba</div>'
        '</div>'
        '<div style="flex:1;min-width:300px;color:#25333F;font-size:0.9rem;line-height:1.6;">'
        '<div style="background:#fff;border:1px solid #D8DEE4;border-radius:6px;padding:9px 13px;'
        'margin:7px 0;font-family:monospace;font-size:0.85rem;color:#1E2A38;">'
        'completitud de una factura = campos correctos / campos presentes</div>'
        'Por cada factura se cuenta cuántos de los campos que <b>realmente tiene</b> (los presentes en '
        'el <i>gold</i>) extrajo el sistema con el valor exacto. El denominador es variable, '
        'no todas las facturas traen los 8 campos (ej. una exenta no tiene IVA), en promedio son 6,71. '
        'La completitud del sistema es el promedio de esa fracción sobre las 140 facturas de prueba.'
        '<div style="color:#5A6B7B;font-size:0.82rem;margin-top:6px;">'
        'Ejemplo: una factura con 7 campos presentes de los que 5 se extraen bien → 5 / 7 = 71&nbsp;%.</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

ce = DATA.get("completitud_ensamble")
cc = st.columns(2)
if not ce:
    with cc[0]:
        pendiente("completitud_ensamble")
else:
    h = ce.get("hist")
    if h:
        centros = [(h["bins"][i] + h["bins"][i+1]) / 2 for i in range(len(h["counts"]))]
        fig = go.Figure(go.Bar(x=centros, y=h["counts"], marker_color=C_ENS,
                               text=[c or "" for c in h["counts"]], textposition="outside",
                               textfont_size=9, cliponaxis=False))
        fig.update_layout(title="Distribución de la completitud por factura (%)",
                          xaxis_title="% campos completos", yaxis_title="facturas",
                          height=360, margin=dict(t=40, b=10))
        cc[0].plotly_chart(fig, use_container_width=True)
    pem = ce.get("por_emisor_pct", {})
    if pem:
        s = pd.Series(pem).sort_values()
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=C_NEUTRO,
                               text=[f"{v:.0f}%" for v in s.values], textposition="outside"))
        fig.update_layout(title="Completitud media por emisor de test", height=360,
                          margin=dict(t=40, b=10), xaxis_range=[0, 105])
        cc[1].plotly_chart(fig, use_container_width=True)

st.subheader("Confianza (entropía)")
ent = DATA.get("entropia_por_modelo_bits", {})
if ent:
    mods = [m for m in ent if m != "_cols"]
    y_all = [ent[m][0] for m in mods]
    y_ent = [ent[m][1] for m in mods]
    fig = go.Figure()
    fig.add_bar(x=mods, y=y_all, name="todos los tokens", marker_color=C_NEUTRO,
                text=[f"{v:.3f}" for v in y_all], textposition="outside", textfont_size=9)
    fig.add_bar(x=mods, y=y_ent, name="tokens-entidad", marker_color=C_ENS,
                text=[f"{v:.3f}" for v in y_ent], textposition="outside", textfont_size=9)
    fig.update_traces(cliponaxis=False)
    fig.update_layout(barmode="group", title="Entropía predictiva media (bits; menor = más confiado)",
                      height=360, margin=dict(t=40, b=10), yaxis_title="bits")
    st.plotly_chart(fig, use_container_width=True)
    analisis(
        "<b>Por qué el ensamble focal (el modelo elegido) tiene la entropía más alta (0,167 / 0,458 bits) "
        "y aun así es el mejor.</b> "
        "La entropía cuánto reparte el modelo su probabilidad entre las opciones. "
        "Baja entropía = casi toda la probabilidad en una sola clase (predicción tajante), "
        "alta entropía = probabilidad repartida (el modelo duda)."
        "Que el ensamble dude más que sus componentes (LiLT 0,044; LayoutLMv3 0,109 en tokens-entidad) "
        "no es un defecto, sino el sello de cómo se construye, al promediar las probabilidades de dos "
        "modelos, la certeza se reparte y el ensamble solo se compromete con un campo cuando ambos lo "
        "respaldan, donde discrepan, la probabilidad se aplana y la entropía sube. Esa cautela es "
        "justamente lo que sostiene su ventaja, menos falsos positivos y la precisión global más alta del "
        "trabajo (0,777). Conviene no confundir dos ejes, un modelo individual muy confiado (baja "
        "entropía) puede estar seguro y equivocado, mientras que el ensamble prefiere dudar antes que "
        "afirmar de más. En este caso, más duda se traduce en más acierto: mejor calibrado, no peor."
    )

# =========================================================================
# 7) VALIDACION EN 2 CAPAS
# =========================================================================
st.divider()
st.header("Validación en 2 capas")
st.caption("(A) reglas deterministas + (B) cruce por CUFE con la base de referencia "
           "(existencia → coherencia campo a campo). Salida = validez por campo.")
analisis(
    "La validación se apoya en dos preguntas: si la factura existe, es decir su CUFE esté en la base de "
    "referencia <i>gold</i> y si sus campos son coherentes con lo que allí figura. Cruzando 600 facturas por CUFE, "
    "la validez media queda en 0,80. El NIT, la fecha y el número de factura se validan muy bien, "
    "entre el 92% y el 99%, las razones sociales quedan en un rango intermedio, y los montos "
    "son el punto bajo (alrededor del 22% en el total y del 17% en el IVA), esto porque "
    "muchas veces no hay con qué compararlos y en parte por diferencias de formato numérico que conviene "
    "normalizar antes de culpar al modelo. Conviene recordar que, al trabajar con datos enmascarados, "
    "esta validación es sintáctica y de coherencia."
)

vpc = DATA.get("validez_por_campo_pct", {})
if vpc:
    campos = {kk: v for kk, v in vpc.items() if isinstance(v, (int, float)) and kk != "validez_media"}
    s = pd.Series(campos).sort_values()
    colores = [C_OK if v >= 60 else C_ALERTA for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colores,
                           text=[f"{v:.1f}%" for v in s.values], textposition="outside"))
    fig.update_layout(title="Validez por campo (cruce por CUFE)", height=380,
                      margin=dict(t=40, b=10), xaxis_range=[0, 110])
    st.plotly_chart(fig, use_container_width=True)

    vm = vpc.get("validez_media", 0)
    st.markdown(
        '<div style="background:#F1F3F5;border:1px solid #D8DEE4;border-radius:10px;'
        'padding:18px 22px;margin-top:8px;">'
        '<div style="display:flex;align-items:center;gap:28px;flex-wrap:wrap;">'
        '<div style="text-align:center;">'
        '<div style="color:#5A6B7B;font-size:0.78rem;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:4px;">Validez media global</div>'
        f'<div style="color:#2A9D8F;font-size:2.8rem;font-weight:800;line-height:1;">{vm*100:.0f}%</div>'
        '<div style="color:#5A6B7B;font-size:0.78rem;margin-top:4px;">'
        'promedio sobre las 600<br>facturas cruzadas por CUFE</div>'
        '</div>'
        '<div style="flex:1;min-width:300px;color:#25333F;font-size:0.9rem;line-height:1.6;">'
        '<div style="background:#fff;border:1px solid #D8DEE4;border-radius:6px;padding:9px 13px;'
        'margin:7px 0;font-family:monospace;font-size:0.85rem;color:#1E2A38;">'
        'validez de una factura = COINCIDE / (COINCIDE + DIFIERE)</div>'
        'Por cada factura se mira cuántos de sus campos comparables, es decir, los que existen en ambos'
        'lados, coinciden con la base de referencia, los NO_COMPARABLE no entran en el denominador. '
        'La validez media global es el promedio de esa fracción sobre las 600 facturas.'
        '<div style="color:#5A6B7B;font-size:0.82rem;margin-top:6px;">'
        'Ejemplo: una factura con 7 campos comparables de los que 6 coinciden → 6 / 7 = 0,86.</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

rv = DATA.get("reglas_validacion", {})
if rv:
    st.subheader("Las dos capas de validación")
    activas = rv.get("activas", [])
    n_no = rv.get("no_aplica", 0)
    reglas_txt = " · ".join(activas) if activas else "—"
    if n_no:
        reglas_txt += (f" · _(+{n_no} reglas no aplican todavía: requieren campos "
                       "fuera del alcance actual de los 8 campos)_")
    capa_b_txt = rv.get("capa_b", "—")
    tabla = (
        "| Capa | Qué verifica | Reglas / contenido |\n"
        "|:---|:---|:---|\n"
        f"| **A · Reglas deterministas** | Coherencia interna de lo extraído, "
        f"sin mirar la base de referencia | {reglas_txt} |\n"
        f"| **B · Cruce con la base de referencia** | Coherencia de cada campo contra la fuente de "
        f"verdad, tomando el **CUFE** como llave (primero existencia, luego campo a campo) | {capa_b_txt} |\n"
    )
    st.markdown(tabla)
    st.caption("Cada campo del cruce (capa B) sale como COINCIDE, DIFIERE o NO_COMPARABLE, "
               "de ahí se deriva la validez por factura en [0, 1].")

