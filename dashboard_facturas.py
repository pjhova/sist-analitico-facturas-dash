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

st.set_page_config(page_title="Sistema Analítico de Facturas", layout="wide")


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
st.sidebar.title("Contenido")
st.sidebar.markdown("""
- [Resumen ejecutivo](#resumen-ejecutivo)
- [EDA 1 · Base de referencia](#eda-1-base-de-referencia)
- [EDA 2 · OCR](#eda-2-ocr)
- [EDA 3 · Balance del dataset](#eda-3-balance-del-dataset)
- [Comparativa de modelos](#comparativa-de-modelos)
- [Modelo elegido · ensamble focal](#modelo-elegido-ensamble-focal)
- [Validación en 2 capas](#validacion-en-2-capas)
""")
st.sidebar.caption("Página única · desplázate hacia abajo")

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

st.markdown(
    "Se compararon los modelos propuestos: "
    "LayoutLMv3 (texto+posición+imagen) y LiLT (texto+layout), en emisores no vistos "
    "LiLT ganó (F1 macro 0,726 vs 0,671). Se aplicaron las mejoras: focal loss y ensamble por promedio "
    "de probabilidades, se selecciona al ensamble focal (LiLT focal + LayoutLMv3 focal) como modelo final "
    "(F1 micro 0,755), con ventaja estadísticamente significativa sobre el mejor modelo único. "
    "La salida se valida en dos capas (reglas + cruce por CUFE con una base de referencia)."
)


# =========================================================================
# 2) EDA 1 — db_facturas.csv (nativo, desde el CSV)
# =========================================================================
st.divider()
st.header("EDA 1 · Base de referencia")
st.caption("`db_facturas.csv` — facturas reales **enmascaradas** (data masking): el CUFE queda como "
           "*placeholder* → la validación es sintáctica/coherencia, no criptográfica.")

if DB is None:
    st.warning("No se encontró `db_facturas.csv` junto al script.")
else:
    comp = (DB.replace("", np.nan).notna().mean() * 100).round(1).sort_values()
    dfc = pd.DataFrame({"columna": comp.index, "pct": comp.values})
    fig = px.bar(dfc, x="pct", y="columna", orientation="h",
                 text=dfc["pct"].map(lambda v: f"{v:.0f}%"),
                 color="pct", color_continuous_scale="Blues", range_x=[0, 105])
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
        top = DB[cciu].fillna("(vacío)").value_counts().head(15).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h", text=top.values,
                     color_discrete_sequence=[C_NEUTRO])
        fig.update_layout(title="Top-15 ciudades del adquiriente", height=430,
                          xaxis_title="facturas", yaxis_title="", margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Ver muestra de datos (primeras filas)"):
        st.dataframe(DB.head(12), use_container_width=True)

# =========================================================================
# 3) EDA 2 — OCR (nativo, desde eda_ocr)
# =========================================================================
st.divider()
st.header("EDA 2 · OCR")
st.caption("El texto que produce PaddleOCR: cuántos tokens por página, con qué confianza y qué ancho de caja.")
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
        fig = px.histogram(x=tpp, nbins=30, color_discrete_sequence=[C_NEUTRO])
        fig.update_layout(title="Tokens por página", xaxis_title="tokens", yaxis_title="páginas",
                          height=320, margin=dict(t=40, b=10))
        cols[0].plotly_chart(fig, use_container_width=True)
    for box, key, titulo, xt in ((cols[1], "score_hist", "Confianza del OCR", "score"),
                                 (cols[2], "ancho_hist", "Ancho de caja (px)", "ancho")):
        h = ocr.get(key)
        if h:
            centros = [(h["bins"][i] + h["bins"][i+1]) / 2 for i in range(len(h["counts"]))]
            fig = go.Figure(go.Bar(x=centros, y=h["counts"], marker_color=C_NEUTRO))
            fig.update_layout(title=titulo, xaxis_title=xt, yaxis_title="tokens",
                              height=320, margin=dict(t=40, b=10))
            box.plotly_chart(fig, use_container_width=True)

# =========================================================================
# 4) EDA 3 — Balance del dataset (nativo, desde eda_balance)
# =========================================================================
st.divider()
st.header("EDA 3 · Balance del dataset")
st.caption("El split **por emisor** (train/prueba) debe ser representativo para que las conclusiones valgan.")
bal = DATA.get("eda_balance")
if not bal:
    pendiente("eda_balance")
else:
    fpe = bal.get("facturas_por_emisor", {})
    if fpe:
        em = list(fpe.keys())
        fig = go.Figure()
        fig.add_bar(x=em, y=[fpe[e].get("train", 0) for e in em], name="train", marker_color=C_LILT)
        fig.add_bar(x=em, y=[fpe[e].get("test", 0) for e in em], name="test", marker_color=C_LMV3)
        fig.update_layout(barmode="stack", title="Facturas por emisor (train / test)",
                          height=380, margin=dict(t=40, b=10), xaxis_tickangle=-60)
        st.plotly_chart(fig, use_container_width=True)
    c = st.columns(2)
    pc = bal.get("proporcion_campo", {})
    if pc:
        campos = list(pc.get("train", {}).keys())
        fig = go.Figure()
        fig.add_bar(x=campos, y=[pc["train"][c_] for c_ in campos], name="train", marker_color=C_LILT)
        fig.add_bar(x=campos, y=[pc["test"][c_] for c_ in campos], name="test", marker_color=C_LMV3)
        fig.update_layout(barmode="group", title="Proporción de facturas con cada campo (train vs test)",
                          height=400, margin=dict(t=40, b=10), xaxis_tickangle=-40, yaxis_title="proporción")
        c[0].plotly_chart(fig, use_container_width=True)
    comp_t = bal.get("composicion_tokens", {})
    if comp_t:
        fig = go.Figure()
        for split, col in (("train", C_LILT), ("test", C_LMV3)):
            d = comp_t.get(split, {})
            fig.add_bar(x=["O (fondo)", "entidad"], y=[d.get("O", 0), d.get("entidad", 0)],
                        name=split, marker_color=col)
        fig.update_layout(barmode="group", title="Composición de tokens: fondo vs entidad",
                          height=400, margin=dict(t=40, b=10), yaxis_title="tokens")
        c[1].plotly_chart(fig, use_container_width=True)

st.subheader("Depuración del dataset: campos perdidos")
st.caption("Antes de entrenar, el matching token↔caja se refinó en varias pasadas "
           "(simétrico + rescate por colisión), reduciendo los campos perdidos.")
tray = ds.get("campos_perdidos_trayectoria", [])
if tray:
    etapas = ["Inicial", "Simétrico", "+ Rescate", "Final"][: len(tray)]
    fig = go.Figure(go.Bar(x=etapas, y=tray, marker_color=C_NEUTRO,
                           text=tray, textposition="outside"))
    fig.update_layout(title="Campos perdidos por etapa del matching",
                      yaxis_title="campos perdidos", height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# 5) COMPARATIVA DE MODELOS
# =========================================================================
st.divider()
st.header("Comparativa de modelos")

cb = DATA.get("comparativa_base", {})
pe = cb.get("por_entidad_f1", {})
if pe:
    ents = list(pe.keys())
    fig = go.Figure()
    fig.add_bar(x=ents, y=[pe[e].get("LayoutLMv3", 0) for e in ents], name="LayoutLMv3", marker_color=C_LMV3)
    fig.add_bar(x=ents, y=[pe[e].get("LiLT", 0) for e in ents], name="LiLT", marker_color=C_LILT)
    fig.update_layout(barmode="group", title="F1 por entidad — LayoutLMv3 vs LiLT (test: emisores no vistos)",
                      height=420, margin=dict(t=40, b=10), xaxis_tickangle=-40, yaxis_title="F1")
    st.plotly_chart(fig, use_container_width=True)
fm = cb.get("f1_macro", {})
if fm:
    c = st.columns(2)
    c[0].metric("F1 macro — LayoutLMv3", f"{fm.get('LayoutLMv3',0):.3f}")
    c[1].metric("F1 macro — LiLT (ganador base)", f"{fm.get('LiLT',0):.3f}",
                f"+{fm.get('LiLT',0)-fm.get('LayoutLMv3',0):.3f}")

st.subheader("Los sistemas comparados (F1 micro / precisión / recall)")
st.info(
    "**¿Por qué se comparan con F1 _micro_ y no _macro_?** El F1 micro mete todos los campos en "
    "una sola bolsa y calcula un acierto global. En esa bolsa, los campos que aparecen más "
    "veces influyen más en el resultado: por ejemplo, `VALOR_TOTAL` sale en casi todas las facturas, "
    "así que sus aciertos y errores cuentan mucho más que los de un campo que aparece poco. Por eso el "
    "micro refleja el rendimiento real de punta a punta. Además, es la métrica sobre la que corren "
    "las pruebas estadísticas (que remuestrean documentos y recalculan ese F1 global). "
    "En cambio, el F1 macro promedia el F1 de cada campo por igual —un campo raro pesa lo mismo "
    "que uno muy frecuente—, lo cual sirve para detectar campos débiles (por eso se usa en la "
    "comparativa base de arriba), pero para elegir el sistema y demostrar que la ventaja es real se "
    "compara el desempeño global: el micro."
)
with st.expander("Ver ejemplo numérico: F1 micro vs macro"):
    st.markdown(
        "Imagina un modelo que evalúa solo 2 campos, uno frecuente y uno raro:\n\n"
        "| Campo | Casos reales | Aciertos (TP) | Falsos + (FP) | Se escapan (FN) | Precisión | Recall | F1 |\n"
        "|---|--:|--:|--:|--:|--:|--:|--:|\n"
        "| VALOR_TOTAL (frecuente) | 100 | 90 | 10 | 10 | 0,90 | 0,90 | 0,90 |\n"
        "| IVA_TOTAL (raro) | 10 | 4 | 2 | 6 | 0,67 | 0,40 | 0,50 |\n\n"
        "**F1 macro** — promedia los F1 de cada campo *por igual*:  \n"
        "`macro = (0,90 + 0,50) / 2 = 0,70`  \n"
        "→ el IVA (0,50) pesa lo mismo que VALOR (0,90) aunque aparezca solo 10 veces, y lo arrastra hacia abajo.\n\n"
        "**F1 micro** — junta todo en una sola bolsa y calcula una sola vez:  \n"
        "- TP = 90 + 4 = 94  ·  FP = 10 + 2 = 12  ·  FN = 10 + 6 = 16  \n"
        "- Precisión = 94 / 106 = 0,89  ·  Recall = 94 / 110 = 0,85  \n"
        "- `micro = 2·0,89·0,85 / (0,89 + 0,85) = 0,87`  \n"
        "→ VALOR (100 casos) domina la bolsa; el resultado se parece a "
        "*«de todos los campos, uno por uno, ¿cuántos acerté?»*.\n\n"
        "**En resumen:** macro = 0,70 (lupa sobre los campos flojos) · "
        "micro = 0,87 (experiencia real de punta a punta)."
    )
sis = DATA.get("mejora_b_ensamble", {}).get("sistemas", {})
if sis:
    dfs = pd.DataFrame(sis).T.reset_index().rename(columns={"index": "sistema"})
    st.dataframe(dfs.style.format({"f1_micro": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"}),
                 use_container_width=True)
    dsort = dfs.sort_values("f1_micro")
    fig = px.bar(dsort, x="f1_micro", y="sistema", orientation="h",
                 text=dsort["f1_micro"].map(lambda v: f"{v:.3f}"),
                 color="f1_micro", color_continuous_scale="Purples", range_x=[0.6, 0.8])
    fig.update_layout(title="F1 micro por sistema", height=360, margin=dict(t=40, b=10),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Ganador base (LiLT) vs modelo elegido (ensamble focal)")
st.caption("F1 por entidad: cuánto mejora el modelo elegido sobre el mejor modelo único (LiLT base).")
lilt_pe = cb.get("por_entidad_f1", {})
ens_pe = ef.get("por_entidad", {})
if lilt_pe and ens_pe:
    ents = list(ens_pe.keys())
    fig = go.Figure()
    fig.add_bar(x=ents, y=[lilt_pe.get(e, {}).get("LiLT", 0) for e in ents],
                name="LiLT (ganador base)", marker_color=C_LILT)
    fig.add_bar(x=ents, y=[ens_pe[e]["f1"] for e in ents],
                name="Ensamble focal (elegido)", marker_color=C_ENS)
    fig.update_layout(barmode="group", title="F1 por entidad — LiLT base vs ensamble focal",
                      height=420, margin=dict(t=40, b=10), xaxis_tickangle=-40, yaxis_title="F1")
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
st.caption("Adoptado por **máximo rendimiento** (sin restricción de cómputo). "
           "Su ventaja viene de la **precisión** (menos falsos positivos), a costa de algo de recall.")

pent = ef.get("por_entidad", {})
if pent:
    filas = [{"entidad": e, "métrica": m, "valor": pent[e][k_]}
             for e in pent for m, k_ in (("precisión", "precision"), ("recall", "recall"), ("F1", "f1"))]
    dpe = pd.DataFrame(filas)
    fig = px.bar(dpe, x="entidad", y="valor", color="métrica", barmode="group",
                 color_discrete_map={"precisión": C_LILT, "recall": C_LMV3, "F1": C_ENS},
                 range_y=[0, 1.05])
    fig.update_layout(title="Ensamble focal — precisión / recall / F1 por entidad",
                      height=420, margin=dict(t=40, b=10), xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Nota: VALOR_TOTAL con precisión 1,000 y recall 0,475 — cuando lo extrae nunca se equivoca, "
               "pero se le escapa la mitad (perfil ideal para reforzar con reglas).")

col = st.columns(2)
cm = DATA.get("confusion_ensamble_focal", {})
if cm.get("matriz"):
    clases = cm["_clases"]; M = np.array(cm["matriz"], float)
    Mn = M / M.sum(axis=1, keepdims=True).clip(min=1)
    fig = px.imshow(Mn, x=[f"P:{c}" for c in clases], y=[f"V:{c}" for c in clases],
                    color_continuous_scale="Blues", zmin=0, zmax=1, text_auto=".2f", aspect="auto")
    fig.update_layout(title="Matriz de confusión (normalizada por fila = recall)",
                      height=520, margin=dict(t=40, b=10))
    col[0].plotly_chart(fig, use_container_width=True)
    col[0].caption("Casi todo el error fuera de la diagonal cae en la columna **O** → el problema es "
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
        fig.add_bar(x=ents, y=[te[e][j] for e in ents], name=cat, marker_color=paleta.get(cat, C_NEUTRO))
    fig.update_layout(barmode="stack", title="Tipos de error por entidad",
                      height=520, margin=dict(t=40, b=10), xaxis_tickangle=-40)
    col[1].plotly_chart(fig, use_container_width=True)

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

ce = DATA.get("completitud_ensamble")
cc = st.columns(2)
if not ce:
    with cc[0]:
        pendiente("completitud_ensamble")
else:
    h = ce.get("hist")
    if h:
        centros = [(h["bins"][i] + h["bins"][i+1]) / 2 for i in range(len(h["counts"]))]
        fig = go.Figure(go.Bar(x=centros, y=h["counts"], marker_color=C_ENS))
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
    fig = go.Figure()
    fig.add_bar(x=mods, y=[ent[m][0] for m in mods], name="todos los tokens", marker_color=C_NEUTRO)
    fig.add_bar(x=mods, y=[ent[m][1] for m in mods], name="tokens-entidad", marker_color=C_ENS)
    fig.update_layout(barmode="group", title="Entropía predictiva media (bits; menor = más confiado)",
                      height=360, margin=dict(t=40, b=10), yaxis_title="bits")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**Por qué el ensamble focal —el modelo elegido— tiene la entropía más alta (0,167 / 0,458 bits) "
        "y aun así es el mejor.** "
        "La entropía mide cuánto *duda* el modelo: menor entropía = predicciones más tajantes. "
        "Que el ensamble dude más que sus componentes (LiLT 0,044; LayoutLMv3 0,109 en tokens-entidad) "
        "no es un defecto, sino el sello de cómo se construye: al **promediar** las probabilidades de dos "
        "modelos, la certeza se reparte y el ensamble solo se compromete con un campo cuando *ambos* lo "
        "respaldan; donde discrepan, la probabilidad se aplana y la entropía sube. Esa cautela es "
        "justamente lo que sostiene su ventaja: menos falsos positivos y la precisión global más alta del "
        "estudio (0,777). Conviene además no confundir dos ejes: un modelo individual muy confiado (baja "
        "entropía) puede estar seguro *y equivocado*, mientras que el ensamble prefiere dudar antes que "
        "afirmar de más. En este caso, más duda se traduce en más acierto: mejor calibrado, no peor."
    )

# =========================================================================
# 7) VALIDACION EN 2 CAPAS
# =========================================================================
st.divider()
st.header("Validación en 2 capas")
st.caption("(A) reglas deterministas + (B) cruce por CUFE con la base de referencia "
           "(existencia → coherencia campo a campo). Salida = validez por campo, sin veredicto.")

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
    st.metric("Validez media", f"{vpc.get('validez_media',0)*100:.0f}%")

rv = DATA.get("reglas_validacion", {})
if rv:
    c = st.columns(2)
    with c[0]:
        st.markdown("**Capa A — reglas deterministas**")
        for r in rv.get("activas", []):
            st.markdown(f"- {r}")
    with c[1]:
        st.markdown("**Capa B — cruce con base de referencia**")
        st.markdown(rv.get("capa_b", ""))

