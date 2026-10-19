"""
=============================================================================
PROSEGUR - SISTEMA PREDICTIVO DE NOTAS DE CRÉDITO (NC MES SIGUIENTE)
Modelo: Random Forest + class_weight (v2) con Calibración Isotónica
Framework: Streamlit
Identidad Visual: Amarillo (#FFD400) & Negro (#000000 / #141414)
=============================================================================
"""

import os
import sys
import warnings
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# 0. PARCHE DE COMPATIBILIDAD CRÍTICO PARA SCIKIT-LEARN / IMBLEARN
# ---------------------------------------------------------------------------
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

import joblib

warnings.filterwarnings("ignore")

# Rutas de recursos
LOGO_ICON_PATH = os.path.join("recursos_visuales", "logo.png")
LOGO_HEADER_PATH = os.path.join("recursos_visuales", "logo_texto.jpg")
MODEL_PATH = os.path.join("modelo_nc", "modelo_nc_mes_siguiente_v2.joblib")
IMPORTANCIA_PATH = os.path.join("modelo_nc", "importancia_variables_v2.csv")


# ---------------------------------------------------------------------------
# 1. FUNCIONES PURAS Y DE DATOS (IMPORTABLES SIN EFECTOS SECUNDARIOS)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Carga de forma segura el paquete del modelo entrenado y calibrado."""
    if not os.path.exists(path):
        return None, f"No se encontró el archivo del modelo en la ruta: '{path}'."
    try:
        pkg = joblib.load(path)
        req_keys = ['modelo', 'calibrador', 'umbral', 'features', 'num', 'cat']
        for k in req_keys:
            if k not in pkg:
                return None, f"El archivo joblib carece de la clave requerida: '{k}'."
        return pkg, None
    except Exception as e:
        return None, f"Error al deserializar el modelo: {str(e)}"


@st.cache_data(show_spinner=False)
def load_feature_importance(path: str):
    """Carga el archivo CSV de importancia por permutación si existe."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def predict_dataframe(df_input: pd.DataFrame, pkg: dict) -> pd.DataFrame:
    """
    Aplica el pipeline de preprocesamiento, clasificador y calibrador isotónico
    sobre un dataframe de entrada con validación robusta de columnas.
    """
    df = df_input.copy()
    features = pkg['features']
    num_cols = pkg['num']
    cat_cols = pkg['cat']
    umbral = pkg['umbral']

    for col in num_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    for col in cat_cols:
        if col not in df.columns:
            df[col] = 'SIN DATO'
        else:
            df[col] = df[col].astype(str).str.strip().str.upper().replace(
                {'': 'SIN DATO', 'NAN': 'SIN DATO', '<NA>': 'SIN DATO', 'NONE': 'SIN DATO'}
            )

    if 'clasificacion' in df.columns:
        df['clasificacion'] = df['clasificacion'].where(
            df['clasificacion'].isin(['NORMAL', 'ESTRATEGICO', 'PREFERENTE', 'VIP']), 'OTRO'
        )

    X = df[features]

    prob_bruta = pkg['modelo'].predict_proba(X)[:, 1]
    prob_calibrada = pkg['calibrador'].predict(prob_bruta)
    prob_calibrada = np.clip(prob_calibrada, 0.0, 1.0)

    df_out = df_input.copy()
    df_out['prob_nc_mes_siguiente'] = prob_calibrada
    df_out['prob_nc_porcentaje'] = (prob_calibrada * 100).round(2)
    df_out['alerta'] = (prob_calibrada >= umbral).astype(int)
    
    def categorizar_riesgo(p):
        if p >= umbral:
            return "ALTO RIESGO"
        elif p >= umbral / 2.0:
            return "RIESGO MEDIO"
        else:
            return "BAJO RIESGO"

    df_out['nivel_riesgo'] = [categorizar_riesgo(p) for p in prob_calibrada]
    return df_out


def crear_tacometro_riesgo(prob: float, umbral: float):
    """Genera un Gauge Chart estilizado de Plotly con la paleta de Prosegur."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%", 'font': {'size': 44, 'color': '#FFFFFF'}},
        title={'text': "Probabilidad Calibrada de Nota de Crédito", 'font': {'size': 18, 'color': '#FFD400'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#777777", 'tickfont': {'color': '#AAAAAA'}},
            'bar': {'color': "#FFD400", 'thickness': 0.3},
            'bgcolor': "#1B1B1B",
            'borderwidth': 2,
            'bordercolor': "#333333",
            'steps': [
                {'range': [0, (umbral / 2) * 100], 'color': "rgba(0, 230, 118, 0.25)"},
                {'range': [(umbral / 2) * 100, umbral * 100], 'color': "rgba(255, 212, 0, 0.3)"},
                {'range': [umbral * 100, 100], 'color': "rgba(255, 75, 75, 0.35)"}
            ],
            'threshold': {
                'line': {'color': "#FF4B4B", 'width': 4},
                'thickness': 0.85,
                'value': umbral * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=25, r=25, t=50, b=20)
    )
    return fig


# ---------------------------------------------------------------------------
# 2. APLICACIÓN PRINCIPAL (UI STREAMLIT)
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Prosegur | Predictor de Notas de Crédito",
        page_icon=LOGO_ICON_PATH if os.path.exists(LOGO_ICON_PATH) else "🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .main {
            background-color: #0A0A0A;
            color: #FFFFFF;
        }
        .prosegur-header {
            background: linear-gradient(90deg, #161616 0%, #202020 100%);
            border-bottom: 3px solid #FFD400;
            padding: 1.2rem 1.8rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .prosegur-title {
            color: #FFFFFF;
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .prosegur-subtitle {
            color: #FFD400;
            font-size: 0.95rem;
            font-weight: 600;
            margin: 0;
            margin-top: 4px;
        }
        .metric-card {
            background: #141414;
            border: 1px solid #2B2B2B;
            border-left: 4px solid #FFD400;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        .metric-label {
            font-size: 0.85rem;
            color: #AAAAAA;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-top: 4px;
        }
        .veredicto-alerta {
            background: rgba(255, 75, 75, 0.12);
            border: 2px solid #FF4B4B;
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
        }
        .veredicto-normal {
            background: rgba(0, 230, 118, 0.12);
            border: 2px solid #00E676;
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
        }
        .veredicto-medio {
            background: rgba(255, 212, 0, 0.12);
            border: 2px solid #FFD400;
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 2px solid #2B2B2B;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #141414;
            color: #CCCCCC;
            border-radius: 6px 6px 0px 0px;
            padding: 8px 18px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFD400 !important;
            color: #000000 !important;
            font-weight: 700 !important;
        }
        div.stButton > button:first-child {
            background-color: #FFD400;
            color: #000000;
            font-weight: 700;
            border: none;
            border-radius: 6px;
            padding: 0.55rem 1.4rem;
            transition: all 0.2s ease-in-out;
        }
        div.stButton > button:first-child:hover {
            background-color: #FFE033;
            color: #000000;
            box-shadow: 0 4px 14px rgba(255, 212, 0, 0.4);
            transform: translateY(-1px);
        }
        [data-testid="stSidebar"] {
            background-color: #0F0F0F;
            border-right: 1px solid #222222;
        }
    </style>
    """, unsafe_allow_html=True)

    model_pkg, model_err = load_model(MODEL_PATH)
    df_importancia = load_feature_importance(IMPORTANCIA_PATH)

    with st.sidebar:
        if os.path.exists(LOGO_HEADER_PATH):
            st.image(LOGO_HEADER_PATH, use_container_width=True)
        elif os.path.exists(LOGO_ICON_PATH):
            st.image(LOGO_ICON_PATH, width=120)
        
        st.markdown("### 🛡️ **Centro de Control Analítico**")
        st.markdown(
            "<div style='font-size: 0.85rem; color: #BBB; line-height: 1.4;'>"
            "Predicción inteligente del riesgo de emisión de <b>Notas de Crédito (NC)</b> "
            "en el mes siguiente (t+1) para gestión proactiva de clientes."
            "</div>",
            unsafe_allow_html=True
        )
        st.divider()

        if model_pkg:
            st.markdown("#### ⚙️ **Parámetros del Modelo (v2)**")
            umbral_val = float(model_pkg.get('umbral', 0.0853))
            st.markdown(f"**Algoritmo:** `Random Forest + Calibración`")
            st.markdown(f"**Umbral Decisión:** `{umbral_val:.4f}` ({umbral_val*100:.2f}%)")
            st.markdown(f"**Variables Totales:** `{len(model_pkg.get('features', []))}`")
            
            st.markdown(
                f"""
                <div class="metric-card" style="margin-top: 15px;">
                    <div class="metric-label">Captura en Top 10%</div>
                    <div class="metric-value" style="color: #FFD400;">46.2%</div>
                    <div style="font-size: 0.75rem; color: #AAA; margin-top: 4px;">
                        Revisando el 10% de clientes con mayor score se previene el 46.2% de casos y 44.8% del monto.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("⚠️ Modelo no disponible en disco.")

        st.divider()
        st.caption("Prosegur Perú • Gerencia de Analítica de Negocios • 2026")

    st.markdown(
        """
        <div class="prosegur-header">
            <div>
                <h1 class="prosegur-title">Sistema de Alerta Temprana de Notas de Crédito</h1>
                <p class="prosegur-subtitle">PROSEGUR • MODELO PREDICTIVO MES SIGUIENTE (t+1)</p>
            </div>
            <div style="text-align: right;">
                <span style="background: #000; color: #FFD400; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #FFD400;">
                    PRODUCCIÓN v2.0
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if model_err:
        st.error(f"❌ Error crítico de inicialización: {model_err}")
        st.stop()

    tab_individual, tab_masiva, tab_analytics = st.tabs([
        "👤 Evaluación Individual", 
        "📁 Evaluación Masiva (Lote)", 
        "📈 Desempeño y Explicabilidad"
    ])

    # =======================================================================
    # PESTAÑA 1: EVALUACIÓN INDIVIDUAL
    # =======================================================================
    with tab_individual:
        st.markdown("#### Ingrese los datos operativos y comerciales del cliente")
        
        PRESETS = {
            "alto": {
                "cod_cliente": "CLI-ALTO-999",
                "delegacion": "DELEGACION LIMA",
                "clasificacion": "NORMAL",
                "transaccion": "Recurrente",
                "tipo_servicio": "ALARMAS",
                "motivo_frec": "DESCUENTO POR ITEM",
                "area_frec": "ATC",
                "mes_objetivo": "11",
                "meses_desde_ultima": 0.0,
                "antiguedad_meses": 24.0,
                "frac_meses_con_nc": 0.65,
                "meses_con_nc_6m": 5,
                "n_nc_1m": 3,
                "n_nc_3m": 6,
                "n_nc_6m": 9,
                "n_nc_12m": 14,
                "n_nc_hist": 25,
                "nc_lag1": 1,
                "nc_lag2": 1,
                "nc_lag3": 1,
                "nc_lag5": 1,
                "nc_lag11": 0,
                "tendencia_3v3": 3,
                "monto_1m": 2100.0,
                "monto_3m": 4800.0,
                "monto_6m": 7200.0,
                "monto_hist": 19500.0,
                "monto_prom_nota": 780.0
            },
            "saludable": {
                "cod_cliente": "CLI-SALUD-001",
                "delegacion": "DELEGACION AREQUIPA",
                "clasificacion": "ESTRATEGICO",
                "transaccion": "Recurrente",
                "tipo_servicio": "ALARMAS",
                "motivo_frec": "DISMINUCIÓN EN EL VALOR",
                "area_frec": "COBRANZAS",
                "mes_objetivo": "11",
                "meses_desde_ultima": 18.0,
                "antiguedad_meses": 36.0,
                "frac_meses_con_nc": 0.03,
                "meses_con_nc_6m": 0,
                "n_nc_1m": 0,
                "n_nc_3m": 0,
                "n_nc_6m": 0,
                "n_nc_12m": 0,
                "n_nc_hist": 1,
                "nc_lag1": 0,
                "nc_lag2": 0,
                "nc_lag3": 0,
                "nc_lag5": 0,
                "nc_lag11": 0,
                "tendencia_3v3": 0,
                "monto_1m": 0.0,
                "monto_3m": 0.0,
                "monto_6m": 0.0,
                "monto_hist": 200.0,
                "monto_prom_nota": 200.0
            },
            "promedio": {
                "cod_cliente": "CLI-PROM-105",
                "delegacion": "DELEGACION LIMA",
                "clasificacion": "NORMAL",
                "transaccion": "Recurrente",
                "tipo_servicio": "ALARMAS",
                "motivo_frec": "DESCUENTO POR ITEM",
                "area_frec": "ATC",
                "mes_objetivo": "11",
                "meses_desde_ultima": 3.0,
                "antiguedad_meses": 18.0,
                "frac_meses_con_nc": 0.22,
                "meses_con_nc_6m": 1,
                "n_nc_1m": 0,
                "n_nc_3m": 1,
                "n_nc_6m": 2,
                "n_nc_12m": 4,
                "n_nc_hist": 7,
                "nc_lag1": 0,
                "nc_lag2": 1,
                "nc_lag3": 0,
                "nc_lag5": 0,
                "nc_lag11": 0,
                "tendencia_3v3": 0,
                "monto_1m": 0.0,
                "monto_3m": 350.0,
                "monto_6m": 890.0,
                "monto_hist": 3200.0,
                "monto_prom_nota": 457.14
            }
        }

        def aplicar_preset(nombre_preset):
            datos = PRESETS[nombre_preset]
            for campo, valor in datos.items():
                st.session_state[f"inp_{campo}"] = valor
            st.session_state["preset_nombre"] = nombre_preset
            st.session_state["ejecutar_al_cargar"] = True

        if "form_initialized" not in st.session_state:
            for campo, valor in PRESETS["promedio"].items():
                st.session_state[f"inp_{campo}"] = valor
            st.session_state["form_initialized"] = True
            st.session_state["preset_nombre"] = "promedio"
            st.session_state["ejecutar_al_cargar"] = False

        st.markdown("**Perfiles Rápidos de Prueba:**")
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.button("🔥 Cargar Cliente de Alto Riesgo", on_click=aplicar_preset, args=("alto",), use_container_width=True)
        col_p2.button("📊 Cargar Cliente Promedio", on_click=aplicar_preset, args=("promedio",), use_container_width=True)
        col_p3.button("🛡️ Cargar Cliente Saludable", on_click=aplicar_preset, args=("saludable",), use_container_width=True)

        nombre_actual = st.session_state.get("preset_nombre", "personalizado")
        map_nombres = {
            "alto": "🔥 Cliente de Alto Riesgo",
            "promedio": "📊 Cliente Promedio",
            "saludable": "🛡️ Cliente Saludable",
            "personalizado": "✏️ Personalizado"
        }
        st.caption(f"Perfil actualmente activo en el formulario: **{map_nombres.get(nombre_actual, 'Personalizado')}**")

        delegaciones_list = [
            'DELEGACION LIMA', 'DELEGACION AREQUIPA', 'DELEGACION TRUJILLO',
            'DELEGACION PIURA', 'DELEGACION CHICLAYO', 'DELEGACION ICA',
            'DELEGACION CUSCO', 'DELEGACION HUANCAYO', 'DELEGACION TELEVENTAS', 'OTRA'
        ]
        clasificaciones_list = ['NORMAL', 'ESTRATEGICO', 'PREFERENTE', 'VIP', 'OTRO']
        servicios_list = ['ALARMAS', 'GPS', 'OJO DEL HALCON', 'NO GIRO', 'OTRO']
        transaccion_list = ['Recurrente', 'Adicionales', 'No Giro', 'Ampliacion', 'Alta', 'OTRO']
        motivos_list = [
            'DESCUENTO POR ITEM', 'DISMINUCIÓN EN EL VALOR', 'ANULACIÓN DE LA OPERACIÓN',
            'DESCUENTO GLOBAL', 'ANULACIÓN POR ERROR EN EL RUC', 'DEVOLUCIÓN TOTAL', 'OTRO'
        ]
        areas_list = ['ATC', 'COBRANZAS', 'RETER', 'CUENTAS NUEVAS', 'COMERCIAL', 'FACTURACION', 'OTRO']
        meses_obj_list = [str(i) for i in range(1, 13)]

        with st.form("form_prediccion_individual"):
            c_id1, c_id2 = st.columns([1, 2])
            c_id1.text_input("Código de Cliente:", key="inp_cod_cliente")
            c_id2.caption("Identificador único del cliente en Prosegur (para trazabilidad en reportes).")

            with st.expander("🏢 1. Datos Comerciales y Operativos", expanded=True):
                col_c1, col_c2, col_c3 = st.columns(3)
                col_c1.selectbox("Delegación:", delegaciones_list, key="inp_delegacion")
                col_c2.selectbox("Clasificación Comercial:", clasificaciones_list, key="inp_clasificacion")
                col_c3.selectbox("Tipo de Servicio:", servicios_list, key="inp_tipo_servicio")

                col_c4, col_c5, col_c6, col_c7 = st.columns(4)
                col_c4.selectbox("Transacción:", transaccion_list, key="inp_transaccion")
                col_c5.selectbox("Motivo Frecuente:", motivos_list, key="inp_motivo_frec")
                col_c6.selectbox("Área Frecuente:", areas_list, key="inp_area_frec")
                col_c7.selectbox("Mes a Predecir (1-12):", meses_obj_list, key="inp_mes_objetivo")

            with st.expander("⏳ 2. Historial de Recencia y Antigüedad", expanded=True):
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                col_r1.number_input(
                    "Meses desde Última NC:", min_value=0.0, max_value=120.0, step=1.0, key="inp_meses_desde_ultima",
                    help="Variable predictiva #1: meses transcurridos desde que se le emitió la última NC."
                )
                col_r2.number_input(
                    "Antigüedad (meses):", min_value=0.0, max_value=240.0, step=1.0, key="inp_antiguedad_meses",
                    help="Meses desde la primera nota de crédito registrada del cliente."
                )
                col_r3.number_input(
                    "Fracción Meses con NC:", min_value=0.0, max_value=1.0, step=0.01, key="inp_frac_meses_con_nc",
                    help="Proporción histórica de meses con NC sobre la antigüedad total."
                )
                col_r4.number_input(
                    "Meses con NC en últimos 6m:", min_value=0, max_value=6, step=1, key="inp_meses_con_nc_6m"
                )

            with st.expander("🔢 3. Conteo de Notas de Crédito y Estacionalidad", expanded=False):
                col_n1, col_n2, col_n3, col_n4, col_n5 = st.columns(5)
                col_n1.number_input("NC último mes (1m):", min_value=0, max_value=200, key="inp_n_nc_1m")
                col_n2.number_input("NC últimos 3 meses:", min_value=0, max_value=400, key="inp_n_nc_3m")
                col_n3.number_input("NC últimos 6 meses:", min_value=0, max_value=600, key="inp_n_nc_6m")
                col_n4.number_input("NC últimos 12 meses:", min_value=0, max_value=1200, key="inp_n_nc_12m")
                col_n5.number_input("NC Histórico Total:", min_value=0, max_value=5000, key="inp_n_nc_hist")

                col_l1, col_l2, col_l3, col_l4, col_l5, col_l6 = st.columns(6)
                col_l1.selectbox("NC hace 1 mes:", [0, 1], key="inp_nc_lag1")
                col_l2.selectbox("NC hace 2 meses:", [0, 1], key="inp_nc_lag2")
                col_l3.selectbox("NC hace 3 meses:", [0, 1], key="inp_nc_lag3")
                col_l4.selectbox("NC hace 5 meses:", [0, 1], key="inp_nc_lag5")
                col_l5.selectbox("NC hace 11 meses:", [0, 1], key="inp_nc_lag11", help="Mismo mes del año pasado.")
                col_l6.number_input("Tendencia (3m vs ant):", min_value=-50, max_value=50, key="inp_tendencia_3v3")

            with st.expander("💰 4. Montos de Notas de Crédito (S/)", expanded=False):
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                col_m1.number_input("Monto último mes (S/):", min_value=0.0, step=50.0, key="inp_monto_1m")
                col_m2.number_input("Monto últimos 3m (S/):", min_value=0.0, step=100.0, key="inp_monto_3m")
                col_m3.number_input("Monto últimos 6m (S/):", min_value=0.0, step=100.0, key="inp_monto_6m")
                col_m4.number_input("Monto Histórico Total (S/):", min_value=0.0, step=500.0, key="inp_monto_hist")
                col_m5.number_input("Monto Promedio por NC (S/):", min_value=0.0, step=20.0, key="inp_monto_prom_nota")

            submit_btn = st.form_submit_button("⚡ EJECUTAR PREDICCIÓN", use_container_width=True)

        debe_ejecutar = submit_btn or st.session_state.get("ejecutar_al_cargar", False)

        if debe_ejecutar:
            dict_row = {
                'cliente': st.session_state["inp_cod_cliente"],
                'n_nc_1m': st.session_state["inp_n_nc_1m"],
                'n_nc_3m': st.session_state["inp_n_nc_3m"],
                'n_nc_6m': st.session_state["inp_n_nc_6m"],
                'n_nc_hist': st.session_state["inp_n_nc_hist"],
                'monto_1m': st.session_state["inp_monto_1m"],
                'monto_3m': st.session_state["inp_monto_3m"],
                'monto_6m': st.session_state["inp_monto_6m"],
                'monto_hist': st.session_state["inp_monto_hist"],
                'meses_con_nc_6m': st.session_state["inp_meses_con_nc_6m"],
                'meses_desde_ultima': st.session_state["inp_meses_desde_ultima"],
                'antiguedad_meses': st.session_state["inp_antiguedad_meses"],
                'monto_prom_nota': st.session_state["inp_monto_prom_nota"],
                'nc_lag1': st.session_state["inp_nc_lag1"],
                'nc_lag2': st.session_state["inp_nc_lag2"],
                'nc_lag3': st.session_state["inp_nc_lag3"],
                'nc_lag5': st.session_state["inp_nc_lag5"],
                'nc_lag11': st.session_state["inp_nc_lag11"],
                'n_nc_12m': st.session_state["inp_n_nc_12m"],
                'frac_meses_con_nc': st.session_state["inp_frac_meses_con_nc"],
                'tendencia_3v3': st.session_state["inp_tendencia_3v3"],
                'delegacion': st.session_state["inp_delegacion"],
                'clasificacion': st.session_state["inp_clasificacion"],
                'transaccion': st.session_state["inp_transaccion"],
                'tipo_servicio': st.session_state["inp_tipo_servicio"],
                'motivo_frec': st.session_state["inp_motivo_frec"],
                'area_frec': st.session_state["inp_area_frec"],
                'mes_objetivo': str(st.session_state["inp_mes_objetivo"])
            }

            st.session_state["ejecutar_al_cargar"] = False

            df_single = pd.DataFrame([dict_row])
            res = predict_dataframe(df_single, model_pkg).iloc[0]
            prob_val = res['prob_nc_mes_siguiente']
            umbral_val = model_pkg['umbral']

            st.markdown("### 📊 Resultado de la Evaluación")
            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                st.plotly_chart(crear_tacometro_riesgo(prob_val, umbral_val), use_container_width=True)

            with col_res2:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                if prob_val >= umbral_val:
                    st.markdown(
                        f"""
                        <div class="veredicto-alerta">
                            <h2 style="color: #FF4B4B; margin: 0; font-size: 1.6rem;">🔴 ALERTA: ALTO RIESGO DE NC</h2>
                            <p style="color: #FFFFFF; font-size: 1.1rem; margin-top: 8px;">
                                Probabilidad Calibrada: <b style="color: #FFD400;">{prob_val*100:.2f}%</b> (Umbral: {umbral_val*100:.2f}%)
                            </p>
                            <p style="color: #DDD; font-size: 0.9rem; text-align: justify; margin: 0;">
                                <b>Acción Recomendada:</b> El cliente <b>{dict_row['cliente']}</b> presenta un comportamiento estadístico recurrente 
                                asociado a emisión de Notas de Crédito. Se sugiere contactar preventivamente al cliente, verificar la 
                                facturación del mes anterior con ATC/Cobranzas y validar la exactitud de los cargos antes del cierre contable.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif prob_val >= umbral_val / 2.0:
                    st.markdown(
                        f"""
                        <div class="veredicto-medio">
                            <h2 style="color: #FFD400; margin: 0; font-size: 1.6rem;">🟡 RIESGO MODERADO</h2>
                            <p style="color: #FFFFFF; font-size: 1.1rem; margin-top: 8px;">
                                Probabilidad Calibrada: <b style="color: #FFD400;">{prob_val*100:.2f}%</b> (Umbral: {umbral_val*100:.2f}%)
                            </p>
                            <p style="color: #DDD; font-size: 0.9rem; text-align: justify; margin: 0;">
                                <b>Acción Recomendada:</b> El cliente presenta señales leves de recurrencia en montos o transacciones. 
                                Mantener en observación para el siguiente ciclo y monitorear ajustes menores en facturación.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="veredicto-normal">
                            <h2 style="color: #00E676; margin: 0; font-size: 1.6rem;">🟢 BAJO RIESGO / SALUDABLE</h2>
                            <p style="color: #FFFFFF; font-size: 1.1rem; margin-top: 8px;">
                                Probabilidad Calibrada: <b style="color: #00E676;">{prob_val*100:.2f}%</b> (Umbral: {umbral_val*100:.2f}%)
                            </p>
                            <p style="color: #DDD; font-size: 0.9rem; text-align: justify; margin: 0;">
                                <b>Acción Recomendada:</b> La probabilidad estimada está muy por debajo del umbral de alerta. 
                                El cliente mantiene estabilidad operativa y no requiere acciones preventivas extraordinarias.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("#### 🔍 Comparativa del Cliente vs. Cartera Típica")
            col_bench1, col_bench2, col_bench3, col_bench4 = st.columns(4)
            col_bench1.metric("Meses desde última NC", f"{dict_row['meses_desde_ultima']:.0f} m", delta=f"{dict_row['meses_desde_ultima'] - 4.5:.1f} m vs promedio", delta_color="inverse")
            col_bench2.metric("NC en últimos 3m", f"{dict_row['n_nc_3m']}", delta=f"{dict_row['n_nc_3m'] - 0.8:.1f} vs promedio", delta_color="inverse")
            col_bench3.metric("Monto en 6m", f"S/ {dict_row['monto_6m']:,.2f}", delta=f"S/ {dict_row['monto_6m'] - 850:,.2f} vs prom.", delta_color="inverse")
            col_bench4.metric("Fracción Meses con NC", f"{dict_row['frac_meses_con_nc']*100:.1f}%", delta=f"{(dict_row['frac_meses_con_nc'] - 0.20)*100:.1f}% vs prom.", delta_color="inverse")

    # =======================================================================
    # PESTAÑA 2: EVALUACIÓN MASIVA (BATCH)
    # =======================================================================
    with tab_masiva:
        st.markdown("#### Procesamiento en lote para calificar carteras masivas de clientes")
        st.markdown(
            "Cargue un archivo en formato **CSV o Excel (.xlsx)** con los registros mensuales de los clientes. "
            "El sistema validará las columnas, imputará automáticamente campos no críticos y generará el ranking completo de riesgo."
        )

        col_btn_temp, col_info_temp = st.columns([1, 3])
        with col_btn_temp:
            if os.path.exists("template_evaluacion_masiva.csv"):
                with open("template_evaluacion_masiva.csv", "rb") as f:
                    st.download_button(
                        label="📥 Descargar Plantilla CSV",
                        data=f.read(),
                        file_name="plantilla_clientes_prosegur.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        with col_info_temp:
            st.caption("ℹ️ La plantilla incluye las 27 características requeridas más el campo identificador `cliente`.")

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Seleccionar archivo de datos masivos:",
            type=["csv", "xlsx", "xls"],
            help="Cargue un archivo CSV o Excel con las variables calculadas al mes t."
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_batch_raw = pd.read_csv(uploaded_file)
                else:
                    df_batch_raw = pd.read_excel(uploaded_file)

                st.success(f"✅ Archivo cargado correctamente: **{uploaded_file.name}** ({len(df_batch_raw):,} registros cargados)")

                with st.expander("👀 Vista previa de los datos cargados", expanded=False):
                    st.dataframe(df_batch_raw.head(10), use_container_width=True)

                if st.button("🚀 PROCESAR Y CALIFICAR CARTERA", use_container_width=True):
                    with st.spinner("Ejecutando modelo predictivo y calibrador isotónico..."):
                        df_res_batch = predict_dataframe(df_batch_raw, model_pkg)
                        df_res_batch = df_res_batch.sort_values(by="prob_nc_mes_siguiente", ascending=False).reset_index(drop=True)

                    st.markdown("### 🏆 Resumen Global de la Cartera Evaluada")
                    total_clientes = len(df_res_batch)
                    total_alertas = int(df_res_batch['alerta'].sum())
                    tasa_alertas = (total_alertas / total_clientes) * 100 if total_clientes > 0 else 0.0
                    monto_riesgo = df_res_batch.loc[df_res_batch['alerta'] == 1, 'monto_6m'].sum() if 'monto_6m' in df_res_batch.columns else 0.0

                    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                    c_m1.metric("Total Clientes Evaluados", f"{total_clientes:,}")
                    c_m2.metric("Clientes en Alerta", f"{total_alertas:,}", delta=f"{tasa_alertas:.1f}% cartera", delta_color="inverse")
                    c_m3.metric("Umbral de Decisión", f"{model_pkg['umbral']*100:.2f}%")
                    c_m4.metric("Monto 6m en Riesgo", f"S/ {monto_riesgo:,.2f}")

                    col_g1, col_g2 = st.columns([1, 1])
                    with col_g1:
                        fig_dist = px.histogram(
                            df_res_batch,
                            x="prob_nc_porcentaje",
                            nbins=30,
                            title="Distribución de Probabilidad de Riesgo en la Cartera",
                            labels={'prob_nc_porcentaje': 'Probabilidad (%)'},
                            color_discrete_sequence=['#FFD400']
                        )
                        fig_dist.add_vline(
                            x=model_pkg['umbral'] * 100,
                            line_width=3,
                            line_dash="dash",
                            line_color="#FF4B4B",
                            annotation_text=f"Umbral ({model_pkg['umbral']*100:.1f}%)"
                        )
                        fig_dist.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='#141414',
                            font=dict(color="#FFFFFF")
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

                    with col_g2:
                        fig_pie = px.pie(
                            df_res_batch,
                            names="nivel_riesgo",
                            title="Composición de la Cartera por Nivel de Riesgo",
                            color="nivel_riesgo",
                            color_discrete_map={
                                "ALTO RIESGO": "#FF4B4B",
                                "RIESGO MEDIO": "#FFD400",
                                "BAJO RIESGO": "#00E676"
                            }
                        )
                        fig_pie.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color="#FFFFFF")
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    st.markdown("#### 📋 Ranking de Clientes en Riesgo (Ordenado Descendente)")
                    
                    vis_cols = ['cliente', 'prob_nc_porcentaje', 'alerta', 'nivel_riesgo']
                    extra_cols = ['n_nc_1m', 'n_nc_3m', 'n_nc_12m', 'monto_6m', 'meses_desde_ultima', 'delegacion', 'clasificacion', 'motivo_frec']
                    final_cols = [c for c in vis_cols if c in df_res_batch.columns] + [c for c in extra_cols if c in df_res_batch.columns]

                    st.dataframe(
                        df_res_batch[final_cols].style.format({
                            'prob_nc_porcentaje': '{:.2f}%',
                            'monto_6m': 'S/ {:,.2f}' if 'monto_6m' in df_res_batch.columns else '{}',
                            'meses_desde_ultima': '{:.0f}' if 'meses_desde_ultima' in df_res_batch.columns else '{}'
                        }),
                        use_container_width=True,
                        height=400
                    )

                    csv_buffer = io.StringIO()
                    df_res_batch.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 DESCARGAR REPORTE COMPLETO DE PREDICCIONES (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="reporte_prediccion_nc_prosegur.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {str(e)}")

    # =======================================================================
    # PESTAÑA 3: DESEMPEÑO Y EXPLICABILIDAD
    # =======================================================================
    with tab_analytics:
        st.markdown("#### Diagnóstico Analítico y Validación del Modelo en Producción")
        
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        metricas = model_pkg.get('metricas_test_2026', {})
        col_d1.metric("AUC-ROC (Prueba 2026)", f"{metricas.get('AUC_ROC', 0.738):.3f}")
        col_d2.metric("AUC-PR (Prueba 2026)", f"{metricas.get('AUC_PR', 0.286):.3f}")
        col_d3.metric("Recall en Top 10%", f"{metricas.get('recall_top10%', 0.462)*100:.1f}%")
        col_d4.metric("Monto Capturado Top 10%", f"{metricas.get('monto_top10%', 0.448)*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        col_imp, col_neg = st.columns([1, 1])

        with col_imp:
            st.markdown("##### 📌 Top Variables más Influyentes (Permutation Importance)")
            if df_importancia is not None and not df_importancia.empty:
                top_imp = df_importancia.head(12).sort_values(by="importancia", ascending=True)
                fig_imp = px.bar(
                    top_imp,
                    x="importancia",
                    y="variable",
                    orientation='h',
                    title="Caída de AUC-PR al desordenar cada variable",
                    color="importancia",
                    color_continuous_scale=['#444444', '#FFD400']
                )
                fig_imp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#141414',
                    font=dict(color="#FFFFFF"),
                    showlegend=False,
                    height=420,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("El archivo de importancia de variables no está disponible.")

        with col_neg:
            st.markdown("##### 💼 Justificación de Negocio y Deciles de Riesgo")
            st.markdown(
                """
                En problemas de detección de notas de crédito con fuerte desbalance de clases (~9% positivos), 
                el objetivo de negocio no es optimizar únicamente la precisión global, sino **concentrar el mayor 
                riesgo operativo en la menor cantidad de clientes a revisar**:
                
                - **Lift en Decil 1 (Top 10%):** Revisando únicamente el **10%** de clientes calificados con mayor 
                  probabilidad por el modelo, Prosegur captura el **46.2%** de todos los clientes que emitirán una NC 
                  y el **44.8% del monto en disputa** del mes siguiente.
                - **Calibración Isotónica:** Las probabilidades arrojadas reflejan fielmente la tasa real de eventos 
                  (a diferencia de los scores brutos de un Random Forest con submuestreo o balanceo de clases).
                - **Umbral Óptimo (8.53%):** Seleccionado rigurosamente mediante la curva Precision-Recall sobre el 
                  año de validación (2025) sin incurrir en fuga de información sobre la prueba (2026).
                """
            )
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Impacto Estimado</div>
                    <div style="font-size: 1.05rem; color: #FFF; margin-top: 6px;">
                        Priorizar el Top 10% reduce en más de <b>4.5x</b> el esfuerzo operativo de revisión 
                        frente a una asignación aleatoria o manual.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br><hr style='border-color: #222222;'><br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; color: #777777; font-size: 0.85rem;'>"
        "Prosegur Security Analytics © 2026 • Modelo NC Mes Siguiente v2 • Desarrollado con Streamlit & Scikit-Learn"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
