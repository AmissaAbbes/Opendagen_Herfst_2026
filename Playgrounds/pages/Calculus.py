import streamlit as st
import numpy as np
import pandas as pd
import time

# Forceer breedbeeldmodus en compacte paddings
st.set_page_config(page_title="Calculus", layout="wide", initial_sidebar_state="expanded")

# Ultra-compacte titel en introductie
st.title("Handmatig vs. Machine Learning Optimalisatie")
st.write("Pas de parameters aan om de lijn te fitten ($f(x) = ax^2 + bx + c$), Lineair regression de beste paramaters inschatten")

# 1. Genereer vaste data
@st.cache_data
def generate_target_data():
    np.random.seed(42)
    x = np.linspace(-5, 5, 40)
    y_ruis = 1.5 * (x**2) - 2 * x + 3 + np.random.normal(0, 2.5, size=len(x))
    return x, y_ruis

x_data, y_data = generate_target_data()

# AI State bijhouden
if 'ai_parameters' not in st.session_state:
    st.session_state.ai_parameters = None
if 'ai_tijd' not in st.session_state:
    st.session_state.ai_tijd = None

# Hoofd-verdeling: Links de controls en metrics, rechts de grote grafiek
main_col1, main_col2 = st.columns([1.1, 1.4], gap="medium")

with main_col1:
    st.subheader("Sliders (Jouw Model)")
    # Sliders compacter naast elkaar zetten via sub-kolommen om hoogte te besparen
    sub_c1, sub_c2, sub_c3 = st.columns(3)
    with sub_c1:
        a = st.slider("a (Kromming)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
    with sub_c2:
        b = st.slider("b (Helling)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    with sub_c3:
        c = st.slider("c (Hoogte)", min_value=-10.0, max_value=15.0, value=0.0, step=0.5)

    # Bereken handmatige fit
    y_pred_mens = a * (x_data**2) + b * x_data + c
    mse_mens = np.mean((y_data - y_pred_mens) ** 2)

    st.markdown("---")
    
    # Actie-sectie en metrics compacter indelen
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.metric(label="Jouw Foutmarge (MSE)", value=f"{mse_mens:.1f}")
        if st.button("🤖 Start Regressie (AI)", type="primary", use_container_width=True):
            start_time = time.perf_counter()
            ai_a, ai_b, ai_c = np.polyfit(x_data, y_data, 2)
            end_time = time.perf_counter()
            st.session_state.ai_parameters = (ai_a, ai_b, ai_c)
            st.session_state.ai_tijd = (end_time - start_time) * 1000
            st.rerun()

    with metric_col2:
        if st.session_state.ai_parameters is not None:
            ai_a, ai_b, ai_c = st.session_state.ai_parameters
            y_pred_ai = ai_a * (x_data**2) + ai_b * x_data + ai_c
            mse_ai = np.mean((y_data - y_pred_ai) ** 2)
            
            st.metric(label="AI Foutmarge (MSE)", value=f"{mse_ai:.1f}")
            st.metric(label="AI Rekentijd", value=f"{st.session_state.ai_tijd:.4f} ms")
            
            if st.button("Reset AI", use_container_width=True):
                st.session_state.ai_parameters = None
                st.session_state.ai_tijd = None
                st.rerun()
        else:
            y_pred_ai = None
            st.info("Klik op de knop Lineaire regressie te activeren")

with main_col2:
    # Grafiek vult de gehele rechterkant op gelijke hoogte
    chart_dict = {
        'x': x_data,
        'Echte Data': y_data,
        'Jouw Model': y_pred_mens
    }
    if y_pred_ai is not None:
        chart_dict['AI Model'] = y_pred_ai
        
    chart_data = pd.DataFrame(chart_dict)
    
    # Gebruik st.line_chart met een vaste, compacte hoogte die op elk scherm past
    st.line_chart(chart_data.set_index('x'), height=340)
