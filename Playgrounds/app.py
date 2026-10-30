import streamlit as st

st.set_page_config(page_title="Open Dagen Dashboard", layout="wide")

# 1. MAAK EERST DE PAGINA-OBJECTEN AAN (zodat Python ze kent)
ml_page = st.Page(
    "pages/Machine_learning_Statistiek.py", 
    title="Machine Learning & Statistiek", 
    icon="🚀"
)

calculus_page = st.Page(
    "pages/Calculus.py", 
    title="Calculus & Curve Fitting", 
    icon="📊"
)

# Koppel ook je overige pagina's uit je map zodat Streamlit ze indexeert
deep_learning_page = st.Page("pages/Deep_learning.py", title="Deep Learning", icon="🧠")
storytelling_page = st.Page("pages/Business_Storytelling.py", title="Business Storytelling", icon="📈")


# 2. DEFINIEER DAARNA PAS DE LAYOUT VAN DE HOMEPAGINA
def home_page_layout():
    st.title("👋 Welkom bij het Data Science & AI Dashboard")
    st.write("Ontdek hoe we wiskunde, statistiek en machine learning gebruiken om voorspellingen te doen!")

    st.markdown("---")
    st.subheader("Kies een interactief onderdeel om te starten:")

    col1, col2 = st.columns(2)

    with col1:
        st.info("### 🤖 Machine Learning & Statistiek")
        st.write("Zie hoe een echt Machine Learning model live huizenprijzen berekent op basis van jouw invoer.")
        # Dit werkt nu omdat ml_page hierboven al is aangemaakt!
        st.page_link(ml_page, label="Start Machine Learning", icon="🚀")

    with col2:
        st.success("### 🧮 Calculus & Curve Fitting")
        st.write("Word zelf het algoritme! Pas handmatig de wiskundige parameters aan om een formule perfect op de data te fitten.")
        # Dit werkt nu omdat calculus_page hierboven al is aangemaakt!
        st.page_link(calculus_page, label="Start Calculus", icon="📊")

    st.markdown("---")
    st.caption("Gemaakt voor de Open Dag - Opleiding Data Science & AI")


# 3. KOPPEL DE FUNCTIE AAN DE HOME PAGINA EN START
home_page = st.Page(home_page_layout, title="Home", icon="🏠", default=True)

# Start de navigatie engine
pg = st.navigation([home_page, ml_page, calculus_page, deep_learning_page, storytelling_page])
pg.run()
