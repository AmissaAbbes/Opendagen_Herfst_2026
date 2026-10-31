import streamlit as st

st.set_page_config(page_title="Open Dagen Dashboard", layout="wide")

# ==============================================================================
# 🎃 HALLOWEEN BACKGROUND & THEME LAYOUT (CSS INJECTION)
# ==============================================================================
# Sfeervolle rechtenvrije Halloween vectorachtergrond
halloween_bg = "https://unsplash.com"

st.markdown(
    f"""
    <style>
    /* De volledige achtergrond van de app voorzien van de Halloween afbeelding met een donkere filterlaag */
    .stApp {{
        background-image: linear-gradient(rgba(13, 6, 20, 0.88), rgba(13, 6, 20, 0.88)), url("{halloween_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* De sidebar en header donker houden voor maximale leesbaarheid van het menu */
    [data-testid="stSidebar"], [data-testid="stHeader"] {{
        background-color: #0d0614 !important;
    }}
    
    /* Griezelig typemachine-lettertype en pompoenoranje voor alle koppen */
    h1, h2, h3, h4, label, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: #ff7518 !important; 
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: bold;
    }}

    /* Standaard paragrafen en teksten wit/lichtpaars maken tegen de donkere achtergrond */
    p, span, .stMarkdown p {{
        color: #f0e6ff !important;
        font-family: 'Courier New', Courier, monospace !important;
    }}
    
    /* De dashboardkaarten stylen als mystieke panelen met een oranje gloed en automatische hoogte */
    [data-testid="stAlert"] {{
        background-color: rgba(28, 17, 36, 0.95) !important; 
        border: 2px solid #ff7518 !important; 
        border-radius: 12px !important;
        box-shadow: 0 6px 20px rgba(255, 117, 24, 0.4) !important;
        height: auto !important; 
        min-height: 290px !important; /* Zorgt voor een mooie gelijke hoogte over alle kaarten */
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        padding: 20px !important;
    }}
    
    /* Zorg dat de titels binnen de st.alert kaarten ook altijd netjes oranje zijn */
    [data-testid="stAlert"] h3 {{
        color: #ff7518 !important;
        margin-top: 0px !important;
    }}
    
    /* De navigatieknoppen (st.page_link) transformeren naar lichtgevend gifgroen */
    div[data-testid="stPageLink"] a {{
        background-color: rgba(36, 58, 31, 0.9) !important;
        border: 2px solid #39ff14 !important;
        color: #39ff14 !important;
        font-weight: bold;
        font-family: 'Courier New', Courier, monospace !important;
        border-radius: 6px;
        text-decoration: none;
        padding: 10px;
        display: inline-block;
        width: 100%;
        text-align: center;
        margin-top: auto !important; /* Duwt de knop altijd strak naar de bodem van de kaart */
    }}
    
    /* Effect wanneer studenten met de muis over de knoppen bewegen */
    div[data-testid="stPageLink"] a:hover {{
        background-color: #39ff14 !important;
        color: #0d0614 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# 1. MAAK DE PAGINA-OBJECTEN AAN
# ==============================================================================
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

deep_learning_page = st.Page(
    "pages/Deep_learning.py", 
    title="Deep Learning", 
    icon="🧠"
)

storytelling_page = st.Page(
    "pages/Business_Storytelling.py", 
    title="Business Storytelling", 
    icon="📈"
)


# ==============================================================================
# 2. DEFINIEER DE LAYOUT VAN DE HOMEPAGINA
# ==============================================================================
def home_page_layout():
    st.title("🎃 Welkom bij het Data Science & AI Dashboard")
    st.write("Ontdek hoe we wiskunde, statistiek en machine learning gebruiken om voorspellingen te doen!")

    st.markdown("---")
    st.subheader("Kies een interactief onderdeel om te starten:")

    # Vier kolommen netjes naast elkaar voor alle onderdelen
    col1, col2, col3, col4 = st.columns(4)

    # Let op: de icon="" zorgt ervoor dat de kapotte icon-tekst verdwijnt!
    with col1:
        st.info("### 🤖 ML & Statistiek", icon="")
        st.write("Zie hoe een echt Machine Learning model live huizenprijzen berekent op basis van jouw invoer.")
        st.page_link(ml_page, label="Start Machine Learning", icon="🚀")

    with col2:
        st.success("### 🧮 Calculus & Fitting", icon="")
        st.write("Word zelf het algoritme! Pas handmatig parameters aan om een formule perfect te fitten.")
        st.page_link(calculus_page, label="Start Calculus", icon="📊")

    with col3:
        st.warning("### 🧠 Deep Learning", icon="")
        st.write("Duik in de wereld van neurale netwerken en ontdek hoe computers afbeeldingen herkennen.")
        st.page_link(deep_learning_page, label="Start Deep Learning", icon="🧠")

    with col4:
        st.error("### 📈 Storytelling", icon="")
        st.write("Leer hoe je complexe data vertaalt naar een helder en overtuigend verhaal voor bedrijven.")
        st.page_link(storytelling_page, label="Start Storytelling", icon="📈")

    st.markdown("---")
    st.caption("Gemaakt voor de Open Dag - Opleiding Data Science & AI 👻")


# ==============================================================================
# 3. KOPPEL DE FUNCTIE AAN DE HOME PAGINA EN RUN DE NAVIGATIE
# ==============================================================================
home_page = st.Page(home_page_layout, title="Home", icon="🏠", default=True)

# Start de navigatie engine van Streamlit
pg = st.navigation([home_page, ml_page, calculus_page, deep_learning_page, storytelling_page])
pg.run()
