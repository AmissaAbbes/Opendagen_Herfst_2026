import streamlit as st
import pandas as pd
import plotly.express as px

# Forceer breedbeeldmodus voor een strak presentatie-dashboard
st.set_page_config(page_title="Data Storytelling", layout="wide")

st.title("📖 Data Storytelling: Van Data naar Actie")
st.write("Ontdek hoe je met de juiste visualisatie een directeur overtuigt om te investeren in Data Science.")

# Keuzemenu voor de bezoeker om het effect van storytelling te ervaren
st.markdown("### Stap 1: Kies hoe je de data presenteert")
presentatie_stijl = st.radio(
    "Hoe laat je de druktegegevens van het pretpark zien aan de directie?",
    ["De Exceltabel (Geen Storytelling)", "Het visuele dashboard (Wel Storytelling)"]
)

st.markdown("---")

# Uitgebreide demodata die de business case cijfermatig onderbouwt
data_attracties = {
    'Attractie': ['Achtbaan Python', 'Waterbaan', 'Spookhuis', 'Draaimolen', 'Kinderland'],
    'Gemiddelde Wachttijd (Minuten)': [75, 60, 35, 15, 10],
    'Klanttevredenheid Score (1-10)': [4.2, 5.1, 6.5, 8.2, 8.8],
    'Gemiddelde Horeca Uitgave (p.p.)': [3.50, 4.20, 8.50, 14.00, 16.50],
    'Status': ['Kritiek', 'Kritiek', 'Gemiddeld', 'Goed', 'Goed']
}
df = pd.DataFrame(data_attracties)

# Scenario 1: De saaie tabel (hoe het vaak misgaat)
if presentatie_stijl == "De Exceltabel (Geen Storytelling)":
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📊 Ruwe Data Overzicht")
        st.write("Je laat de directeur deze tabel zien tijdens de vergadering:")
        st.dataframe(df, use_container_width=True)
        
    with col2:
        st.subheader("Reactie van de Directeur")
        st.error("'Bedankt voor de cijfers, we kijken er volgend kwartaal naar. Ik zie zo snel niet wat we hiermee moeten.'")
        st.info(
            "**Waarom dit faalt:** De data bevat wel alle antwoorden, maar er is geen focus. "
            "De directeur ziet niet in één oogopslag dat lange wachttijden direct omzet kosten."
        )

# Scenario 2: Het visuele verhaal (Data Storytelling & Onderbouwing)
else:
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.subheader("Het Inzicht: Wachttijd vreet Horeca-omzet op!")
        
        # Visuele weergave van de harde data: Wachttijd vs Horeca uitgaven
        fig = px.scatter(
            df, 
            x='Gemiddelde Wachttijd (Minuten)', 
            y='Gemiddelde Horeca Uitgave (p.p.)',
            text='Attractie',
            color='Status',
            color_discrete_map={'Kritiek': '#ff4b4b', 'Gemiddeld': '#ffa500', 'Goed': '#00cc96'},
            size='Gemiddelde Wachttijd (Minuten)',
            size_max=30
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Reactie van de Directeur")
        st.success("'Aha! Wie in de rij staat voor de Python geeft geen geld uit op onze terrassen. Wat is de business case?'")
        
        st.markdown("### 📊 De Harde Cijfers achter de Oplossing:")
        
        # Berekeningen op basis van de data
        huidige_horeca_gemiddelde = df['Gemiddelde Horeca Uitgave (p.p.)'].mean()
        verwachte_horeca_na_optimalisatie = 11.50 # Conservatieve schatting als wachttijden dalen
        extra_omzet_per_bezoeker = verwachte_horeca_na_optimalisatie - huidige_horeca_gemiddelde
        
        # Presenteer de data-onderbouwing in metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="Huidige Horeca Gemiddelde", value=f"€ {huidige_horeca_gemiddelde:.2f}")
        with m2:
            st.metric(label="Extra Omzet per Bezoeker", value=f"+ € {extra_omzet_per_bezoeker:.2f}", delta="Na app-optimalisatie")
            
        st.markdown("---")
        st.write(
            f"**Het Data Verhaal:** Uit onze regressieanalyse blijkt een sterke negatieve correlatie "
            f"tussen wachttijd en parkuitgaven. Bezoekers bij de Draaimolen (15 min wachttijd) geven "
            f"gemiddeld **€ {df.loc[3, 'Gemiddelde Horeca Uitgave (p.p.)']:.2f}** uit, terwijl dit bij de Python "
            f"(75 min wachttijd) slechts **€ {df.loc[0, 'Gemiddelde Horeca Uitgave (p.p.)']:.2f}** is. "
            f"Ons ML-model spreidt bezoekers, verkort de rijen met 20%, en verhoogt direct de horeca-omzet."
        )
