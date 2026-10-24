import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Laad de volledige dataset inclusief GPS-coördinaten
california = fetch_california_housing(as_frame=True)
df_full = california.frame

# Data voorbereiden en opschonen
df = pd.DataFrame({
    'Inkomen': df_full['MedInc'],
    'Leeftijd': df_full['HouseAge'],
    'Kamers': df_full['AveRooms'],
    'lat': df_full['Latitude'],
    'lon': df_full['Longitude'],
    'Prijs': df_full['MedHouseVal'] * 100000
})

# Filter uitschieters en neem een sample voor snelle rendering op de kaart
df = df[(df['Kamers'] < 10) & (df['Prijs'] < 490000)].sample(1000, random_state=42)

# Train het Machine Learning model
X = df[['Inkomen', 'Leeftijd', 'Kamers']]
y = df['Prijs']
model = LinearRegression()
model.fit(X, y)

# Streamlit interface
st.title("Machine Learning: Huisprijs Voorspeller")
st.write("Pas de schuifbalken aan en zie hoe het model live de prijs berekent.")

# Layout opbreken in kolommen voor een dashboard-look
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Knoppen")
    input_inc = st.slider("Mediaan inkomen (x $10.000)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
    input_age = st.slider("Leeftijd van de huizen (jaren)", min_value=1, max_value=52, value=25, step=1)
    input_kamers = st.slider("Gemiddeld aantal kamers", min_value=1, max_value=10, value=5, step=1)

    # Bereken voorspelling en pak het eerste element uit de array om de TypeError te voorkomen
    voorspelling = max(0.0, float(model.predict([[input_inc, input_age, input_kamers]])[0]))
    
    st.markdown("---")
    st.metric(label="Voorspelde Huisprijs", value=f"$ {voorspelling:,.2f}")
    
    # Visuele prijs-indicator (progress bar)
    max_prijs_schaal = 500000.0
    procent = min(1.0, voorspelling / max_prijs_schaal)
    st.progress(procent, text="Prijsklasse indicator")

with col2:
    st.subheader("Waar komt de data vandaan?")
    st.write("De 3D-kolommen tonen de echte huizenprijzen in Californië.")
    
    # Pydeck HexagonLayer definitie
    layer = pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position=["lon", "lat"],
        radius=12000,
        elevation_scale=100,
        elevation_range=[0, 3000],
        extruded=True,
        coverage=1,
        upper_percentile=99,
        color_range=[[8, 104, 172]])
    
    # Kaart weergave instellen
    view_state = pdk.ViewState(
        latitude=36.7783,
        longitude=-119.4179,
        zoom=5.0,
        pitch=45,
        bearing=0
    )
    
    # Render de kaart in Streamlit
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
