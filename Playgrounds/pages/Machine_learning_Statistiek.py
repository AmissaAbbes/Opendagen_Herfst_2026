import streamlit as st
import pandas as pd
import pydeck as pdk
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Machine Learning", layout="wide")

# Gebruik caching zodat de data en het model niet bij elke klik opnieuw laden
@st.cache_resource
def setup_ml_model():
    california = fetch_california_housing(as_frame=True)
    df_full = california.frame
    
    # Data opschonen
    df_clean = pd.DataFrame({
        'Inkomen': df_full['MedInc'],
        'Leeftijd': df_full['HouseAge'],
        'Kamers': df_full['AveRooms'],
        'lat': df_full['Latitude'],
        'lon': df_full['Longitude'],
        'Prijs': df_full['MedHouseVal'] * 100000
    })
    
    # Filter uitschieters en neem een sample voor de kaart
    df_sample = df_clean[(df_clean['Kamers'] < 10) & (df_clean['Prijs'] < 490000)].sample(1000, random_state=42)
    
    # Model trainen
    X = df_sample[['Inkomen', 'Leeftijd', 'Kamers']]
    y = df_sample['Prijs']
    model = LinearRegression().fit(X, y)
    
    return df_sample, model

# Laad de data en het getrainde model
df, model = setup_ml_model()

st.title("Machine Learning: Huisprijs Voorspeller")
st.write("Pas de schuifbalken aan en zie hoe het model live de prijs berekent.")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Knoppen")
    input_inc = st.slider("Mediaan inkomen (x $10.000)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
    input_age = st.slider("Leeftijd van de huizen (jaren)", min_value=1, max_value=52, value=25, step=1)
    input_kamers = st.slider("Gemiddeld aantal kamers", min_value=1, max_value=10, value=5, step=1)

    # Live voorspelling direct in Streamlit
    prediction = model.predict([[input_inc, input_age, input_kamers]])[0]
    voorspelling = max(0.0, float(prediction))
    
    st.markdown("---")
    st.metric(label="Voorspelde Huisprijs", value=f"$ {voorspelling:,.2f}")
    
    procent = min(1.0, voorspelling / 500000.0)
    st.progress(procent, text="Prijsklasse indicator")

with col2:
    st.subheader("Echte data locaties")
    layer = pdk.Layer(
        "HexagonLayer", data=df, get_position=["lon", "lat"], radius=12000,
        elevation_scale=100, extruded=True, color_range=[[8, 104, 172]]
    )
    view_state = pdk.ViewState(latitude=36.7783, longitude=-119.4179, zoom=5.0, pitch=45)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
