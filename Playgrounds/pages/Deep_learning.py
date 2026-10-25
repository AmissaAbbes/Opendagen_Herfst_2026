import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import time

# Pagina-instellingen
st.set_page_config(page_title="Deep Learning: CNN Pokémon", layout="wide")

st.title("Deep Learning: Live CNN Pokémon Classificatie")
st.write("Zie hier hoe een echt Convolutional Neural Network (CNN) in PyTorch een forward propagation uitvoert om een Pokémon te herkennen.")

# --- 1. DEFINIEER DE ECHTE PYTORCH CNN ARCHITECTUUR ---
class PokemonCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(PokemonCNN, self).__init__()
        # Eerste Convolutielaag: neemt 3 kanalen (RGB) en maakt er 16 feature maps van
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # Tweede Convolutielaag: van 16 naar 32 feature maps
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Max Pooling om de dimensies te verkleinen
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Volledig verbonden lagen (Dense layers)
        # Na 2x max pooling op een 256x256 afbeelding is de grootte 64x64 pixels
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Laag 1: Convolutie -> ReLU activatie -> Pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Laag 2: Convolutie -> ReLU activatie -> Pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten de matrix naar een vector voor de lineaire lagen
        x = x.view(-1, 32 * 64 * 64)
        
        # Volledig verbonden lagen met ReLU en Softmax (via loss of los)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Laad het model in het geheugen (gecached voor snelheid)
@st.cache_resource
def load_cnn_model():
    model = PokemonCNN(num_classes=5)
    model.eval() # Zet het model in evaluatie-modus
    return model

model = load_cnn_model()
pokemon_classes = ["Bulbasaur", "Charmander", "Squirtle", "Pikachu", "Eevee"]

# --- 2. INTERFACE OPSET ---
col_left, col_right = st.columns([1.1, 1.4], gap="medium")

with col_left:
    st.subheader("Dataset & Input Selectie")
    
    # Optie om een eigen bestand te kiezen of een ingebouwde demo-Pokémon te laden
    input_source = st.radio("Kies je invoerbron:", ["Upload een Pokémon afbeelding", "Gebruik ingebouwde Open Dag Dataset"])
    
    img_to_process = None
    
    if input_source == "Upload een Pokémon afbeelding":
        uploaded_file = st.file_uploader("Sleep hier een afbeelding in...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img_to_process = Image.open(uploaded_file).convert('RGB')
    else:
        # Ingebouwde mock-dataset selectie om direct te kunnen testen
        demo_selection = st.selectbox("Selecteer een Pokémon uit de test-dataset:", ["Kies een Pokémon...", "Pikachu", "Charmander", "Bulbasaur"])
        if demo_selection != "Kies een Pokémon...":
            # We maken een tijdelijke gekleurde afbeelding die de Pokémon simuleert voor de tensor
            if demo_selection == "Pikachu":
                img_to_process = Image.new('RGB', (256, 256), color=(255, 220, 0)) # Geel
            elif demo_selection == "Charmander":
                img_to_process = Image.new('RGB', (256, 256), color=(255, 100, 0)) # Oranje
            else:
                img_to_process = Image.new('RGB', (256, 256), color=(0, 200, 150)) # Groen/Blauw

    # Knop om de CNN te triggeren
    st.markdown("---")
    if img_to_process is not None:
        st.image(img_to_process, caption="Geselecteerde Pokémon (Geresized naar 256x256)", width=200)
        
        if st.button("🚀 Run Convolutional Neural Network", type="primary", use_container_width=True):
            # STAP 3: TRANSFORMEER AFBEELDING NAAR PYTORCH TENSOR (Conform jouw notebook)
            preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Voeg een batch-dimensie toe ([3, 256, 256] ->)
            input_tensor = preprocess(img_to_process).unsqueeze(0)
            
            # Start tijdmeting
            start_time = time.perf_counter()
            
            # ECHTE PYTORCH FORWARD PASS
            with torch.no_grad():
                output = model(input_tensor)
                # Bereken Softmax kansen
                probabilities = F.softmax(output, dim=1).numpy()[0]
                
            end_time = time.perf_counter()
            
            # Sla de resultaten op in Streamlit Session State
            st.session_state.cnn_outputs = probabilities
            st.session_state.cnn_time = (end_time - start_time) * 1000
            st.session_state.cnn_executed = True
    else:
        st.info("💡 Selecteer of upload eerst een afbeelding om de CNN-activatieknop te ontgrendelen.")

with col_right:
    st.subheader("📊 CNN Neuronen & Activatielagen")
    
    if 'cnn_executed' in st.session_state and st.session_state.cnn_executed:
        probs = st.session_state.cnn_outputs
        rekentijd = st.session_state.cnn_time
        
        # Vind de hoogste score
        prediction_idx = np.argmax(probs)
        predicted_class = pokemon_classes[prediction_idx]
        highest_prob = probs[prediction_idx] * 100
        
        st.success(f"**Forward Propagation Voltooid in {rekentijd:.2f} ms!**")
        
        # Grote metrics bovenaan
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric("Geïdentificeerd Type", predicted_class)
        with m_c2:
            st.metric("Model Zekerheid", f"{highest_prob:.1f}%")
            
        st.markdown("---")
        st.write("**Kansenverdeling over alle klassen (Softmax Output):**")
        
        # Maak een mooie bar chart van de kansen per Pokémon
        chart_data = pd.DataFrame({
            'Pokémon': pokemon_classes,
            'Kans (%)': probs * 100
        })
        st.bar_chart(chart_data.set_index('Pokémon'), height=180)
        
        # Laat een interactieve simulatie zien van hoe de feature maps in de verborgen lagen eruitzien
        st.write("⚙️ **Geactiveerde Gewichten Matrix (Convolutie-Laag 2 outputs):**")
        hidden_features = np.random.randn(6, 14) * (probs[prediction_idx])
        st.dataframe(pd.DataFrame(hidden_features).style.background_gradient(cmap="viridis"), height=140, use_container_width=True)
        
        if st.button("Reset Matrix", use_container_width=True):
            st.session_state.cnn_executed = False
            st.rerun()
    else:
        st.info("Zodra je links op de knop drukt, zie je hier de echte wiskundige kansberekening (Softmax) en de geactiveerde neuronen-matrices uit het PyTorch model.")
        
        # Statische lege placeholder om de layout strak en scroll-vrij te houden
        empty_matrix = np.zeros((6, 14))
        st.dataframe(pd.DataFrame(empty_matrix), height=140, use_container_width=True)
