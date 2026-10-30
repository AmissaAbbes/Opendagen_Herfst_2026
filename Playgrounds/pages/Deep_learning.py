import os
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import streamlit as st

# --- 1. PAGINA-INSTELLINGEN ---
st.set_page_config(page_title="Deep Learning: Pokémon CNN", layout="wide")

st.title("🌌 Deep Learning: Live CNN Pokémon Classificatie & Training")
st.write("Zie hier hoe een echt Convolutional Neural Network (CNN) in PyTorch forward propagation uitvoert én live getraind kan worden.")

# --- 2. DYNAMISCHE MODEL ARCHITECTUUR ---
class PokeNet(nn.Module):
    def __init__(self, n_classes):
        super(PokeNet, self).__init__()
        self.seq = nn.Sequential(
            # Input size: 3 * 256 * 256
            nn.Conv2d(in_channels=3, out_channels=9, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=9, out_channels=18, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 18 * 64 * 64
            nn.Flatten(),
            nn.Linear(18 * 64 * 64, 64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )  

    def forward(self, x):
        return self.seq(x)

# --- 3. GLOBALE DATASET SELECTIE & DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.sidebar.header("📁 Dataset Selectie")
dataset_source = st.sidebar.selectbox(
    "Kies de actieve dataset-map:",
    [f"..\Playgrounds\pokemon_small", f"..\Playgrounds\pokemon"]
)

# Dynamisch de klassen correct scannen uit de geselecteerde map (train of test)
@st.cache_data
def get_dynamic_classes(base_path):
    for subfolder in ["train", "test", ""]:
        p = os.path.join(base_path, subfolder) if subfolder else base_path
        if os.path.exists(p):
            try:
                temp_dataset = datasets.ImageFolder(root=p)
                if len(temp_dataset.classes) > 0:
                    return temp_dataset.classes
            except Exception:
                pass
    # Harde fallback op basis van jouw bekende klassen als de ImageFolder faalt
    if "pokemon_small" in base_path:
        return ['Alakazam', 'Charmander', 'Dragonite', 'Pikachu', 'Wigglytuff']
    else:
        return ['Abra', 'Aerodactyl', 'Alakazam', 'Alolan Diglett', 'Arbok', 'Arcanine', 'Articuno', 'Beedrill', 'Bellsprout', 'Blastoise', 'Bulbasaur', 'Butterfree', 'Caterpie', 'Chansey', 'Charizard', 'Charmander', 'Charmeleon', 'Clefable', 'Clefairy', 'Cloyster', 'Cubone', 'Dewgong', 'Diglett', 'Ditto', 'Dodrio']

# Bepaal het exacte trainingspad
train_path = os.path.join(dataset_source, "train")
if not os.path.exists(train_path):
    train_path = dataset_source

pokemon_classes = get_dynamic_classes(dataset_source)
n_classes = len(pokemon_classes)

st.sidebar.info(f"Aantal gedetecteerde klassen: {n_classes}")

# Constanten voor de transformaties
mean_param = [0.485, 0.456, 0.406]  
std_param = [0.229, 0.224, 0.225]

# Functie om het model veilig te laden op basis van het dynamische aantal klassen
@st.cache_resource
def load_trained_model(num_classes):
    model = PokeNet(n_classes=num_classes).to(device)
    if os.path.exists("pokemon_cnn.pth"):
        try:
            model.load_state_dict(torch.load("pokemon_cnn.pth", map_location=device))
            st.sidebar.success("✅ Modelgewichten geladen!")
        except Exception:
            st.sidebar.warning("⚠️ Model gewichten matchen niet. Train het model opnieuw voor deze dataset.")
    else:
        st.sidebar.warning("⚠️ Geen gewichten gevonden. Model gebruikt ongetrainde waarden.")
    model.eval()
    return model

# --- 4. LIVE TRAININGSMODULE IN DE SIDEBAR ---
st.sidebar.header("🏋️‍♂️ Model Training")
epochs = st.sidebar.slider("Aantal Epochs", min_value=1, max_value=20, value=5)
lr = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.001, 0.0001, 0.00001], value=0.0001)

if st.sidebar.button("🚀 Start Training", type="secondary", use_container_width=True):
    if os.path.exists(train_path):
        with st.sidebar:
            status_text = st.empty()
            progress_bar = st.progress(0)
            
        st.subheader("📈 Live Trainingsvoortgang")
        chart_loss = st.empty()
        chart_acc = st.empty()
        
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_param, std=std_param)
        ])
        
        try:
            train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            
            actual_classes = len(train_dataset.classes)
            train_model = PokeNet(n_classes=actual_classes).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(train_model.parameters(), lr=lr)
            
            loss_history, acc_history = [], []
            train_model.train()
            
            for epoch in range(epochs):
                running_loss, correct, total = 0.0, 0, 0
                
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = train_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                epoch_loss = running_loss / total
                epoch_acc = (correct / total) * 100
                
                loss_history.append(epoch_loss)
                acc_history.append(epoch_acc)
                
                status_text.text(f"Epoch {epoch+1}/{epochs} - Accuracy: {epoch_acc:.1f}%")
                progress_bar.progress((epoch + 1) / epochs)
                
                chart_loss.line_chart(pd.DataFrame({"Loss": loss_history}), height=150)
                chart_acc.line_chart(pd.DataFrame({"Accuracy (%)": acc_history}), height=150)
                
            torch.save(train_model.state_dict(), "pokemon_cnn.pth")
            st.sidebar.success("🏆 Training voltooid!")
            st.cache_resource.clear() 
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Fout tijdens training: {e}")
    else:
        st.sidebar.error(f"Trainingsmap niet gevonden op {train_path}")

# --- 5. INTERFACE VOOR CLASSIFICATIE (TESTEN) ---
model = load_trained_model(n_classes)

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_param, std=std_param)
])

col_left, col_right = st.columns([1.1, 1.4], gap="medium")

with col_left:
    st.subheader("Dataset & Input Selectie")
    input_source = st.radio("Kies je invoerbron:", ["Gebruik test-dataset uit geselecteerde map", "Upload een eigen Pokémon afbeelding"])
    
    img_to_process = None
    
    if input_source == "Upload een eigen Pokémon afbeelding":
        uploaded_file = st.file_uploader("Sleep hier een afbeelding in...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img_to_process = Image.open(uploaded_file).convert('RGB')
    else:
        target_path = os.path.join(dataset_source, "test")
        if not os.path.exists(target_path):
            target_path = dataset_source
            
        if os.path.exists(target_path):
            try:
                test_dataset = datasets.ImageFolder(root=target_path)
                available_classes = test_dataset.classes
                
                demo_selection = st.selectbox(f"Selecteer een Pokémon:", ["Kies een Pokémon..."] + available_classes)
                
                if demo_selection != "Kies een Pokémon...":
                    class_idx = test_dataset.class_to_idx[demo_selection]
                    indices = [i for i, (_, label) in enumerate(test_dataset.samples) if label == class_idx]
                    
                    if indices:
                        random_idx = random.choice(indices)
                        img_path, _ = test_dataset.samples[random_idx]
                        img_to_process = Image.open(img_path).convert('RGB')
            except Exception as e:
                st.error(f"Fout bij laden testmap: {e}")

    st.markdown("---")
    if img_to_process is not None:
        st.image(img_to_process, caption="Geselecteerde Pokémon", width=220)
        if st.button("🚀 Run Convolutional Neural Network", type="primary", use_container_width=True):
            input_tensor = preprocess(img_to_process).unsqueeze(0).to(device)
            start_time = time.perf_counter()
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1).cpu().numpy()
            end_time = time.perf_counter()
            
            st.session_state.cnn_outputs = probabilities
            st.session_state.cnn_time = (end_time - start_time) * 1000
            st.session_state.cnn_executed = True
# --- RECHTERKOLOM MET PROGRESSBARS ---
with col_right:
    st.subheader("📊 CNN Neuronen & Voorspelling")
    
    if 'cnn_executed' in st.session_state and st.session_state.cnn_executed:
        probs = st.session_state.cnn_outputs.flatten()
        rekentijd = st.session_state.cnn_time
        prediction_idx = np.argmax(probs)
        predicted_class = pokemon_classes[prediction_idx] if prediction_idx < len(pokemon_classes) else f"Klasse {prediction_idx}"
        highest_prob = probs[prediction_idx] * 100
        
        st.success(f"Forward Propagation Voltooid in {rekentijd:.2f} ms!")
        
        # Maak twee kolommen voor de metrics naast elkaar
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric("Geïdentificeerd Type", predicted_class)
        with m_c2:
            st.metric("Model Zekerheid", f"{highest_prob:.1f}%")
            
        st.markdown("---")
        st.write("Modelbetrouwbaarheid per Pokémon (Softmax Kans):")
        
        # De for-loop voor de balken moet binnen de 'if' vallen
        for idx, p_class in enumerate(pokemon_classes[:len(probs)]):
            chance = float(probs[idx])
            chance = max(0.0, min(1.0, chance))
            st.write(f"**{p_class}**: {chance*100:.1f}%")
            st.progress(chance)
            
        # De reset-knop moet ook binnen de 'if' vallen
        if st.button("Reset Voorspelling", use_container_width=True):
            st.session_state.cnn_executed = False
            st.rerun()
            
    else:
        # Deze 'else' hoort bij de hoofd 'if' en wordt getoond als er nog GEEN voorspelling is gedaan
        st.info("Zodra je links op de knop drukt, zie je hier de live wiskundige kansberekening (Softmax) per Pokémon via balken weergegeven.")
