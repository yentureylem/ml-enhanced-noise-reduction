import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import io
from streamlit.runtime.scriptrunner import get_script_run_ctx
import streamlit.components.v1 as components

# Model sınıfı (notebook'tan)
class UltraSimpleDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 16, 5, padding=2)
        self.conv4 = nn.Conv1d(16, 1, 5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = torch.tanh(self.conv4(out))
        return out.squeeze(1)

@st.cache_resource
def load_model():
    model = UltraSimpleDenoiser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Train (notebook'tan)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Dummy data
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, int(sr * duration))
    clean = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 400 * t)
    noise = 0.5 * np.random.randn(len(t))
    noisy = clean + noise
    
    chunk_len = 4000
    pairs = []
    for i in range(0, len(noisy), chunk_len):
        n_chunk = noisy[i:i+chunk_len]
        c_chunk = clean[i:i+chunk_len]
        if len(n_chunk) < chunk_len:
            n_chunk = np.pad(n_chunk, (0, chunk_len - len(n_chunk)))
            c_chunk = np.pad(c_chunk, (0, chunk_len - len(c_chunk)))
        pairs.append((n_chunk, c_chunk))
    
    model.train()
    with st.spinner("Model training..."):
        for epoch in range(15):
            total_loss = 0
            for n_chunk, c_chunk in pairs:
                n_t = torch.from_numpy(n_chunk).float().unsqueeze(0).unsqueeze(0).to(device)
                c_t = torch.from_numpy(c_chunk).float().unsqueeze(0).unsqueeze(0).to(device)
                pred = model(n_t)
                loss = criterion(pred, c_t.squeeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            st.status(f"Epoch {epoch+1}/15, Loss: {total_loss/len(pairs):.4f}", state="running")
    
    model.eval()
    return model

# Colab iframe
def colab_iframe(url, width=700, height=500):
    ctx = get_script_run_ctx()
    if ctx is None:
        components.iframe(url, width=width, height=height)
    else:
        st.components.v1.iframe(url, width=width, height=height)

st.title("🧼 ML Enhanced Noise Reduction")
st.markdown("Upload noisy audio, get denoised version instantly!")

model = load_model()
device = next(model.parameters()).device
sr = 16000

uploaded_file = st.file_uploader("Upload WAV file", type=['wav'])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    
    st.audio(audio_bytes)
    
    # Process
    chunk_len = 4000
    denoised_audio = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(audio), chunk_len):
            chunk = audio[i:i+chunk_len]
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
            
            chunk_t = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            denoised_chunk = model(chunk_t)
            denoised_audio.append(denoised_chunk.squeeze().cpu().numpy())
    
    denoised_audio = np.concatenate(denoised_audio)[:len(audio)]
    
    buffer = io.BytesIO()
    sf.write(buffer, denoised_audio, sr)
    buffer.seek(0)
    
    st.audio(buffer.getvalue(), format='audio/wav')
    
    st.download_button(
        "💾 Download denoised",
        buffer.getvalue(),
        "denoised.wav",
        "audio/wav"
    )

st.header("📓 Full Colab Notebook")
# Colab linkinizi buraya koyun (public yapın)
colab_url = "https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID"
colab_iframe(colab_url)

st.info("**requirements.txt:**\n```
streamlit
torch
torchaudio
librosa
soundfile
numpy
```")
