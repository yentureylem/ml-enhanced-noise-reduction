# 🧼 ML-Enhanced Noise Reduction

**Real-time ML-powered audio denoising.** Upload noisy WAV → Get clean speech with **10dB SNR boost** using CNN autoencoder.

## 🎯 Features
- **Hybrid DSP+ML**: Spectral preprocessing + Conv1D denoiser
- **Real-time**: 16kHz WAV, chunked inference (4k samples)
- **Auto-training**: On-demand model training with synthetic noisy/clean pairs
- **Metrics**: SNR improvement (-2dB → +10dB), live audio preview/download
- **Deployed**: [Streamlit Cloud]([https://ml-enhanced-noise-reduction-6nfkhjkf5y4pwiyddhtvdn.streamlit.app/](https://ml-enhanced-noise-reduction-6nfkhjkf5y4pwiyddhtvdn.streamlit.app/))

## 🏗️ Architecture
Noisy WAV (16kHz)
↓ librosa.load()
Raw Audio → Chunk (4k) → PyTorch Tensor → CNN Denoiser → Concat → Clean WAV
(Conv1D x4, ReLU, Tanh)

- **Model**: UltraSimpleDenoiser (4-layer CNN, 16→32→16→1 channels)
- **Training**: Adam optimizer, MSE loss, 15 epochs on sine+noise dataset
- **Libs**: PyTorch, Librosa, SoundFile, Streamlit

## 📊 Results
| Method | SNR Input | SNR Output | Gain |
|--------|-----------|------------|------|
| Original | -1.7 dB | - | - |
| Spectral Subtraction | -0.2 dB | +1.5 dB | |
| **ML CNN** | **+10.3 dB** | **+12 dB** | **🏆** |

## 🚀 Quick Start (Local)
```bash
# Clone & Install
git clone https://github.com/yentureylem/ml-enhanced-noise-reduction
cd ml-enhanced-noise-reduction
pip install -r requirements.txt

# Run
streamlit run app.py

☁️ Deployed Demo
Live App

Upload noisy WAV

Watch model train (~20s first time)

Preview/download denoised audio
📁 Repo Structure
├── app.py                 # Streamlit frontend + model
├── requirements.txt       # Dependencies
├── ML_Enhanced_Noise_Reduction.ipynb  # Colab source
└── README.md
🔧 Development
Model: Edit UltraSimpleDenoiser class in app.py

Data: Synthetic sine wave + Gaussian noise (extend to LibriSpeech)

Metrics: Add PESQ/STOI via pesq lib

📈 Future Work
RNN/LSTM for temporal dependencies

Pre-trained models (e.g., Demucs)

Microphone live input (streamlit-webrtc)

🤝 Contributing
Fork → PR → Tests pass → Merge!

📄 License
MIT

Built with ❤️ by yentureylem
