# ML Audio Denoising
**12dB SNR Gain** – CNN (PyTorch) + Spectral Gating

[![Demo Plot](denoising_results.png)][image:105]

| Method | SNR |
|--------|-----|
| Noisy | -1.7dB |
| Spectral | -0.2dB |
| **ML CNN** | **10.3dB** |

## Live Demo
Streamlit app upload noisy audio → instant ML clean [app.py](../app.py)

## Colab
[Full Notebook](ml_audio_denoising.ipynb)

Tech: PyTorch Conv1D, librosa, SciPy PESQ.
