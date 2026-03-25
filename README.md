# 🧠 Research Projects

Welcome to my personal repository of data science, deep learning and research-based projects. This collection showcases a range of topics including deep learning, quantum computing, bioinformatics, recommendation systems, and scientific applications.

## 📁 Contents

Below is a summary of each project notebook included:

### 🔬 Machine Learning & Deep Learning
- **Nvidia_Deep_Learning.ipynb**
  Demonstrates various deep learning models using NVIDIA technologies and libraries, including CUDA, cuDNN, or TensorRT.
  Transfer learning with an ImageNet-pretrained model for fresh vs rotten fruit classification using TensorFlow/Keras

- **Nvidia_recommenderIntelligentSystems.ipynb**
  A deep dive into intelligent recommendation systems with NVIDIA's frameworks, using Merlin or related tools.
  Deploying a TensorFlow recommender system with Triton Inference Server and NVTabular for preprocessing.

- **WaveAdamw.ipynb**
  Benchmarks a custom optimizer (Wave-AdamW, applying row-wise Laplacian smoothing to AdamW) against standard AdamW on MLP (two-moons), CIFAR-10 CNN (ResNet-like, mixed precision), and AG News Transformer text

- **asl_augmentation.ipynb**
  Data augmentation techniques applied to American Sign Language datasets to improve model generalization and reduce overfitting

###  🌐 Computational Physics

  - **PINN_physicNN.ipynb** Physics-Informed Neural Networks (PINNs) with PyTorch.
    Solves the damped harmonic oscillator ODE, covering both forward simulation (given initial conditions) and inverse parameter estimation from
  noisy observations

### 🧬 Bioinformatics & Chemistry & Innovation

- **alphafold.ipynb**
  Exploration or implementation of AlphaFold for protein structure prediction.
  Molecular dynamics simulation of alpha-synuclein (PDB: 1XQ8) using OpenMM with AMBER force fields. Includes RMSD structural analysis, FFT vibrational frequency extraction, and simulation of external       frequency perturbations on protein dynamics

- **Sentinel_Hydrogen_Detection.ipynb**
  A scientific approach to detecting hydrogen presence using satellite (Sentinel) or sensor data.
  Uses Sentinel Hub satellite imagery (Sentinel-2) for anomaly detection in the Meda/A Veiga region (Galicia, Spain). Computes spectral indices (NDVI, clay mineral ratio, ferrous iron ratio), applies
  clustering for mineral/anomaly detection, and visualizes results on interactive Folium maps

### 🌐 AI Research & Innovation: Computer Vision & Generative Models

- **4DGaussians.ipynb**
    Setup and training pipeline for 4D Gaussian Splatting for dynamic scene reconstruction (bouncing balls, animated characters, etc.).

- **StreamDiffusion.ipynb**
  Approach to diffusion models in real-time or streaming contexts.
  Demo of StreamDiffusion for real-time text-to-image and image-to-image generation using diffusion models.

### 🌐 Quantum Research & Innovation

- **quantum_randomness.ipynb**
  Investigates randomness generation using principles of quantum mechanics.
  Explores quantum randomness vs classical pseudo-random number generation. Implements a Monte Carlo estimation of pi using classical (linear congruential), quantum circuit-based

- **qbits_coherence.ipynb**
  Analysis of coherence in qubits, a fundamental concept in quantum computing.
  Simulates photon energy transfer inspired by photosynthesis using QuTiP. Models quantum walks, superposition/coherence in a photonic waveguide, light-harvesting complex interactions, time-crystal-like periodic driving, and decoherence dynamics.

### 🧪 Autonomous AI Research & Innovation: Biophysical Language Models

- **[Autoresearch_QHH_RunPod/](Autoresearch_QHH_RunPod/)**
  A [Karpathy-style autonomous AI research](https://github.com/karpathy/autoresearch) setup designed to run on RunPod with an H100 GPU, with Claude Code as the autonomous research agent running on it.
  The agent autonomously modifies training code, runs experiments (fixed 5-min wall-clock budget each), evaluates results, and keeps or discards changes. Explores whether Hodgkin-Huxley spiking neurons (the biophysical equations governing real ion channels) can replace attention entirely for language modeling, using a multi-timescale HH state-space model with 4 receptor-type channels (AMPA, NMDA, Calcium, Neuromodulatory). Includes three model variants: pure HH-SSM, hybrid HH+Attention, and a GPT baseline with Flash Attention 3 + Muon optimizer.


## 🛠️ Tech Stack

### Languages & Environments
Python · Jupyter Notebook · CUDA/cuDNN · Bash

### Deep Learning & Optimization
PyTorch · TensorFlow · Keras · Torchvision · Mixed Precision Training (AMP) · `torch.compile` · Flash Attention 3 · MuonAdamW · AdamW · Wave-AdamW

### State-Space Models & Biophysical Neural Networks
Hodgkin-Huxley SSM · Mamba-style Gating (Conv1d + SiLU) · Linear Scan Recurrence · Surrogate Gradient Spiking

### Generative AI & Diffusion Models
Diffusers (Hugging Face) · StreamDiffusion · Stable Diffusion · 4D Gaussian Splatting

### Quantum Computing & Simulation
Qibo · QuTiP · Quantum Circuit Simulation · Lindblad Master Equation

### Molecular Dynamics & Bioinformatics
OpenMM · MDAnalysis · BioPython · py3Dmol · AMBER Force Fields

### Computational Physics & Symbolic Math
PINNs (Physics-Informed Neural Networks) · SymPy · PySR (Symbolic Regression) · mpmath · SciPy

### Computer Vision & 3D Reconstruction
OpenCV · 4D Gaussian Splatting · xformers · RoPE · GQA

### NLP & Tokenization
BPE Tokenizers (rustbpe, tiktoken) · Hugging Face Transformers · Hugging Face Datasets

### Data Science & Machine Learning
NumPy · Pandas · scikit-learn · Seaborn · PyArrow

### Geospatial & Remote Sensing
Sentinel Hub · GeoPandas · Shapely · Folium

### NVIDIA Ecosystem & MLOps
cuDF (RAPIDS) · NVTabular · Triton Inference Server · kernels

### Cloud & Autonomous Research
RunPod (H100) · Claude Code (AI Agent) · uv (Python project manager)

### Visualization
Matplotlib · Seaborn · Folium · py3Dmol
