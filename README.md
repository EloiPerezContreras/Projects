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



  Tech Stack

  Python, PyTorch, TensorFlow/Keras, Qibo, QuTiP, OpenMM, MDAnalysis, BioPython, Sentinel Hub, Folium, Triton Inference Server, CUDA/cuDNN


