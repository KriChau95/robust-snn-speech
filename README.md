# Robust SNN Keyword Classification

This project trains and evaluates a **Spiking Neural Network (SNN)** for **binary keyword classification** on the **Google Speech Commands** dataset. Audio is converted into **spike-train representations** using **Speech2Spikes**, with neuron dynamics implemented via **snnTorch**.

Experiments focus on the keywords **“cat”** and **“dog”**, though any of the six keywords in the dataset can be used.

---

## Dataset

Google Speech Commands (Kaggle):  
https://www.kaggle.com/datasets/neehakurelli/google-speech-commands

## Installation

Create a Python environment and install dependencies:
```bash
pip install numpy torch torchaudio snntorch speech2spikes librosa soundfile matplotlib tqdm
```

## Pipeline

1. Audio augmentation (white noise, pitch shift)
2. Audio-to-spike conversion
3. SNN training on clean or augmented data
4. Evaluation on clean and distorted datasets
5. ANN baseline for comparison

## Usage and Testing

1. To run the pipeline, first generate augmented audio by running ``` python noise.py```, which creates new datasets for white-noise and pitched audio.
2. Next, convert audio to spikes by running:
   -  ```python preprocess.py``` for clean data,
   -  ```python preprocess_distorted.py``` for white-noise data, and
   -  ```python preprocess_pitch.py```for pitched data
   -  these scripts are located in the preprocessing folder, and they write spike arrays and labels into their respective *_processed_spike_data/ folders.
3. Then, train a model using:
   - ```python snn_binary_rate.py``` (clean only)
   - ```python snn_binary_rate_wn.py``` (clean + white-noise combined), or
   - ```python snn_binary_rate_pitch.py ```(clean + pitched combined), which will save a .pth checkpoint.
   - These scripts are located in the snn folder. Running them will also output each epoch ran and their respective training/test loss and accuracy.
4. Finally, evaluate a baseline clean-trained checkpoint with python
   - ```test_og_og.py``` (clean spikes),
   - ```python test_og_wn.py``` (white-noise spikes), or
   - ```python test_og_pitch.py``` (pitched spikes); each test script reports accuracy and is also located in the snn folder.
5. Similar approach used for baseline ANN for comparison against the SNNs.

## Detailed Technical Report

[Paper: Robust Binary Audio Classification with SNNs](Robust_Binary_Audio_Classification_with_SNNs.pdf)

