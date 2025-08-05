# Detecting Hidden Backdoors in LLMs

This repository contains experiments for detecting hidden backdoors in Large Language Models (LLMs), focusing on the DeepSeek model. These experiments are part of a Bachelor's thesis project.

## Table of Contents
- [Overview](#overview)
- [Setup & Requirements](#setup--requirements)
- [Usage](#usage)
- [Experiment Descriptions](#experiment-descriptions)
- [Outputs & Logs](#outputs--logs)
- [Troubleshooting & Notes](#troubleshooting--notes)
- [References & Further Reading](#references--further-reading)

## Overview
This project investigates methods for detecting hidden backdoors in LLMs using both black-box and white-box techniques. The primary model used is [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) from Hugging Face.

The experiments:
- Track data flow and activations within the model
- Monitor device (CPU/GPU) transfers
- Log external downloads and suspicious network activity
- Track memory usage during inference
- Inspect model weights for anomalies
- Apply black-box and white-box backdoor detection techniques

## Setup & Requirements

### Python Dependencies
Install the following packages (preferably in a virtual environment):

```bash
pip install torch transformers accelerate psutil scikit-learn requests matplotlib
```

> **Note:** You need a machine with a GPU and sufficient memory to load and run the DeepSeek 8B model.

## Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook src/deepseek_backdoor_experiments.ipynb
```

- The notebook will generate `deepseek_experiment.log` with detailed logs.
- Visualizations and results will be shown inline in the notebook.

## Experiment Descriptions

### 1. Data Flow & Activation Tracking
- Registers hooks on transformer layers to capture and log input/output shapes and activations during inference.
- Visualizes how data moves through the model.

### 2. Device Transfer Tracking (CPU/GPU)
- Logs the device (CPU or GPU) for each model parameter and activation.
- Helps detect unexpected device placements or transfers.

### 3. External Downloads Logging
- Monkey-patches Python’s open() function to log whenever a file is accessed (e.g., model weights, tokenizer files).
- Tracks all local file access during runtime.

### 4. Memory Usage Tracking
- Logs memory usage before and after model inference using `psutil`.
- Detects memory spikes or leaks.

### 5. Network Activity Logging
- Monkey-patches the `requests` library to log all HTTP requests made by the process.
- Detects suspicious or unexpected network activity during model usage.

### 6. Model Weights Inspection
- Logs basic statistics (mean, std, min, max) for each model parameter tensor.
- Helps spot anomalous or outlier weights.

### 7. Backdoor Detection Techniques
- **Trigger Scanning:** Feeds the model suspicious trigger phrases (e.g., `"open sesame"`, `"sudo rm -rf /"`) to check for hidden behaviors or backdoors.
- **Chain of Scrutiny (COS):** Compares reasoning-based and direct answers for consistency, revealing logical contradictions or deceptive generation patterns.
- **Output Anomaly Detection:** Compares outputs against expected keywords to flag missing or incorrect responses.
- **Perplexity/Outlier Detection:** Calculates perplexity of known prompts to detect unusual or uncertain model behavior.
- **Embedding Layer Inspection:** Extracts input embeddings for trigger phrases, and then performs PCA for visualization and KMeans clustering to detect suspicious patterns in embedding space.

## Outputs & Logs
- **Notebook Output:** Key results, anomaly flags, and cluster labels are shown inline.
- **Log File:** All details, including activations, device info, file/network activity, and experiment results, are saved in `deepseek_experiment.log`.
- **Plots:** PCA scatter plot of trigger embeddings is shown using matplotlib.

## Troubleshooting & Notes
- The notebook is resource-intensive due to the size of the DeepSeek model. Ensure you have adequate hardware (GPU with >16GB VRAM recommended).
- If you encounter `meta` device errors, it means some model parameters are not loaded into memory. This can happen with `device_map='auto'` and large models. Try reducing model size or increasing available memory.
- For best results, run in a clean environment and monitor the log file for any suspicious or unexpected activity.
- You can customize the list of triggers, questions, or add more detection techniques as needed.
- These experiments complement additional tests performed with Ollama, Docker, and WireShark.

## References & Further Reading
- [DeepSeek Model on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)

---
