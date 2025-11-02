# Adversarial Machine Learning for ICS Anomaly Detection

## Overview
This project is part of the **Study Project: Adversarial Machine Learning – Use of Reverse Engineering for Anomaly Detection in Industrial Control Systems (ICS)** at **Brandenburg University of Technology (BTU)**, under the supervision of **Prof. Dr.-Ing. Andriy Panchenko** and **Asya Mitseva, M.Sc.**.  
The goal is to develop and evaluate machine learning-based intrusion detection systems (IDS) for ICS networks using datasets **Electra (S7comm)** and **QUT S7comm**. The focus lies on analyzing proprietary network protocols, extracting meaningful statistical patterns, computing flow-based similarity metrics, and implementing a **Stacked Denoising Autoencoder (SDA)** for unsupervised anomaly detection.

---

## Project Structure
### **Task 1 – Statistical Analysis of ICS Network Traffic**
Perform statistical analysis on PCAP-based network captures to understand:
- Total packet counts (normal vs. attacked).  
- Application- and transport-layer protocol distributions.  
- Average, median, and standard deviation for packet lengths and inter-arrival times.  
- Host-pair communication patterns and per-protocol timing characteristics.  
- Generation of **CDF plots** for packet header and payload sizes **(without using prebuilt CDF functions)**.

### **Task 2 – Similarity Between Datasets**
Generate network **flows** for 2-, 4-, and 6-minute windows and compute similarity metrics:
- Build flows based on packets exchanged between unique host pairs over the same highest-level protocol.  
- Extract features: **packet size, direction, and inter-arrival time**.  
- Compute **Chebyshev distances** manually (no external library use).  
- Create **histograms** (custom implementation) for normal and attacked flows.  
- Apply **t-SNE** visualization to project high-dimensional flow data, tuning hyperparameters for best clustering.  
- Compute **packet-level Chebyshev distances** and plot histograms to compare benign vs. malicious packets.

### **Task 3 – Unsupervised Detection of Network Attacks**
Implement and train a **Stacked Denoising Autoencoder (SDA)** for unsupervised intrusion detection:
- Input: `m × n` packets (where `n` = number of bytes per packet).  
- Model supports multiple **layer types** and **activation functions** (e.g., ReLU, Sigmoid, Tanh).  
- Optimize **Mean Squared Error (MSE)** for normal operation data.  
- Save learned **feature encodings** for further analysis.  
- Implement a **threshold-based classifier** that flags a packet as under attack if its average reconstruction error exceeds a user-defined γ.  

---

## Technologies Used
- **Python 3.10+**  
- **NumPy, Pandas** (data handling)  
- **Matplotlib** (for visualization)  
- **Scikit-learn / TensorFlow / PyTorch** (for deep learning)  
- **TShark / Scapy / Pyshark** (for PCAP processing)

---

## Datasets
- [**Electra (S7comm)**](http://perception.inf.um.es/ICS-datasets/csv/electra_s7comm.zip)  
- [**QUT S7comm**](https://github.com/qut-infosec/2017QUT_S7comm)  
Both datasets provide **labeled ICS network traffic** with attack and normal segments, captured from different testbeds.

---

## Expected Output
- Statistical summaries and CDF plots (Task 1).  
- Chebyshev-based histograms and t-SNE projections (Task 2).  
- SDA training results, reconstruction errors, and attack classification outcomes (Task 3).  
- Source code and plots must be submitted via **Moodle** and demonstrated in a **40-minute Q&A session**.

---
