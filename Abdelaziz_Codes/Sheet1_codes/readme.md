## Task2:  Similarity between Different Datasets
- To run the task:
  - > python task2_script.py


## 🎯 Goal
The goal of this task is to analyze the **similarity between network traffic datasets** (provided as `.pcap` files).  
We will compare both **normal** and **attack** traffic across multiple datasets using **Chebyshev distance** and visualize the results using **histograms** and **t-SNE** plots.

---

## 🧩 Task Breakdown

### **a) Flow Generation**
- Extract traffic **flows** from each dataset.  
- Each flow corresponds to a **time window** of **2, 4, or 6 minutes**.  
- Only include communication between **two endpoints** using the **same highest-level protocol**.  
- From each packet sequence, extract:
  - **Packet size**
  - **Direction**
  - **Inter-arrival time**
- Ensure **all flows have equal length** and separate **attack** and **normal** flows.

---

### **b) Chebyshev Distance Between Flows**
- For each dataset and each time interval, compute the **Chebyshev distance** between every pair of flows.  
- The Chebyshev computation must be **implemented manually** — no library functions are allowed.  
- Compute distances **within the same class only**:
  - Attack ↔ Attack  
  - Normal ↔ Normal  

---

### **c) Histograms of Flow-Level Distances**
- Create **six histograms** showing distributions of the Chebyshev distances:
  - 3 time windows (2, 4, 6 minutes)
  - × 2 traffic types (attack & normal)
- Implement the histogram computation **manually** (no external functions).  
- Submit both **source code** and **plots**.

---

### **d) t-SNE Visualization**
- Use **t-distributed stochastic neighbor embedding (t-SNE)** to visualize the flow representations.  
- Each **color** in the plot should represent a **dataset**.  
- Perform **hyperparameter tuning** (e.g., perplexity, learning rate) to improve visualization.  
- Submit both the **source code** and **final plots**.

---

### **e) Packet-Level Distance (QUT Dataset Only)**
- For the **QUT dataset**, compute the **Chebyshev distance** between **every pair of raw packets** (as byte sequences).  
- The computation must be done **without external distance functions**.  
- Compute distances only:
  - Between attack packets  
  - Or between normal packets  

---

### **f) Packet-Level Histograms**
- Create **two histograms**:
  - Attack packets  
  - Normal packets  
- Show the distribution of **Chebyshev distances** for individual packets.  
- Histogram logic must also be implemented manually.

---