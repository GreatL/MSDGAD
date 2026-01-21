# MSDGAD: Multi-Scale Temporal Modeling for Edge-Level Anomaly Detection in Dynamic Graphs

This repository contains the official implementation of **MSDGAD**, a multi-scale
dynamic graph anomaly detection framework for **edge-level anomaly detection** in
dynamic graphs.

The code accompanies the paper:

> **Multi-Scale Temporal Modeling for Edge-Level Anomaly Detection in Dynamic Graphs**  

---

## Overview

Dynamic graphs are ubiquitous in real-world applications such as financial trust
networks and communication systems. MSDGAD addresses the problem of detecting
**anomalous interactions (edges)** in dynamic graphs by jointly modeling structural
information and heterogeneous temporal dynamics.

The key features of MSDGAD include:

- **Edge-level anomaly detection** under evolving graph structures
- **Node-aligned induced subgraphs** to handle node inconsistency over time
- **Multi-scale temporal modeling** that captures both short-term irregularities
  and long-term behavioral trends
- A unified framework applicable to both financial and communication networks

An overview of the framework
![](https://github.com/GreatL/MSDGAD/raw/main/MSDGAD.png)
---

## Environment Setup

The code is implemented in **Python 3.8+**.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Datasets
We evaluate MSDGAD on the following real-world dynamic graph datasets:

- Bitcoin-Alpha and Bitcoin-OTC: Financial trust networks with signed edges.
- CollegeMsg: Email communication network.
- Wikipedia Talk: Large-scale communication network.

## Reproducing Experimental Results
To reproduce the main results reported in the paper:
- Download the datasets and place them in the data/ directory.
- Install dependencies using requirements.txt.
- Run train.py for MSDGAD and baseline methods.
- Evaluation results (AUC) will be printed to the console.


