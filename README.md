# OncoTerrain

![Workflow](Workflow.tiff)

**OncoTerrain** enables clinicians to investigate cellular phenotypes to capture intratumor heterogeneity and developmental trajectories within tumors. It provides a reproducible, pathway-aware framework for exploring both malignant and non-malignant cell populations.

## Overview

`OncoTerrain` integrates seamlessly with **AnnData** and **Scanpy**, leveraging deep learning architectures inspired by Google’s Tabular Network.  
It predicts cell-of-origin, developmental trajectories, and malignant potential in 10x Genomics scRNA-seq datasets.

Our manuscript demonstrates its use in analyzing ligand–receptor interactions, tumor–microenvironment dynamics, and malignant evolution in non-small-cell lung cancer (NSCLC).  
The model was trained using **CELLxGENE Lung Cancer Atlas (LuCA)** and the **Normal Lung Atlas**, and validated across multiple independent datasets.

## Highlights
1. Neoplastic gene-expression profiles (GEPs) reflect differentiation along tumor-specific trajectories, while non-neoplastic GEPs lack clonotypic structure.  
2. Oncogene-driven NSCLCs (e.g., *RAS*, *EGFR*) show reproducible perturbations in hallmark pathways such as cell cycle, apoptosis, and EMT.  
3. Tumors remodel their microenvironment, upregulating markers such as **STAT4**, **CCR7**, **LAG3** (lymphoid) and **FAP**, **ACTA2**, **COL1A1** (fibroblast).  
4. Systemic rewiring of key ligand–receptor axes (e.g., **MIF–CD74/CD44/CXCR4**, **ANXA1–FPR1**, **PPIA–BSG**) emerges as a hallmark of tumorigenesis.  
5. `OncoTerrain` accurately identifies malignant cells and abnormal stromal/immune populations without requiring CNA inference.

## Installation
``` pip install oncoterrain ```

## Directory Structure 

```
OncoTerrain/ 
|-- src/ 
|   |-- oncocli/ 
|   |   |-- OncoTerrain.py 
|   |   |-- __init__.py 
|   |-- preprocessing/  
|   |   |-- preprocessing.py 
|   |   |-- main.py 
|-- setup.py 
|-- MANIFEST.in
```

### Tutorials & Vignettes
For complete tutorials and end-to-end examples, visit the vignettes directory

### Computing Resources
All model development and training were performed on the Rutgers Amarel HPCU with nodes providing 256 GiB RAM and 32 CPU cores.

### Team
Contact: vvv11@scarletmail.rutgers.edu

### References
- Hu X. et al. Deconstructing evolutionary histories of complex rearrangements during tumorigenesis in lung. bioRxiv.
- Venkat V. et al. Disruptive changes in tissue microenvironment prime oncogenic processes at different stages of carcinogenesis in lung. bioRxiv (2024).

### Acknowledgements
We thank colleagues at the Rutgers Cancer Institute and members of the De Laboratory for their guidance and support.
