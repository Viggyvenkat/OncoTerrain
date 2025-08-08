# OncoTerrain
![Workflow](Workflow.tiff)

```OncoTerrain``` enables clinicians to investigate cellular phenotypes to capture intratumor heterogeneity and developmental trajectories within tumors. It offers a framework to interrogate not only malignant cells, but also microenvironmental populations that exhibit aberrant phenotypes relative to their non-cancerous counterparts.

## Overview 
```OncoTerrain``` is an AI/ML model designed to integrate seamlessly with **AnnData** and Scanpy objects. It is primarily built upon Google’s Tabular Network architecture to predict cell-of-origin, developmental pathways, and cell malignancy in 10x Genomics scRNA-seq datasets.
Our manuscript introduces novel insights into ligand-receptor (LR) interactions, epithelial, stromal, and immune cell dynamics, along with practical demonstrations of OncoTerrain. The model was trained using data from the **CELLxGENE Lung Cancer Atlas (LuCA)** and the **Normal Lung Atlas**, and validated across a diverse range of internal and external scRNA-seq cohorts. All LR interactions were further validated using external 10x Visium spatial transcriptomics datasets.

## Highlights 🌠
1. **Neoplastic gene expression profiles (GEPs)** evolve clonotypically, reflecting differentiation along tumor-specific trajectories. In contrast, non-neoplastic GEPs lack phylogenetic structure under a clonotypic framework.
2. In NSCLC samples driven by oncogenes such as **RAS** and **EGFR**, we observe consistent perturbations in hallmark pathways, including **cell cycle, apoptosis, and epithelial–mesenchymal transition (EMT).**
3. **NSCLC tumors actively remodel their microenvironment**, commonly upregulating **STAT4, CCR7, LAG3** in lymphoid cells and **FAP, ACTA2, COL1A1** in fibroblast populations.
4. Tumorigenesis induces systemic rewiring of key LR interactions, most notably the **MIF–CD74/CD44/CXCR4** axis, the **ANXA1–FPR1**, and **PPIA–BSG** axes.
5. **OncoTerrain can accurately identify malignant tumor cells and abnormal stromal/immune populations**, enabling robust downstream scRNA-seq analyses—without relying on copy number alteration (CNA) inference. It integrates seamlessly into existing **AnnData-based** workflows 

## How to Navigate 🔄
This GitHub repository contains all the scripts necessary to preprocess data and reproduce figures presented in our manuscript. The codebase is structured into the following key directories:

```
OncoTerrain/ 
|-- src/
|   |-- fig-generation/
|   |   |-- copyKAT-val.R
|   |   |-- figure-1.py
|   |   |-- figure-2.py
|   |   |-- figure-3.py
|   |   |-- figure-4-spatial.py
|   |   |-- figure-5.py
|   |   |-- tcga-val.R
|   |   |-- creating-vectors.py
|   |   |-- figure-4-cellchat.R
|   |-- oncocli/
|   |   |-- OncoTerrain.py
|   |   |-- __init__.py
|   |   |-- OncoTerrain.joblib
|   |   |-- oncocli.py
|   |-- preprocessing/
|   |   |-- preprocessing.py
|   |   |-- main.py
|-- setup.py
|-- MANIFEST.in
```

Our preprocessing pipeline is robust and supports additional datasets, as long as they are placed in the ```../../data/ directory``` relative to ```preprocessing.py```. 

**Note:** We are not distributing the contents of ```src/fig-generation``` or ```src/TMEGPT``` as standalone packages, as these modules are tightly integrated into the CLI. Figure generation scripts live under ```src/fig-generation``` and exploratory/data cleaning scripts are located within ```src/TMEGPT```. 

Documentation is provided via **docstrings** throughout the codebase for clarity. The trained model is serialized using ```joblib```, enabling easy reuse and sharing of the model state. If you do not wish to use the CLI, you can load the model and access its components as follows: 

``` 
self.model_bundle = joblib.load("OncoTerrain.joblib")
self.OncoTerrain = self.model_bundle['model']
self.model_features = self.model_bundle['features']
```

in order to access the features and the OncoTerrain model object. 

## OncoTerrain CLI 🖥️
We’ve built a CLI to help clinicians and researchers easily interact with OncoTerrain from the terminal.

### Installation
To install OncoTerrain, simply run: 
```pip install oncoterrain``` 

To view all available commands and usage options, run: 
```oncoterrain --help``` 

### Running OncoTerrain

For a **single sample** in either 10x or adata format, please run: 

```oncoterrain infer {path/to/10x_sample_dir} --output-dir {output_dir} --no-save-adata```
- ```--output-dir```: Specifies the directory
- ```--no-save-adata```: If this flag is present then the adata will not be saved to output_dir

For a **group of 10x-style subfolders**, please run:

```oncoterrain batch {path/to/10x_sample_dirs} --output-dir {output_dir}  ```
- ```--output-dir```: Specifies the directory

```oncoterrain batch``` will:
- Iterate each folder in ```{path/to/10x_sample_dirs}```
- Skip any that aren’t valid 10x directories
- Write each sample’s outputs under ```{output_dir} ```

## Computing Resources 💻
All development and model training were conducted on the Rutgers [Amarel HPCU](https://oarc.rutgers.edu/resources/amarel/) using nodes equipped with 256 GiB RAM and 32 dedicated CPU cores.

## Team 👥
**Contributor(s):** Vignesh Venkat & Subhajyoti De, PhD \
**Contact:** ```vvv11@scarletmail.rutgers.edu```

## References 📄
If you use OncoTerrain, its methods, or its insights in your research, we kindly request you cite:
1. Hu, X. et al. Deconstructing evolutionary histories of complex rearrangements during tumorigenesis in lung. bioRxiv
2. Venkat, V. et al. Disruptive changes in tissue microenvironment prime oncogenic processes at different stages of carcinogenesis in lung. bioRxiv, 1-20 (2024). 

## Acknowledgements:
The authors acknowledge scholarly input from other members of Rutgers Cancer Institute & special thanks to members of the [De Laboratory](https://www.sjdlab.org/).