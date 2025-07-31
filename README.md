# All Preprocessing & Method Development of OncoTerrain
OncoTerrain enables clinicians to investigate cell phenotypes to capture intratumor heterogeneity and tumor developmental trajectories. The output of ```OncoTerrain``` can be utilized to truly investigate malignant cells as well as microenvironmental cells that exhibit abnormal phenotypes respective to their non-cancer counterparts.

### Overview 
---
OncoTerrain is a powerful AI/ML Model, built to support AnnData & scanpy objects. OncoTerrain is largely based on Google's Tabular Network to predict cells of origin, developmental trajectories, and cell malignancy in 10x genomics scRNA-seq data. Our paper goes into novel ligand-receptor (LR) interactions; Epithelial, Stromal, Immune cell dynamics, and demoing OncoTerrain. OncoTerrain was trained using the CELLxGENE Lung Cancer Atlas (LuCA) and the Normal Lung Atlas and applied to a variety of in-house and external scRNA-seq cohorts. All LR interactions were validated using external 10x Visium spatial RNA-seq cohorts.  

### Highlights
---
1. Gene Expression Profiles (GEPs) undergo systematic change along clonotypic lines.
2. Major oncogenic drivers i.e. RAS and EGFr-driven NSCLC samples **experience perturbation to hallmark pathways** such as Cell Cycle, Apoptotis, and Epithelial-Mesenchymel transition. 
3. **NSCLC tumors remodel their microenvironment**, often upregulating STAT4, CCR7, LAG3 in Lymphoid cells and FAP, ACTA2, COL1A1 in Fibroblast cells. 
4. We observed systemic change to the **MIF ligand's interaction with CD74, CD44, CXCR4 receptors** as well as the **ANAX1-FPR1**, **PPIA-BSG** axes through tumorigenesis.
5. OncoTerrain can identify malignant cells, to aide in downstream scRNA-seq analytics, rapidly, accurately, and **seamlessly into AnnData-based workflows**. 

### How to Navigate
---
This GitHub Repository contains all of the scripts required to preprocess and generate any of the figures in our manuscript. We have divided our codebase into easily navigable sections listed below:

```
OncoTerrain-paper/ 
|-- src/
|   |-- fig-generation/
|   |   |-- copyKAT-val.R
|   |   |-- figure-1.py
|   |   |-- figure-2.py
|   |   |-- figure-3.py
|   |   |-- figure-4-spatial.py
|   |   |-- figure-5.py
|   |   |-- tcga-val.R
|   |-- TMEGPT/
|   |   |-- OncoTerrain.py
|   |   |-- __init__.py
|   |   |-- preprocessing.py
|   |   |-- main.py
|-- notebooks/
|   |-- creating_vectors.ipynb
|   |-- figure4-cellchat.ipynb
```

Our preprocessing script has been engineered to handle other samples as well, as long as they are in the: ``` ../../data ``` respective of where the preprocessing.py script lies. All of the code for the figures lies in ```src/fig-generation``` or in ```notebooks```, whereas all of the data cleaning, preprocessing, and exploration lies within ```src/TMEGPT```. All of the documentation (in the form of docstrings) lies within the files for futher, clear explanations. For utilizing OncoTerrain, we have set up a seperate [OncoTerrain repository](https://github.com/Viggyvenkat/OncoTerrain.git) with instructions on how to utilize our model for inference.

### Computing Resources
---
All development was conducted on the Rutgers [Amarel HPCU](https://oarc.rutgers.edu/resources/amarel/) with 256GiB of RAM and 32 dedicated cores

### References
---
If you use OncoTerrain, its methods, or its insights in your research, we kindly request you cite:
1. Hu, X. et al. Deconstructing evolutionary histories of complex rearrangements during tumorigenesis in lung. bioRxiv
2. Venkat, V. et al. Disruptive changes in tissue microenvironment prime oncogenic processes at different stages of carcinogenesis in lung. bioRxiv, 1-20 (2024). 

### Acknowledgements:
The authors acknowledge scholarly input from other members of Rutgers Cancer Institute & special thanks to members of the [De Laboratory](https://www.sjdlab.org/).