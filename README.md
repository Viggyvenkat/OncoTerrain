# All Preprocessing & Method Development of OncoTerrain

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

Our preprocessing script has been engineered to handle other samples as well, as long as they are in the: ``` ../../data ``` respective of where the preprocessing.py script lies.

### Computing Resources
All development was conducted on the Rutgers [Amarel HPCU](https://oarc.rutgers.edu/resources/amarel/)

### References
If you use OncoTerrain or its insights in your research, make sure to cite:
1. Hu, X. et al. Deconstructing evolutionary histories of complex rearrangements during tumorigenesis in lung. bioRxiv
2. Venkat, V. et al. Disruptive changes in tissue microenvironment prime oncogenic processes at different stages of carcinogenesis in lung. bioRxiv, 1-20 (2024). 