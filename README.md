# All Preprocessing & Method Development of OncoTerrain

This GitHub Repository contains all of the scripts required to preprocess and generate any of the figures in our manuscript. We have divided our codebase into easily navigable sections listed below:

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

Our preprocessing script has been engineered to handle other samples as well, as long as they are in the:

``` ../../data ```

respective of where the preprocessing.py script lies