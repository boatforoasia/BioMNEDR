## Overview

![overview](/img/graphical_abstract.png)

## Data
All data in this manuscript complied from numerous public data sources including:<br> 
[DrugBank 5.1.1](https://go.drugbank.com/)<br>
[Drug Repurposing Hub](https://clue.io/repurposing)<br>
[Drug Repurposing Database](http://apps.chiragjpgroup.org/repoDB/)<br>
[Drug Indication Database]<br>
[DisGeNet](https://www.disgenet.org/)<br> 
[Disease Ontology](https://disease-ontology.org/)<br>
[HUGO](https://www.genenames.org/)<br> 
[the Unified Medical Language System](https://www.nlm.nih.gov/research/umls/index.html)<br>
[the Biological General Repository for Interaction Datasets](https://thebiogrid.org/)<br> 
[the Database of Interacting Proteins](https://dip.doe-mbi.ucla.edu/dip/Main.cgi)<br>
[the Human Reference Protein Interactome Mapping Project](http://www.interactome-atlas.org/)<br>
[the Gene Ontology]<br>
[Gene Ontology Plus](ttp://geneontology.org/)<br>
[the Broad Connectivity Map](https://clue.io/cmap)<br>
[the Pharmacogenomics Knowledgebase](https://www.pharmgkb.org/)<br>
[Uberon](http://uberon.github.io/)<br>
[the Cell Ontology](http://www.obofoundry.org/ontology/cl.html)<br>
[the Human Cell Atlas Ontology](https://github.com/HumanCellAtlas/ontology).

## Repository layout
- [data/](data/) — raw input TSVs.  
- [msi/](msi/) — graph component classes and loader.  
- [embedding_for_BioMNEDR/](embedding_for_BioMNEDR/). 
- [feature/](feature/) — precomputed feature files used in the paper (embedding vectors).  
- [model/](model/) — saved trained models.  
- [main.py](main.py) — training pipeline and utilities.  
- [case_study_demo.py](case_study_demo.py) — demo to run inference with a saved model (uses [`case_study_demo.load_features`](case_study_demo.py) and [`case_study_demo.main`](case_study_demo.py)).  
- [requirements.txt](requirements.txt) — Python dependencies.

## Quickstart
1. Create Python environment and install deps:
   ```sh
   pip install -r requirements.txt
2. Train model (using precomputed features):
   ```sh
   python main.py
3. Run case study:
   ```sh
   python case_study_demo.py
