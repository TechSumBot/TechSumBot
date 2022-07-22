# Biased TextRank
This repository contains code and data for our paper: 
**Biased textrank: Unsupervised Graph-Based Content Extraction: Ashkan Kazemi, Veŕonica Pérez-Rosas, and Rada Mihalcea. COLING 2020**.

Biased TextRank is an unsupervised, graph-based method for extracting content from text with a given focus. In this repository,
you can find code for two experiments described in our paper; 1) focused summarization of US presidential debates and 2)
supporting explanation extraction for fact-checking of political claims. 

### Requirements
To install the required packages for running the codes on your machine, please run ``pip install -r requirements.txt``
first. 

### Content
* ``/data/``: This directory contains the two datasets used in the experiments. The ``/data/liar/`` directory contains files
for the LIAR-PLUS dataset. The ``/data/us-presidential-debates/``  directory contains the novel presidential debates 
dataset described in the paper.
* ``/src/``: This directory contains implementations of the described experiments in the paper. To run the *biased summarization*
experiment, run ``/src/biased_summarization.py``. For the *explanation extraction* experiment, run 
``/src/explanation_generation.py``. 

### Citation
If you plan to use our methods or data, please cite our work using the following bibtex:

```
@inproceedings{kazemi-etal-2020-biased,
title={Biased TextRank: Unsupervised Graph-Based Content Extraction},
author={Kazemi, Ashkan and P{\'e}rez-Rosas, Ver{\'o}nica and Mihalcea, Rada},
booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
year={2020}
}
```
