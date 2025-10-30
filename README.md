# ParlaMint Exploration

This repo aims to explore the [ParlaMint](https://clarin.eu/parlamint) dataset, focusing initially on the Italian English version records (Further exploration could involve multilingual comparisons).

The goal is to understand the dataset and prepare for future tasks. Some ideas to explore include sentiment, stance detection, or other aspects of parliamentary language.

## Todo ✅

- [x] Understand dataset structure and metadata
- [x] Build a simple interactive dashboard to explore the data

  - Note: Still WIP but moved to [parlamint-visual-dashboard](https://github.com/DinosaurMauricio/parlamint-visual-dashboard)

- [ ] Experiment multi-classification using a pretrained LLM (e.g., RoBERTa).
- [ ] Optimize code to have reusable components to use in Colab, no other way to run it :(
- [ ] Explore ideas other ideas

```
parlamint-nlp/
│
├─ dataset/ # Dataset classes
├─ model/ # Model definitions
├─ notebooks/ # Jupyter notebooks to experiment with the data
├─ training/ # Class for model training and validation.
├─ utils/ # Helper functions and utilities (file loader, collate, seed, etc.)
├─ config.yaml # Hyperparameters and configuration
└─ main.py
```
