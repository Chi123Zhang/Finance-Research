# decagon-polypharmacy-graph-learning
Benchmarking different models on the Decagon dataset for polypharmacy side-effect prediction.
# Graph-Based Polypharmacy Side Effect Prediction

This project benchmarks graph-based learning models on the Decagon dataset for predicting polypharmacy-induced side effects.  
We investigate both binary association prediction and multi-class side effect prediction under a unified graph learning framework.

## Models
The following models are implemented and evaluated:
- Frequency-based baseline
- RESCAL (binary and multi-class settings)
- R-GCN (binary and multi-class settings)

## Dataset
We use the Decagon drug–drug interaction dataset, which contains known side effects induced by combinations of drugs.  
Each drug pair may be associated with one or more side effect types.

## Repository Structure
- `notebooks/` — codes model training, evaluation, and visualization notebooks  
- `report/` — final project report (PDF)  
- `slides/` — forum presentation slides
