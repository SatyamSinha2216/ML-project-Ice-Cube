# IceCube: Neutrinos in Deep Ice, Direction Prediction with a GNN

**Author:** Satyam Sinha
**Competition:** [IceCube - Neutrinos in Deep Ice](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice) (Kaggle)

# Overview
The IceCube detector records light pulses from neutrino interactions in Antarctic ice using 5,160 sensors.
The task is to predict each neutrino's direction (azimuth and zenith) from its variable-length set of pulses.
The metric is the mean angular error in radians (lower is better). A constant guess scores about 1.57 rad (90°) which is about average and we need to beat this.

# Result
| Metric| Score 
|---|---|
 Public leaderboard | 1.1357 
 Private leaderboard | 1.1359 (about 65°) 

Submitted after the competition deadline (late submission). Rank: [roughly around 588 due to late submission of Kaggle competition].
The model was trained on about 1M events, roughly 0.15% of the training data, because of compute limits.

# Repository contents
| File | Description |
|---|---|
| `[Ice Cube training ... .ipynb]` | Training notebook: EDA, preprocessing, GNN model, streamed training over batches 2 to 6 |
| `[IceCube-Deep N... .ipynb]` | Inference notebook: loads the trained weights and writes `submission.csv` |
| `gnn_stream_b6.pt` | Trained model weights (final checkpoint, used for the submission) |
| `README.md` | This file |

# Approach
- **Preprocessing:** each event is capped at 128 pulses (non-auxiliary first, then earliest). Features are
  x, y, z, time (relative to the event's first pulse), log charge and the auxiliary flag, plus a padding mask.
- **Model:** an EdgeConv graph neural network in PyTorch. Pulses are nodes linked to their
  k = 8 nearest neighbours in (x, y, z, time). It has 3 message-passing layers, mean and max pooling, and
  a head that outputs a unit direction vector. It has 232,707 parameters.
- **Training:** loss `1 - cos(angle)`, AdamW (Stochastic Optimization method), OneCycle learning rate schedule, batch size 256, streamed over
  batches 2 to 6 (2 epochs each). Validation used 10,000 events from batch 1.

# How to run
This is a Kaggle code competition, and the notebooks are written for Kaggle paths.
1. Create a Kaggle notebook and attach the IceCube competition data.
2. **Training:** run the training notebook. It saves a checkpoint after each batch (`gnn_stream_b2.pt` to `gnn_stream_b6.pt`).
3. **Inference:** upload `gnn_stream_b6.pt` as a Kaggle dataset and attach it, then run the inference notebook
   with a GPU enabled. It finds the data and weights paths automatically and writes `submission.csv`.

To run locally you need Python 3.10+, `torch`, `numpy`, `pandas` and `pyarrow`, plus the competition data.
Edit the `DATA` and `WEIGHTS` paths at the top of each notebook.

## Possible improvements
Train on many more batches, rebuild the graph from learned features (dynamic graph), add event-centred
position features, use a larger model, and try a von Mises-Fisher loss.

## Note
The required task may take really long to compute even for 5-6 batches. So any other more batches would require extensive time for processing and checking it.
