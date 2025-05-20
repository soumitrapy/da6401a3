# Seq2Seq Transliteration on Dakshina Dataset

This project implements sequence-to-sequence models for transliteration using the Dakshina dataset. It includes:

* Vanilla seq2seq (RNN/GRU/LSTM)
* Attention-based seq2seq model
* Hyperparameter tuning using Weights & Biases sweeps
* Evaluation and prediction saving

---

## Project Structure

```
.
├── configs/
│   ├── vanilla_sweep.yaml
│   └── attention_sweep.yaml
├── data/
│   └── dakshina_dataset_v1.0/
│       └── <language_code>/
│           └── lexicons/
│               ├── train.tsv
│               ├── dev.tsv
│               └── test.tsv
├── models/
│   ├── seq2seq_vanilla.py
│   └── seq2seq_attention.py
├── train/
│   ├── train_vanilla.py
│   └── train_attention.py
├── utils/
│   ├── data_utils.py
│   ├── train_utils.py
│   └── evaluate.py
├── predictions_vanilla/
│   └── *.txt
├── predictions_attention/
│   └── *.txt
├── wandb_sweeps/
│   └── (optional logs)
├── README.md
└── requirements.txt
```

---

## Setup

1. Clone the repo:

```bash
git clone https://github.com/your_username/dakshina-transliteration.git
cd dakshina-transliteration
```

2. Install dependencies (recommend using virtualenv or conda):

```bash
pip install -r requirements.txt
```

3. Download Dakshina dataset from [GitHub](https://github.com/google-research-datasets/dakshina) and place the language folder (e.g., `hi`) under `data/dakshina_dataset_v1.0/`

---

## Training

### Vanilla Seq2Seq Model

Run training with default parameters:

```bash
python train/train_vanilla.py --config configs/vanilla_sweep.yaml
```

### Attention-based Seq2Seq Model

Run training:

```bash
python train/train_attention.py --config configs/attention_sweep.yaml
```

---

## Hyperparameter Sweeps with Weights & Biases

1. Login to wandb:

```bash
wandb login
```

2. Create a sweep for vanilla model:

```bash
wandb sweep configs/vanilla_sweep.yaml
```

3. Start agent to run experiments:

```bash
wandb agent <SWEEP_ID>
```

Repeat similarly for attention model:

```bash
wandb sweep configs/attention_sweep.yaml
wandb agent <SWEEP_ID>
```

---

## Evaluation

Evaluate best model on test set and save predictions:

```bash
python utils/evaluate.py --model_path path/to/best_model.pth --test_data data/dakshina_dataset_v1.0/<lang>/lexicons/test.tsv --output predictions_vanilla/
```

Or for attention model:

```bash
python utils/evaluate.py --model_path path/to/best_attention_model.pth --test_data data/dakshina_dataset_v1.0/<lang>/lexicons/test.tsv --output predictions_attention/
```

---

## Visualizations

* Use wandb dashboard to view:

  * Accuracy vs experiment runs
  * Parallel coordinates plot of hyperparameters
  * Correlation tables of hyperparameters with accuracy/loss

* Attention heatmaps for selected test samples are saved in `utils/visualize_attention.py`

---

## Notes and Tips

* Adjust batch sizes and learning rates according to your hardware.
* Use early stopping in training to avoid overfitting.
* Dropout is applied after embeddings and before RNN layers for regularization.
* Beam search can improve decoding but increases inference time.

---
