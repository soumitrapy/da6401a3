# Assignment 3 DA24M022
# Project Structure
```
dakshina-transliteration/
│
├── data/
│   └── dakshina_dataset_v1.0/      # Extracted dataset (not pushed to GitHub)
│
├── models/
│   └── seq2seq.py                  # Seq2Seq model class
│
├── utils/
│   ├── dataset.py                  # Dataset loading & preprocessing
│   └── vocab.py                    # Vocabulary helpers
│
├── train.py                        # Regular training script
├── sweep.py                        # wandb sweep logic
├── sweep_config.yaml               # wandb sweep configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project description and instructions
└── .gitignore                      # Ignore datasets, checkpoints, etc.
```

# Dakshina Transliteration with wandb Sweeps

This project trains a Seq2Seq model on the Dakshina dataset (e.g., Hindi) to perform transliteration from Latin script to native script.

## Setup

```bash
git clone https://github.com/yourname/dakshina-transliteration
cd dakshina-transliteration
pip install -r requirements.txt
```
* Download the dataset
```
wget https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.zip -P data/
unzip data/dakshina_dataset_v1.0.zip -d data/
```
* Train
```
python train.py
```
Run sweep
```
wandb sweep sweep_config.yaml
wandb agent your-sweep-id
```