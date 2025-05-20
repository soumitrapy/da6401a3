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