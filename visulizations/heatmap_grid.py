import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


def plot_attention_heatmaps(model, test_loader, trg_vocab, device, num_samples=9):
    model.eval()
    samples = 0
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)
            encoder_outputs, hidden = model.encoder(src)
            input_token = trg[:, 0]
            for i in range(1, trg.size(1)):
                output, hidden = model.decoder(input_token, hidden, encoder_outputs)
                attn_weights = model.attention(hidden, encoder_outputs).cpu().numpy()

                # Plot heatmap
                sns.heatmap(attn_weights, ax=axes[samples], cbar=False)
                axes[samples].set_title(f"Sample {samples+1} Attention")
                axes[samples].set_xlabel("Input sequence")
                axes[samples].set_ylabel("Attention weight")
                samples += 1
                if samples >= num_samples:
                    break
            if samples >= num_samples:
                break

    plt.tight_layout()
    plt.show()
