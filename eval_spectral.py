import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch import nn

from utils import eval_model, SpectralData, LABELS
from models import MLP

# load models
model_path = os.path.dirname(os.path.realpath(__file__)) + "/weights/"
model_files = [
    # model_path + "spectral/spectral_20220228_170159_74",
    model_path + "spectral/spectral_20220301_231905_81",
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

data_path = os.path.dirname(os.path.realpath(__file__)) + "/data/"
train_test_val_split = [0.6, 0.2, 0.2]
batch_size = 16
num_workers = 4
spectral_data = SpectralData(data_path, train_test_val_split, batch_size, 
                             num_workers, load_data=True)
D = spectral_data.data_dim
K = spectral_data.n_classes
num_perceptrons = [256, 128]
num_hidden_layers = len(num_perceptrons)

loss_fn = nn.CrossEntropyLoss()

# evaluate models on test set

label_names = [LABELS[l] + " " for l in LABELS]

for file in model_files:
    model = MLP(D, K, num_hidden_layers, num_perceptrons).to(device).double()
    model.load_state_dict(torch.load(file))
    results = eval_model(model, spectral_data, device, loss_fn)
    avg_test_loss, accuracy, true_labels, pred_labels = results

    # view confusion matrix 
    # (i, j is # samples with true label i and pred label j)
    cm = confusion_matrix(true_labels, pred_labels)

    class_names_cap = list(map(lambda x: x.capitalize(), label_names))
    plt.rcParams["font.family"] = "Times New Roman"
    cm = confusion_matrix(true_labels, pred_labels)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cmn, annot=True, fmt='.3f', xticklabels=class_names_cap, yticklabels=class_names_cap, cmap="viridis")
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=45, fontsize=12) 
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title("Spectral Confusion Matrix", fontsize=18)
    plt.savefig("results/spectral/spectral_conf_mat.png")
    print("f1 score:", f1_score(true_labels, pred_labels, average='macro'))
    plt.show()