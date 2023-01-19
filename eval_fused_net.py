import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader

from utils.eval import eval_fused_model
from utils.fused_data_loader import FusedData, LABELS
from utils.best_model_params import fused_num_perceptrons, fused_num_hidden_layers
from models.fused_net import FusedNet

# decide which data modalities to use

use_img = True
use_spec = True
use_imu = True
use_models = {
    "img": use_img,
    "spec": use_spec,
    "imu": use_imu
}

# load model(s)
model_path = os.path.dirname(os.path.realpath(__file__)) + "/weights/"
model_files = [
    model_path + "val_fused_img_spec_imu_20230118_223719_23.wts",
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

data_dir = os.path.dirname(os.path.realpath(__file__)) + "/data/all_data_fused_labeled/"

test_data = FusedData(data_dir, use_models, test=True)
N_test = len(test_data)
batch_size = 2
num_workers = 8
test_loader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

# set up model
n_models = 3
n_classes = len(LABELS)

loss_fn = nn.CrossEntropyLoss()

# evaluate models on test set
label_names = [LABELS[l] + " " for l in LABELS]

for i, file in enumerate(model_files):
    model = FusedNet(
        use_models, n_classes, fused_num_hidden_layers, fused_num_perceptrons
    ).to(device).double().eval()
    model.load_state_dict(torch.load(file))
    results = eval_fused_model(model, test_loader, device, loss_fn, N_test)
    avg_test_loss, accuracy, true_labels, pred_labels = results
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
    plt.title("Fused Confusion Matrix", fontsize=18)
    plt.savefig("results/fused_conf_mat_{}.png".format(i))
    plt.show()
    print("f1 score:", f1_score(true_labels, pred_labels, average='macro'))