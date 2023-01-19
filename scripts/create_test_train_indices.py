import os
import numpy as np
from sklearn.model_selection import train_test_split

total_files = len(os.listdir(os.path.abspath('./data/all_data_fused_labeled'))) // 4
indices = np.arange(1, total_files)
X_train, X_test, _, _ = train_test_split(indices, indices, test_size=0.2, random_state=1)
X_val, X_test, _, _ = train_test_split(X_test, X_test, test_size=0.50, random_state=1)

# Save values
np.save("./data/all_data_fused_labeled/train_idx.npy", X_train)
np.save("./data/all_data_fused_labeled/val_idx.npy", X_val)
np.save("./data/all_data_fused_labeled/test_idx.npy", X_test)