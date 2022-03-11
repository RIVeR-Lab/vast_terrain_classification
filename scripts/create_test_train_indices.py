import numpy as np
from sklearn.model_selection import train_test_split

indices = np.arange(1, 87301)
X_train, X_test, _, _ = train_test_split(indices, indices, test_size=0.2, random_state=1)
X_val, X_test, _, _ = train_test_split(X_test, X_test, test_size=0.50, random_state=1)

# Save values
np.save("../train_idx.npy", X_train)
np.save("../val_idx.npy", X_val)
np.save("../test_idx.npy", X_test)