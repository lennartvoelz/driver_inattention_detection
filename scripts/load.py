import numpy as np
import pandas as pd

classes_aug = pd.read_csv('../data/labels_aug_2000.csv', delimiter=',', index_col=0).to_numpy()

X = np.load('../data/data_aug_2000.npy')

X_aug = X/255.0
y_aug = classes_aug

np.save('../data/X_aug.npy', X_aug)
np.save('../data/y_aug.npy', y_aug)