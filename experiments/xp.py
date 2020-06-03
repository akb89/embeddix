"""Compute RMSE across same-dim models, after applying AO+scaling."""
import os

import itertools
import numpy as np

import matrixor


if __name__ == '__main__':
    INPUT_DIRPATH = '/Users/akb/Github/embeddix/models/original'
    MODELS_PATH = [os.path.join(INPUT_DIRPATH, filename) for filename in
                   os.listdir(INPUT_DIRPATH) if filename.endswith('.npy')]
    for A_path, B_path in itertools.combinations(MODELS_PATH, 2):
        A_name = os.path.basename(A_path).split('.npy')[0]
        B_name = os.path.basename(B_path).split('.npy')[0]
        A = np.load(A_path)
        B = np.load(B_path)
        rmse = matrixor.align(A, B)
        print('{}\t{}\tRMSE = {}'.format(A_name, B_name, rmse))
