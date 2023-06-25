from __future__ import division, absolute_import, print_function

import os
import sys
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import random
import json
import pickle
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torchvision.datasets as Datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision.utils import save_image
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist, load_cifar10

def set_seed(args):
    """
    :param args:
    :return:
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

def normalize_mean(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

def normalize_linear(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test

class GetLoader(Data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    
    def __len__(self):
        return len(self.data)

def get_tpr_fpr(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    AP = np.sum(true_labels)
    AN = np.sum(1-true_labels)

    tpr = TP/AP if AP>0 else np.nan
    fpr = FP/AN if AN>0 else np.nan

    return tpr, fpr, TP, AP, FP, AN


def evalulate_detection_test(Y_detect_test, Y_detect_pred):
    accuracy = accuracy_score(Y_detect_test, Y_detect_pred, normalize=True, sample_weight=None)
    tpr, fpr, tp, ap, fp, an = get_tpr_fpr(Y_detect_test, Y_detect_pred)
    return accuracy, tpr, fpr, tp, ap, fp, an

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def get_next_class(Y_test):
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)
    Y_test_labels = (Y_test_labels + 1) % num_classes
    return np.eye(num_classes)[Y_test_labels]

def get_least_likely_class(Y_pred):
    num_classes = Y_pred.shape[1]
    Y_target_labels = np.argmin(Y_pred, axis=1)
    return np.eye(num_classes)[Y_target_labels]

class Average_Saliency(object):
    def __init__(self, model, output_index=0):
        pass

    def get_grad(self, input_image):
        pass

    def get_average_grad(self, input_image, stdev_spread=.2, nsamples=50):
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image, dtype = np.float64)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples

# class Single_Saliency(Average_Saliency):

#     def __init__(self, model, output_index = 0):
#         # Define the function to compute the gradient
#         input_tensors = [model.get_input_at(0)]
#         gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
#         self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

#     def get_grad(self, input_image):
#         # Execute the function to compute the gradient
#         x_value = np.expand_dims(input_image, axis=0)
#         gradients = self.compute_gradients([x_value])[0][0]

#         return gradients