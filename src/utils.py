import pickle 
import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd 
import seaborn as sn
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

## Fix random seeds
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

## Calculate Precision and Recall
def calculatePrecisionRecall(output,target):
    length = len(output)
    classes = [0,3,8]
    for c in classes:
        tp_fn, tp, tp_fp = 0, 0, 0
        for idx in range(length):
            if target[idx] == c:
                tp_fn += 1 
                if output[idx] == c:
                    tp += 1 
            if output[idx] == c:
                tp_fp += 1 
        if tp_fn != 0:
            recall = tp/tp_fn 
        else:
            print('Class {}: TP+FN==0, Calculate recall error'.format(c))
        if tp_fp != 0:
            precision = tp/tp_fp
        else:
            print('Class {}: TP+FP==0, Calculate precision error'.format(c))
        print('Class {}: Recall={}, Precision={}'.format(c,recall,precision))
    

## Plot confusion matrix
def plotConfusionMatrix(output,target):
    label = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    outlen = len(output)
    matrix = np.zeros((10,10),dtype=np.int)
    for idx in range(outlen):
        matrix[target[idx],output[idx]] += 1
    df_cm = pd.DataFrame(matrix, index = label,
                  columns = label)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)  
    plt.savefig('../image/VGG16ConfusionMatrix.jpg')
    plt.clf()
    return

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

## Plot ROC and calculate AUC score
def plotAUCCurve(prob, label):
    for c in range(10):
        image_path = '../image/ROCclass{}.jpg'.format(c)
        prob_class = prob[:,c]
        label_class = label == c
        label_class = [1 if x == True else 0 for x in label_class]
        auc = roc_auc_score(label_class,prob_class)
        print('Class %d: ROC AUC=%.3f' % (c,auc))
        fpr, tpr, _ = roc_curve(label_class, prob_class)
        plt.plot(fpr, tpr, marker='.', label='VGG16, AUC={}'.format(auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(image_path)
        plt.clf()

## Load CIFAR dataset
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(image_dir, val_ratio=0.02):
    cifar10_dir = image_dir
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    X_dev = X_train[int(len(X_train)*(1-val_ratio)):]
    y_dev = y_train[int(len(y_train)*(1-val_ratio)):]
    X_train = X_train[:int(len(X_train)*(1-val_ratio))]
    y_train = y_train[:int(len(y_train)*(1-val_ratio))]

    return X_train.reshape(-1,3,32,32).transpose(0,2,3,1), y_train, X_dev.reshape(-1,3,32,32).transpose(0,2,3,1), y_dev, X_test.reshape(-1,3,32,32).transpose(0,2,3,1), y_test