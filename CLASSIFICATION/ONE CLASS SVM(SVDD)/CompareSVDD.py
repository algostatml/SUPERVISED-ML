#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:25:09 2019

@author: kenneth
"""
import time
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1111)
from DualSVDD import DualSVDD, DualSVDD_NE, MiniDualSVDD, MiniDualSVDD_NE
np.random.seed(1000)
plt.rcParams.update({'font.size': 8})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 200

def SVDDdataset(n_samples= None, outlier_size = None, n_features = None):
    if not n_samples:
        n_samples = 1000
    else:
        n_samples = n_samples
    if not n_features:
        n_features = 2
    else:
        n_features = n_features
    if not outlier_size:
        outlier_size = 1000//20
    else:
        outlier_size = outlier_size
    insample = n_samples - outlier_size
    inliers = np.random.randn(insample, n_features) * .7
    inliers_y = np.ones(insample)
    outsample = np.random.uniform(low = -7, high = 7, size = (outlier_size, n_features))
    outsample_y = np.zeros(outlier_size)
    X = np.vstack((inliers, outsample))
    y = np.hstack((inliers_y, outsample_y))
    return X, y


#%%
X, y = SVDDdataset()
X = np.hstack((X, y.reshape(-1, 1)))
df = X[X[:, 2] == 1][:, [0, 1]]
dy = X[X[:, 2] == 1][:, 2]
#plt.scatter(df[:, 0], df[:, 1])
#plt.scatter(X[:, 0], X[:, 1])
#plt.scatter(X[:, 0], X[:, 1], c = y, s = 5, cmap = 'coolwarm_r')

#%%
from sklearn.metrics import roc_auc_score
dsvdd = DualSVDD(kernel='linear').fit(df)
#plt.plot(np.arange(100), dsvdd.cost_rec)
dsvdd.predict(X[:, [0, 1]])
dsvdd.summary(y, dsvdd.predict(X[:, [0, 1]]), dsvdd.alpha)
plt.scatter(X[:, 0], X[:, 1], c = dsvdd.predict(X[:, [0, 1]]), cmap = 'coolwarm_r', s = 1)
roc_auc_score(y, dsvdd.predict(X[:, [0, 1]]))
#%% SVDD-GD No Errors
dsvddNE = DualSVDD_NE(kernel='rbf').fit(df)
plt.plot(np.arange(100), dsvddNE.cost_rec)
dsvddNE.predict(X[:, [0, 1]])
plt.scatter(X[:, 0], X[:, 1], c = dsvddNE.predict(X[:, [0, 1]]))


#%% Minibatch with Errors
from sklearn.metrics import roc_auc_score
stochdsvdd = MiniDualSVDD(kernel='locguass').fit(df)
#plt.plot(np.arange(100), stochdsvdd.cost_rec)
stochdsvdd.predict(X[:, [0, 1]])
stochdsvdd.summary(y, stochdsvdd.predict(X[:, [0, 1]]), stochdsvdd.alpha)
plt.scatter(X[:, 0], X[:, 1], c = stochdsvdd.predict(X[:, [0, 1]]), cmap = 'coolwarm_r', s = 1)
roc_auc_score(y, stochdsvdd.predict(X[:, [0, 1]]))
#%% Minibatch version with no errors
stochdsvdd = MiniDualSVDD_NE(kernel='linear').fit(df)
plt.plot(np.arange(100), stochdsvdd.cost_rec)
stochdsvdd.predict(X[:, [0, 1]])
stochdsvdd.summary(y, stochdsvdd.predict(X[:, [0, 1]]), stochdsvdd.alpha)
plt.scatter(X[:, 0], X[:, 1], c = stochdsvdd.predict(X[:, [0, 1]]))
roc_auc_score(y, stochdsvdd.predict(X[:, [0, 1]]))
#%% Decision boundary

def plot_decision_boundary(clf, X, Y, cmap='coolwarm_r'):
    h = 0.25
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha = 0.20)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.01)
    plt.scatter(X[:,0], X[:,1], c =  Y, cmap = cmap, edgecolors='k', label = f'F1: {round(clf.f1(y, clf.predict(X[:, [0, 1]])), 2)}')
    plt.legend()
plot_decision_boundary(dsvdd, X, y, cmap='coolwarm_r')


#%%

kernels  = ['linear', 'rbf', 'sigmoid', 'polynomial',
            'linrbf', 'rbfpoly', 'etakernel', 'laplace']


kernel_outcome = {'linear': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []}, 
                             'rbf': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []}, 
                             'sigmoid': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []}, 
                             'polynomial': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []},
                             'linrbf': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []},
                             'rbfpoly': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []},
                             'etakernel': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []},
                             'laplace': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []}}

for _ in range(3):
    for p, q in train_test.items():
        for ii in kernels:
            if ii == 'linear':
                start = time.time()
                logit = stochasticLogistic(0.1, 10).fit(q['train'][0], q['train'][1])
                end = time.time() - start
                kernel_outcome[ii][f'{p}'] = logit.predict(q['test'][0])
                kernel_outcome[ii]['time'].append(end)
                kernel_outcome[ii]['acc'].append(logit.accuracy(q['test'][1], logit.predict(q['test'][0])))
                kernel_outcome[ii]['prec'].append(logit.precision(q['test'][1], logit.predict(q['test'][0])))
                kernel_outcome[ii]['rec'].append(logit.recall(q['test'][1], logit.predict(q['test'][0])))
                kernel_outcome[ii]['f1'].append(logit.f1(q['test'][1], logit.predict(q['test'][0])))
                kernel_outcome[ii]['impr_f1'].append(logit.fscore(q['test'][1], logit.predict(q['test'][0]), logit.alpha))
            else:
                if p == 'blob':
                    start = time.time()
                    klogit = StochKLR(kernel = ii).fit(q['train'][0], q['train'][1], iterations=50)
                    end = time.time() - start
                    start = time.time()
                    kernel_outcome[ii][f'{p}'] = klogit.predict(q['test'][0])
                    kernel_outcome[ii]['time'].append(end)
                    kernel_outcome[ii]['acc'].append(klogit.accuracy(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['prec'].append(klogit.precision(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['rec'].append(klogit.recall(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['f1'].append(klogit.f1(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['impr_f1'].append(klogit.fscore(q['test'][1], klogit.predict(q['test'][0]), klogit.alpha))
                else:
                    start = time.time()
                    klogit = StochKLR(kernel = ii).fit(q['train'][0], q['train'][1], iterations=10)
                    end = time.time() - start
                    start = time.time()
                    kernel_outcome[ii][f'{p}'] = klogit.predict(q['test'][0])
                    kernel_outcome[ii]['time'].append(end)
                    kernel_outcome[ii]['acc'].append(klogit.accuracy(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['prec'].append(klogit.precision(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['rec'].append(klogit.recall(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['f1'].append(klogit.f1(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['impr_f1'].append(klogit.fscore(q['test'][1], klogit.predict(q['test'][0]), klogit.alpha))
                    
#%%                   
s = .5
color = 'coolwarm_r'
fig, ax = plt.subplots(4, 9, figsize=(12, 4),gridspec_kw=dict(hspace=0, wspace=0),
                       subplot_kw={'xticks':[], 'yticks':[]})

ax[0, 0].scatter(train_test['moon']['train'][0][:, 0], train_test['moon']['train'][0][:, 1], c = train_test['moon']['train'][1], s = 1, cmap = color)
ax[1, 0].scatter(train_test['blob']['train'][0][:, 0], train_test['blob']['train'][0][:, 1], c = train_test['blob']['train'][1], s = 1, cmap = color)
ax[2, 0].scatter(train_test['circle']['train'][0][:, 0], train_test['circle']['train'][0][:, 1], c = train_test['circle']['train'][1], s = 1, cmap = color)
ax[3, 0].scatter(train_test['class']['train'][0][:, 0], train_test['class']['train'][0][:, 1], c = train_test['class']['train'][1], s = 1, cmap = color)

ax[0, 1].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['linear']['moon'], s = 1, cmap = color)
ax[1, 1].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['linear']['blob'], s = 1, cmap = color)
ax[2, 1].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['linear']['circle'], s = 1, cmap = color)
ax[3, 1].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['linear']['class'], s = 1, cmap = color)

ax[0, 2].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['rbf']['moon'], s = 1, cmap = color)
ax[1, 2].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['rbf']['blob'], s = 1, cmap = color)
ax[2, 2].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['rbf']['circle'], s = 1, cmap = color)
ax[3, 2].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['rbf']['class'], s = 1, cmap = color)

ax[0, 3].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['sigmoid']['moon'], s = 1, cmap = color)
ax[1, 3].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['sigmoid']['blob'], s = 1, cmap = color)
ax[2, 3].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['sigmoid']['circle'], s = 1, cmap = color)
ax[3, 3].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['sigmoid']['class'], s = 1, cmap = color)

ax[0, 4].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['polynomial']['moon'], s = 1, cmap = color)
ax[1, 4].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['polynomial']['blob'], s = 1, cmap = color)
ax[2, 4].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['polynomial']['circle'], s = 1, cmap = color)
ax[3, 4].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['polynomial']['class'], s = 1, cmap = color)

ax[0, 5].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['laplace']['moon'], s = 1, cmap = color)
ax[1, 5].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['laplace']['blob'], s = 1, cmap = color)
ax[2, 5].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['laplace']['circle'], s = 1, cmap = color)
ax[3, 5].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['laplace']['class'], s = 1, cmap = color)

ax[0, 6].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['linrbf']['moon'], s = 1, cmap = color)
ax[1, 6].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['linrbf']['blob'], s = 1, cmap = color)
ax[2, 6].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['linrbf']['circle'], s = 1, cmap = color)
ax[3, 6].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['linrbf']['class'], s = 1, cmap = color)

ax[0, 7].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['rbfpoly']['moon'], s = 1, cmap = color)
ax[1, 7].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['rbfpoly']['blob'], s = 1, cmap = color)
ax[2, 7].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['rbfpoly']['circle'], s = 1, cmap = color)
ax[3, 7].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['rbfpoly']['class'], s = 1, cmap = color)

ax[0, 8].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['etakernel']['moon'], s = 1, cmap = color)
ax[1, 8].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['etakernel']['blob'], s = 1, cmap = color)
ax[2, 8].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['etakernel']['circle'], s = 1, cmap = color)
ax[3, 8].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['etakernel']['class'], s = 1, cmap = color)

ax[0, 0].set_title('original')
ax[0, 1].set_title('linear')
ax[0, 2].set_title('rbf')
ax[0, 3].set_title('poly')
ax[0, 4].set_title('sigmoid')
ax[0, 5].set_title('laplace')
ax[0, 6].set_title('rbfpoly')
ax[0, 7].set_title('linrbf')
ax[0, 8].set_title('etakernel')
ax[0, 0].set_ylabel('Moons')
ax[1, 0].set_ylabel('Blob')
ax[2, 0].set_ylabel('Circle')
ax[3, 0].set_ylabel('Classifciation')
fig.set_tight_layout(True)