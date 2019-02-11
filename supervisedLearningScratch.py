#### Attribution of source code used:
# Learning Curve: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
# ROC graph: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
# Confusion matrix: https://gist.github.com/zachguo/10296432
# Probability for LinearSVC: https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from sklearn import preprocessing

import math

import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import tree
import librosa
import librosa.display
import graphviz
import pickle
import utils
import os.path

from pprint import pprint
from time import time
import logging

from sklearn.externals.six.moves import zip
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import print_cm
import confusionMatrix

from sklearn.metrics import confusion_matrix

import scikitplot as skplt
import matplotlib.pyplot as plt

from numpy import genfromtxt
import random

import time

# Toggle dataset
fma = False
bna = True

# Toggle Models of choice
dTree = False
dTree_paramSearch = False
dTree_learningCurve = False
dTree_testAssess = False

neuralNet = False
neuralNet_paramSearch = False
neuralNet_learningCurve = False
neuralNet_testAssess = False

bostedTree = False
boostedTree_paramSearch = False
boostedTree_learningCurve = False
boostedTree_testAssess = False

supVect = True
supVect_paramSearch = False
supVect_learningCurve = False
supVect_testAssess = True

KNN = False
KNN_paramSearch = False
KNN_learningCurve = False
KNN_testAssess = False

# Model parameters
# Decision Tree
# max_depth = 3

# Learning Curve: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def convertToStr_bna(array):

    # convert num labels to strings
    remappedArray = []
    for label in array:
        if label == 0:
            remappedArray.append('Genuine')
        elif label == 1:
            remappedArray.append('Forged')
    return remappedArray


### DATA PROCESSING ############################################################

if fma:
    # Paths and files
    audio_dir = '../../data/fma_metadata/'
    localDataFile = 'trainTestData.pkl'

    if os.path.exists(localDataFile):
        with open(localDataFile, 'rb') as f:
            data = pickle.load(f)
        y_train = data[0]; y_val = data[1]; y_test = data[2]
        X_train = data[3]; X_val = data[4]; X_test = data[5]
    else:
        # Load metadata and features
        tracks = utils.load(audio_dir + 'tracks.csv')
        genres = utils.load(audio_dir + 'genres.csv')
        features = utils.load(audio_dir + 'features.csv')
        echonest = utils.load(audio_dir + 'echonest.csv')

        np.testing.assert_array_equal(features.index, tracks.index)
        assert echonest.index.isin(tracks.index).all()

        # Setup train/test split
        small = tracks['set', 'subset'] <= 'small'

        train = tracks['set', 'split'] == 'training'
        val = tracks['set', 'split'] == 'validation'
        test = tracks['set', 'split'] == 'test'

        y_train = tracks.loc[small & train, ('track', 'genre_top')]
        y_val = tracks.loc[small & val, ('track', 'genre_top')]
        y_test = tracks.loc[small & test, ('track', 'genre_top')]
        # X_train = features.loc[small & train, 'mfcc'] #just mfcc features
        # X_test = features.loc[small & test, 'mfcc']
        X_train = features.loc[small & train] #all audio-extracted features
        X_val = features.loc[small & val]
        X_test = features.loc[small & test]

        print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
        print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))

        # Be sure training samples are shuffled.
        X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance.
        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        scaler.transform(X_val)

        # Save the formatted data:
        with open(localDataFile, 'wb') as f:
            # pickle.dump([y_train, y_test, X_train, X_test], f)
            pickle.dump([y_train, y_val, y_test, X_train, X_val, X_test], f)

if bna:
    # Paths and files
    bna_dir = '../../data/banknoteAuthentication/'
    localDataFile = 'trainTestData_bna.pkl'

    # print(bna_dir)

    # load data
    # bnaData = np.loadtxt(bna_dir + 'banknoteAuthentication.txt')
    bnaData = genfromtxt(bna_dir + 'banknoteAuthentication.txt', delimiter=',')
    # print('eh?')

    # get num from each class
    # numZero = 0
    # numOne = 0
    # for i in range(bnaData.shape[0]):
    #     if bnaData[i,-1] == 0:
    #         numZero += 1
    #     elif bnaData[i,-1] == 1:
    #         numOne += 1
    # print(numZero)
    # print(numOne)

    # print(bnaData[761,:])

    # pos negative split
    negExamp = bnaData[:762,:]
    posExamp = bnaData[762:,:]

    # balance data
    negExamp = negExamp[:610,:]

    #shuffle examples
    np.random.shuffle(negExamp)
    np.random.shuffle(posExamp)

    # print('here:')
    # print(negExamp)

    # formulate train & test sets
    # X_train = np.vstack((negExamp[:549,:-1], posExamp[:549,:-1]))
    # y_train = np.hstack((negExamp[:549,-1], posExamp[:549,-1]))

    # X_test = np.vstack((negExamp[549:,:-1], posExamp[549:,:-1]))
    # y_test = np.hstack((negExamp[549:,-1], posExamp[549:,-1]))

    X_train = np.vstack((negExamp[:488,:-1], posExamp[:488,:-1]))
    y_train = np.hstack((negExamp[:488,-1], posExamp[:488,-1]))

    X_val = np.vstack((negExamp[488:549,:-1], posExamp[488:549,:-1]))
    y_val = np.hstack((negExamp[488:549,-1], posExamp[488:549,-1]))

    X_test = np.vstack((negExamp[549:,:-1], posExamp[549:,:-1]))
    y_test = np.hstack((negExamp[549:,-1], posExamp[549:,-1]))

    # Be sure training samples are shuffled.
    X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance.
    scaler = skl.preprocessing.StandardScaler(copy=False)
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    # print(X_train)
    # print(X_train.shape)


### TRAIN A DECISION TREE ######################################################

# Decision tree classification
if dTree:

    # paramSearch = False

    if dTree_paramSearch:
        # used gridsearch code from here: https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py
        pipeline = Pipeline([
            ('clf', tree.DecisionTreeClassifier()),
        ])

        # mdVals = [20, 40]
        # mslVals = [10, 20]
        # mssVals = [2, 3]

        if fma:
            #iteration 1
            mdVals = [20, 40, 80, 160]
            mslVals = [10, 20, 40, 80]
            mssVals = [2, 3, 4, 5]

            #iteration 2
            mdVals = [10, 15, 20, 25]
            mslVals = [25, 35, 45, 55]
            mssVals = [2, 3, 4, 5]

            #iteration 3
            mdVals = [17, 19, 21, 23]
            mslVals = [5, 15, 25, 35]
            mssVals = [2, 3, 4, 5]

            #iteration 4
            mdVals = [14, 15, 16, 17]
            mslVals = [22, 24, 26, 28]
            mssVals = [2, 3, 4, 5]

            #iteration 5
            mdVals = [12, 13, 14, 15]
            mslVals = [27, 28, 29, 30]
            mssVals = [2, 3, 4, 5]

            #iteration 6
            mdVals = [9, 10, 11, 12]
            mslVals = [30, 31, 32, 33]
            mssVals = [2, 3, 4, 5]

            #iteration 7
            mdVals = [6, 7, 8, 9]
            mslVals = [30, 31, 32, 33]
            mssVals = [2, 3, 4, 5]

            #iteration 8
            mdVals = [8, 9, 10, 11]
            mslVals = [30, 31, 32, 33]
            mssVals = [2, 3, 4, 5]

        elif bna:
            #iteration 1
            mdVals = [20, 40, 80, 160]
            mslVals = [10, 20, 40, 80]
            mssVals = [2, 3, 4, 5]

            #iteration 2
            mdVals = [5, 10, 15, 20]
            mslVals = [4, 6, 8, 10]
            mssVals = [2, 3, 4, 5]

            #iteration 3
            mdVals = [7, 8, 9, 10]
            mslVals = [1, 2, 3, 4]
            mssVals = [4, 5, 6, 7]

            #iteration 4
            mdVals = [5, 6, 7, 8]
            mslVals = [2, 3, 4, 5]
            mssVals = [4, 5, 6, 7]

            #iteration 5
            mdVals = [5, 6, 7, 8]
            mslVals = [1, 2, 3, 4]
            mssVals = [3, 4, 5, 6]

            #iteration 6
            mdVals = [5, 6, 7, 8, 9]
            mslVals = [1, 2, 3, 4, 5]
            mssVals = [2, 3, 4, 5, 6]

        parameters = {
            # 'clf__max_depth': (20, 40, 80, 160),
            # 'clf__min_samples_split': (2, 3, 4, 5),
            # 'clf__min_samples_leaf': (10, 20, 40, 80),
            'clf__max_depth': mdVals,
            'clf__min_samples_leaf': mslVals,
            'clf__min_samples_split': mssVals,
        }

        grid_search = GridSearchCV(pipeline, parameters, cv=3,
                                   n_jobs=-1, verbose=1)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(parameters)
        # t0 = time()
        grid_search.fit(X_train, y_train)
        # print("done in %0.3fs" % (time() - t0))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        # visualize grid search results
        # scores = grid_search.cv_results_['mean_test_score']
        # print(scores)
        scores = grid_search.cv_results_['mean_test_score'] \
                            .reshape((len(parameters['clf__max_depth']), \
                                      len(parameters['clf__min_samples_leaf']), \
                                      len(parameters['clf__min_samples_split'])), \
                                      order='C')
        scores = np.asarray(scores)
        # print(scores)
        md_msl = np.squeeze(np.mean(scores, axis=2))
        md_mss = np.squeeze(np.mean(scores, axis=1))
        msl_mss = np.squeeze(np.mean(scores, axis=0))
        # print(md_mss)

        #colorbar params
        maxVal = np.amax(scores)
        minVal = np.amin(scores)

        # plotting
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(md_msl, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax1.set_ylabel('max depth')
        ax1.set_yticks(np.arange(len(mdVals)))
        ax1.set_yticklabels(mdVals)
        ax1.set_xlabel('min samples leaf')
        ax1.set_xticks(np.arange(len(mslVals)))
        ax1.set_xticklabels(mslVals)

        ax2.imshow(md_mss, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax2.set_ylabel('max depth')
        ax2.set_yticks(np.arange(len(mdVals)))
        ax2.set_yticklabels(mdVals)
        ax2.set_xlabel('min samples split')
        ax2.set_xticks(np.arange(len(mssVals)))
        ax2.set_xticklabels(mssVals)

        im = ax3.imshow(msl_mss, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax3.set_ylabel('min samples leaf')
        ax3.set_yticks(np.arange(len(mslVals)))
        ax3.set_yticklabels(mslVals)
        ax3.set_xlabel('min samples split')
        ax3.set_xticks(np.arange(len(mssVals)))
        ax3.set_xticklabels(mssVals)

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)

        f.suptitle('Decision Tree Hyperparameter Grid Search')

        plt.show()

    if dTree_learningCurve:

        if fma:
            maxDepth = 9
            minSampLeaf = 31
            minSampSplit = 4

            nSplits = 10
        elif bna:
            maxDepth = 7
            minSampLeaf = 1
            minSampSplit = 2

            nSplits = 100

        cv = ShuffleSplit(n_splits=nSplits, test_size=0.2, random_state=0)
        plot_learning_curve(tree.DecisionTreeClassifier(max_depth=maxDepth, \
                                                        min_samples_leaf=minSampLeaf, \
                                                        min_samples_split=minSampSplit), \
                            'Decision Tree Learning Curve \n (max_depth = ' + str(maxDepth) + ', ' +
                            'min_samples_leaf = ' + str(minSampLeaf) + ', min_samples_split = ' + str(minSampSplit) + ')', \
                            X_train, y_train, cv=cv, n_jobs=-1)

        plt.show()

    if dTree_testAssess:

        if fma:
            maxDepth = 9
            minSampLeaf = 31
            minSampSplit = 4

        elif bna:
            maxDepth = 7
            minSampLeaf = 1
            minSampSplit = 2

        #Model Accuracy
        clf = tree.DecisionTreeClassifier(max_depth=maxDepth, min_samples_leaf=minSampLeaf, \
                                          min_samples_split=minSampSplit)
        trainStart = time.time()
        clf = clf.fit(X_train, y_train)
        trainEnd = time.time()

        testStart = time.time()
        score = clf.score(X_test, y_test)
        testEnd = time.time()

        print('Decision Tree accuracy: {:.2%}'.format(score))
        print('Train time: {:.5f}'.format(trainEnd - trainStart))
        print('Test time: {:.5f}'.format(testEnd - testStart))

        if fma:
            uniqueLabels = y_test.unique().tolist()
        elif bna:
            uniqueLabels = ['Genuine', 'Forged']

        #Confusion Mat
        predictY = clf.predict(X_test)

        if fma:
            cm = confusion_matrix(y_test, predictY, labels=uniqueLabels)
        if bna:
            # predictY = convertToStr_bna(predictY)
            cm = confusion_matrix(convertToStr_bna(y_test), convertToStr_bna(predictY), labels=uniqueLabels)

        plt.figure()
        confusionMatrix.plot_confusion_matrix(cm, uniqueLabels, title='Decision Tree Confusion Matrix')
        # plt.show()

        #ROC
        probs = clf.predict_proba(X_test)
        if fma:
            skplt.metrics.plot_roc_curve(y_test, probs, title='Decision Tree ROC Curves')
        if bna:
            skplt.metrics.plot_roc_curve(convertToStr_bna(y_test), probs[::-1], title='Decision Tree ROC Curves')
        plt.show()

        # save decision tree model to PDF
        # dot_data = tree.export_graphviz(clf, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render("genres")

### TRAIN A NEURAL NETWORK #####################################################

if neuralNet:

    # paramSearch = False
    # Training settings
    batch_size = 64

    if fma:
        # create encoder to map from string to int labels
        le = preprocessing.LabelEncoder()
        le.fit(y_train.iloc[:].values)

        torchTrainX = torch.tensor(X_train.iloc[:,:].values)
        torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

        torchTestX = torch.tensor(X_test.iloc[:,:].values)
        torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

        torchValX = torch.tensor(X_val.iloc[:,:].values)
        torchValY = torch.tensor(le.transform(y_val.iloc[:].values))

    elif bna:

        torchTrainX = torch.tensor(X_train)
        torchTrainY = torch.tensor(y_train, dtype=torch.long)

        torchValX = torch.tensor(X_val)
        torchValY = torch.tensor(y_val, dtype=torch.long)

        torchTestX = torch.tensor(X_test)
        torchTestY = torch.tensor(y_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(torchTrainX, torchTrainY)
    test_dataset = torch.utils.data.TensorDataset(torchTestX, torchTestY)
    val_dataset = torch.utils.data.TensorDataset(torchValX, torchValY)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    if neuralNet_paramSearch:

        if fma:
            numClasses = 8
            numFeats = 518

            # search parameter values
            kern1Sizes = [2, 3, 4, 5]
            kern2Sizes = [2, 3, 4, 5]
            kern3Sizes = [2, 3, 4]

        elif bna:
            numClasses = 2
            numFeats = 4

            # search parameter values
            kern1Sizes = [2]
            kern2Sizes = [2]
            kern3Sizes = [1]

        # data structure to store results
        # netAcc = np.zeros((4,4,3))
        netAcc = np.zeros((len(kern1Sizes), len(kern2Sizes), len(kern3Sizes)))

        numValCombos = len(kern1Sizes) * len(kern2Sizes) * len(kern3Sizes)
        iterCount = 0

        for i in range(len(kern1Sizes)):
            for j in range(len(kern2Sizes)):
                for k in range(len(kern3Sizes)):

                    iterCount += 1

                    kern1 = kern1Sizes[i]
                    kern2 = kern2Sizes[j]
                    kern3 = kern3Sizes[k]

                    #calc the input size to fc layer (518 features per example)
                    fcDim = math.floor(numFeats/kern1)
                    fcDim = math.floor(fcDim/kern3)
                    fcDim = math.floor(fcDim/kern2)
                    fcDim = math.floor(fcDim/kern3)
                    fcDim = fcDim * 20 #num output channels from conv2 layer

                    print('Testing value combo ' + str(iterCount) + ' of ' + str(numValCombos))

                    #need to calculate the shape of the input to fc layer

                    class Net(nn.Module):

                        def __init__(self):
                            super(Net, self).__init__()

                            self.conv1 = nn.Conv1d(1, 10, kernel_size=kern1, stride=kern1)
                            self.conv2 = nn.Conv1d(10, 20, kernel_size=kern2, stride=kern2)
                            self.mp = nn.MaxPool1d(kernel_size=kern3, stride=kern3)
                            self.fc = nn.Linear(fcDim, numClasses)
                            # self.fc = nn.Linear(2520, 8)
                            # self.do = nn.Dropout(p=0.5)

                        def forward(self, x):
                            in_size = x.size(0)

                            x = F.relu(self.mp(self.conv1(x)))
                            x = F.relu(self.mp(self.conv2(x)))
                            # x = self.do(x)

                            # flatten tensor
                            x = x.view(in_size, -1)

                            # fully-connected layer
                            x = self.fc(x)
                            return F.log_softmax(x, dim=1)


                    model = Net()

                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

                    def train(epoch):
                        model.train()
                        for batch_idx, (data, target) in enumerate(train_loader):
                            # print(data.size())
                            # print(target.size())
                            data, target = Variable(data), Variable(target)
                            data = data.unsqueeze(1) #testing insertion of dimension
                            data = data.float()
                            optimizer.zero_grad()
                            output = model(data)
                            loss = F.nll_loss(output, target)
                            loss.backward()
                            optimizer.step()
                            if batch_idx % 10 == 0:
                                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, batch_idx * len(data), len(train_loader.dataset),
                                    100. * batch_idx / len(train_loader), loss.item())) #loss.data[0]


                    def test():
                        model.eval()
                        test_loss = 0
                        correct = 0
                        for data, target in val_loader:
                            # data, target = Variable(data, volatile=True), Variable(target)
                            data, target = Variable(data), Variable(target)
                            data = data.unsqueeze(1) #testing insertion of dimension
                            data = data.float()
                            output = model(data)
                            # sum up batch loss
                            # test_loss += F.nll_loss(output, target, size_average=False).data[0]
                            # test_loss += F.nll_loss(output, target, size_average=False).item()
                            test_loss += F.nll_loss(output, target, reduction='sum').item()
                            # get the index of the max log-probability
                            pred = output.data.max(1, keepdim=True)[1]
                            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                        test_loss /= len(val_loader.dataset)
                        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                            test_loss, correct, len(val_loader.dataset),
                            100. * correct / len(val_loader.dataset)))
                        return 100. * correct / len(val_loader.dataset)

                    for epoch in range(1, 10):
                        train(epoch)
                        testAcc = test()

                    netAcc[i,j,k] = testAcc

        # Save the formatted data:
        with open('neuralNetAccData.pkl', 'wb') as f:
            pickle.dump([netAcc], f)

        with open('neuralNetAccData.pkl', 'rb') as f:
            netAcc = pickle.load(f)
        netAcc = netAcc[0]

        # search parameter values
        # kern1Sizes = [2, 3, 4, 5]
        # kern2Sizes = [2, 3, 4, 5]
        # kern3Sizes = [2, 3, 4]

        #max parameter values
        ind = np.unravel_index(np.argmax(netAcc, axis=None), netAcc.shape)
        print('indices of best params:')
        print(ind)

        if fma:
            f1_f2 = np.squeeze(np.mean(netAcc, axis=2))
            f1_f3 = np.squeeze(np.mean(netAcc, axis=1))
            f2_f3 = np.squeeze(np.mean(netAcc, axis=0))

            #colorbar params
            maxVal = np.amax(netAcc)
            minVal = np.amin(netAcc)

        if bna:
            f1_f2 = np.squeeze(np.mean(netAcc, axis=0))
            f1_f3 = np.squeeze(np.mean(netAcc, axis=0))
            f2_f3 = np.squeeze(np.mean(netAcc, axis=0))

            f1_f2 = np.expand_dims(f1_f2, axis=0)
            f1_f2 = np.expand_dims(f1_f2, axis=1)

            f1_f3 = np.expand_dims(f1_f3, axis=0)
            f1_f3 = np.expand_dims(f1_f3, axis=1)

            f2_f3 = np.expand_dims(f2_f3, axis=0)
            f2_f3 = np.expand_dims(f2_f3, axis=1)

            #colorbar params
            maxVal = 100
            minVal = 0

        # plotting
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(f1_f2, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax1.set_ylabel('conv1 kernel size')
        ax1.set_yticks(np.arange(len(kern1Sizes)))
        ax1.set_yticklabels(kern1Sizes)
        ax1.set_xlabel('conv2 kernel size')
        ax1.set_xticks(np.arange(len(kern2Sizes)))
        ax1.set_xticklabels(kern2Sizes)

        ax2.imshow(f1_f3, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax2.set_ylabel('conv1 kernel size')
        ax2.set_yticks(np.arange(len(kern1Sizes)))
        ax2.set_yticklabels(kern1Sizes)
        ax2.set_xlabel('pool kernel size')
        ax2.set_xticks(np.arange(len(kern3Sizes)))
        ax2.set_xticklabels(kern3Sizes)

        im = ax3.imshow(f2_f3, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax3.set_ylabel('conv2 kernel size')
        ax3.set_yticks(np.arange(len(kern2Sizes)))
        ax3.set_yticklabels(kern2Sizes)
        ax3.set_xlabel('pool kernel size')
        ax3.set_xticks(np.arange(len(kern3Sizes)))
        ax3.set_xticklabels(kern3Sizes)

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)

        f.suptitle('Neural Net Hyperparameter Grid Search')

        plt.show()

    if neuralNet_learningCurve:

        if fma:
            kern1 = 2
            kern2 = 5
            kern3 = 2

            numClasses = 8
            numFeats = 518
        elif bna:
            kern1 = 2
            kern2 = 2
            kern3 = 1

            numClasses = 2
            numFeats = 4

        #calc the input size to fc layer (518 features per example)
        fcDim = math.floor(numFeats/kern1)
        fcDim = math.floor(fcDim/kern3)
        fcDim = math.floor(fcDim/kern2)
        fcDim = math.floor(fcDim/kern3)
        fcDim = fcDim * 20 #num output channels from conv2 layer

        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()

                self.conv1 = nn.Conv1d(1, 10, kernel_size=kern1, stride=kern1)
                self.conv2 = nn.Conv1d(10, 20, kernel_size=kern2, stride=kern2)
                self.mp = nn.MaxPool1d(kernel_size=kern3, stride=kern3)
                self.fc = nn.Linear(fcDim, numClasses)
                # self.fc = nn.Linear(2520, 8)
                # self.do = nn.Dropout(p=0.5)

            def forward(self, x):
                in_size = x.size(0)

                x = F.relu(self.mp(self.conv1(x)))
                x = F.relu(self.mp(self.conv2(x)))
                # x = self.do(x)

                # flatten tensor
                x = x.view(in_size, -1)

                # fully-connected layer
                x = self.fc(x)
                return F.log_softmax(x, dim=1)


        model = Net()

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        def train(epoch):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # print(data.size())
                # print(target.size())
                data, target = Variable(data), Variable(target)
                data = data.unsqueeze(1) #testing insertion of dimension
                data = data.float()
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item())) #loss.data[0]


        def test():
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in val_loader:
                # data, target = Variable(data, volatile=True), Variable(target)
                data, target = Variable(data), Variable(target)
                data = data.unsqueeze(1) #testing insertion of dimension
                data = data.float()
                output = model(data)
                # sum up batch loss
                # test_loss += F.nll_loss(output, target, size_average=False).data[0]
                # test_loss += F.nll_loss(output, target, size_average=False).item()
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(val_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(val_loader.dataset),
                100. * correct / len(val_loader.dataset)))
            testAcc = 100. * correct / len(val_loader.dataset)

            test_loss = 0
            correct = 0
            for data, target in train_loader:
                # data, target = Variable(data, volatile=True), Variable(target)
                data, target = Variable(data), Variable(target)
                data = data.unsqueeze(1) #testing insertion of dimension
                data = data.float()
                output = model(data)
                # sum up batch loss
                # test_loss += F.nll_loss(output, target, size_average=False).data[0]
                # test_loss += F.nll_loss(output, target, size_average=False).item()
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            trainAcc = 100. * correct / len(train_loader.dataset)

            return testAcc, trainAcc

        numEpochs = 10
        testTrainAccs = np.zeros((2,numEpochs))
        for epoch in range(1, numEpochs):
            train(epoch)
            testAcc, trainAcc = test()
            testTrainAccs[0, epoch] = testAcc
            testTrainAccs[1, epoch] = trainAcc

        plt.plot(list(range(1,numEpochs+1)), np.squeeze(testTrainAccs[0,:]))
        plt.plot(list(range(1,numEpochs+1)), np.squeeze(testTrainAccs[1,:]))
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend(['Test Accuracy', 'Train Accuracy'])
        plt.title('Neural Net Learning Curve\n' + \
            '(Conv Layer 1 kernel size = ' + str(kern1) + ', Conv Layer 2 kernel size = ' + str(kern2) + ', ' + \
            'Pooling Layer kernel size = ' + str(kern3) + ')')
        plt.show()

    if neuralNet_testAssess:

        if fma:
            kern1 = 2
            kern2 = 5
            kern3 = 2

            numClasses = 8
            numFeats = 518
        elif bna:
            kern1 = 2
            kern2 = 2
            kern3 = 1

            numClasses = 2
            numFeats = 4

        #calc the input size to fc layer (518 features per example)
        fcDim = math.floor(numFeats/kern1)
        fcDim = math.floor(fcDim/kern3)
        fcDim = math.floor(fcDim/kern2)
        fcDim = math.floor(fcDim/kern3)
        fcDim = fcDim * 20 #num output channels from conv2 layer

        #need to calculate the shape of the input to fc layer

        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()

                self.conv1 = nn.Conv1d(1, 10, kernel_size=kern1, stride=kern1)
                self.conv2 = nn.Conv1d(10, 20, kernel_size=kern2, stride=kern2)
                self.mp = nn.MaxPool1d(kernel_size=kern3, stride=kern3)
                self.fc = nn.Linear(fcDim, numClasses)
                # self.fc = nn.Linear(2520, 8)
                # self.do = nn.Dropout(p=0.5)

            def forward(self, x):
                in_size = x.size(0)

                x = F.relu(self.mp(self.conv1(x)))
                x = F.relu(self.mp(self.conv2(x)))
                # x = self.do(x)

                # flatten tensor
                x = x.view(in_size, -1)

                # fully-connected layer
                x = self.fc(x)
                return F.log_softmax(x, dim=1)


        model = Net()

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        def train(epoch):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # print(data.size())
                # print(target.size())
                data, target = Variable(data), Variable(target)
                data = data.unsqueeze(1) #testing insertion of dimension
                data = data.float()
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                # if batch_idx % 10 == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(data), len(train_loader.dataset),
                #         100. * batch_idx / len(train_loader), loss.item())) #loss.data[0]


        def test():
            model.eval()
            test_loss = 0
            correct = 0
            preds = np.empty((0,1))
            # probs = np.empty((0,8))
            probs = torch.zeros(0, 8)
            for data, target in test_loader:
                # data, target = Variable(data, volatile=True), Variable(target)
                data, target = Variable(data), Variable(target)
                data = data.unsqueeze(1) #testing insertion of dimension
                data = data.float()
                output = model(data)
                probs = torch.cat((probs, output), 0)
                # temp = torch.nn.Softmax(output)
                # print(output)
                # quit()
                # print(output)
                # quit()
                # sum up batch loss
                # test_loss += F.nll_loss(output, target, size_average=False).data[0]
                # test_loss += F.nll_loss(output, target, size_average=False).item()
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                #append preds
                preds = np.append(preds, pred.numpy(), axis=0)
                # preds.
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
            # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #     test_loss, correct, len(test_loader.dataset),
            #     100. * correct / len(test_loader.dataset)))
            acc = 100. * correct / len(test_loader.dataset)

            return acc, preds, probs

        trainStart = time.time()
        for epoch in range(1, 10):
            train(epoch)
        trainEnd = time.time()

        testStart = time.time()
        testAcc, preds, probs = test()
        testEnd = time.time()


        # clf = tree.DecisionTreeClassifier(max_depth=9, min_samples_leaf=31, \
        #                                   min_samples_split=4)
        # trainStart = time.time()
        # clf = clf.fit(X_train, y_train)
        # trainEnd = time.time()

        # testStart = time.time()
        # score = clf.score(X_test, y_test)
        # testEnd = time.time()

        print('Neural Net accuracy: {:.2%}'.format(testAcc))
        print('Train time: {:.5f}'.format(trainEnd - trainStart))
        print('Test time: {:.5f}'.format(testEnd - testStart))

        if fma:
            uniqueLabels = y_test.unique().tolist()
        elif bna:
            uniqueLabels = ['Genuine', 'Forged']

        if fma:
            cm = confusion_matrix(y_test, le.inverse_transform(preds.astype(int)), labels=uniqueLabels)
        if bna:
            # predictY = convertToStr_bna(predictY)
            cm = confusion_matrix(convertToStr_bna(y_test), convertToStr_bna(preds), labels=uniqueLabels)

        #Confusion Mat
        plt.figure()
        confusionMatrix.plot_confusion_matrix(cm, uniqueLabels, title='Neural Net Confusion Matrix')
        # plt.show()

        #ROC
        sm = torch.nn.Softmax()
        probabilities = sm(probs)
        probabilities = probabilities.detach().numpy()
        if fma:
            skplt.metrics.plot_roc_curve(y_test, probabilities, title='Neural Net ROC Curves')
        if bna:
            skplt.metrics.plot_roc_curve(convertToStr_bna(y_test), probabilities[::-1], title='Neural Net ROC Curves')
        plt.show()

### TRAIN A Boosted DECISION TREE ##############################################
# code from: https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-multiclass-py

if bostedTree:

    # paramSearch = False

    if boostedTree_paramSearch:
        # used gridsearch code from here: https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py

        if fma:
            numEstimators = [10, 100, 1000]
            maxDepths = [1, 2, 4]
            lRates = [0.5, 1, 2]

        elif bna:
            numEstimators = [10, 100, 1000, 10000]
            maxDepths = [1, 2, 4]
            lRates = [0.5, 1, 2]


        param_grid = {
              # "base_estimator__criterion" : ["gini", "entropy"],
              # "base_estimator__splitter" :   ["best", "random"],
              # "n_estimators": [1, 2]
              'n_estimators': numEstimators,
              'learning_rate': lRates,
              'base_estimator__max_depth': maxDepths,
             }


        DTC = DecisionTreeClassifier(random_state = None, max_features = None, \
                                     class_weight = None)

        ABC = AdaBoostClassifier(base_estimator = DTC)

        # print(ABC)

        # run grid search
        grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, n_jobs=-1, \
                                       verbose=10)
        grid_search_ABC.fit(X_train, y_train)
        best_parameters = grid_search_ABC.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        # visualize grid search results
        # scores = grid_search.cv_results_['mean_test_score']
        # print(scores)
        scores = grid_search_ABC.cv_results_['mean_test_score'] \
                            .reshape((len(param_grid['base_estimator__max_depth']), \
                                      len(param_grid['learning_rate']), \
                                      len(param_grid['n_estimators'])), \
                                      order='C')

        # Save the formatted data:
        with open('boostedTreesParamSearch.pkl', 'wb') as f:
            pickle.dump([scores], f)

        scores = np.asarray(scores)
        # print(scores)
        md_lr = np.squeeze(np.mean(scores, axis=2))
        md_ne = np.squeeze(np.mean(scores, axis=1))
        lr_ne = np.squeeze(np.mean(scores, axis=0))
        # print(md_mss)

        #colorbar params
        maxVal = np.amax(scores)
        minVal = np.amin(scores)

        # plotting
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(md_lr, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax1.set_ylabel('max depth')
        ax1.set_yticks(np.arange(len(maxDepths)))
        ax1.set_yticklabels(maxDepths)
        ax1.set_xlabel('learning rate')
        ax1.set_xticks(np.arange(len(lRates)))
        ax1.set_xticklabels(lRates)

        ax2.imshow(md_ne, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax2.set_ylabel('max depth')
        ax2.set_yticks(np.arange(len(maxDepths)))
        ax2.set_yticklabels(maxDepths)
        ax2.set_xlabel('num estimators')
        ax2.set_xticks(np.arange(len(numEstimators)))
        ax2.set_xticklabels(numEstimators)

        im = ax3.imshow(lr_ne, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax3.set_ylabel('learning rate')
        ax3.set_yticks(np.arange(len(lRates)))
        ax3.set_yticklabels(lRates)
        ax3.set_xlabel('num estimators')
        ax3.set_xticks(np.arange(len(numEstimators)))
        ax3.set_xticklabels(numEstimators)

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)

        f.suptitle('AdaBoost Decision Tree Grid Search')

        plt.show()

    if boostedTree_learningCurve:

        if fma:
            maxDepth = 4
            n_estimators = 1000
            lRate = 0.5

        elif bna:
            maxDepth = 2
            n_estimators = 1000
            lRate = 0.5


        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        plot_learning_curve(AdaBoostClassifier(DecisionTreeClassifier(max_depth=maxDepth), \
                                               n_estimators=n_estimators,  \
                                               learning_rate=lRate), \
                            'Adaboost Decision Tree Learning Curve\n' + \
                            '(max_depth = ' + str(maxDepth) + ', n_estimators = ' + str(n_estimators) + ', ' + \
                            'learning_rate = ' + str(lRate) + ')', \
                            X_train, y_train, cv=cv, n_jobs=-1)

        plt.show()

    if boostedTree_testAssess:

        if fma:
            maxDepth = 4
            n_estimators = 1000
            lRate = 0.5

        elif bna:
            maxDepth = 2
            n_estimators = 1000
            lRate = 0.5

        #Model Accuracy
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=maxDepth), \
                                 n_estimators=n_estimators, \
                                 learning_rate=lRate)
        trainStart = time.time()
        clf = clf.fit(X_train, y_train)
        trainEnd = time.time()

        testStart = time.time()
        score = clf.score(X_test, y_test)
        testEnd = time.time()

        print('Boosted Decision Tree accuracy: {:.2%}'.format(score))
        print('Train time: {:.5f}'.format(trainEnd - trainStart))
        print('Test time: {:.5f}'.format(testEnd - testStart))

        if fma:
            uniqueLabels = y_test.unique().tolist()
        elif bna:
            uniqueLabels = ['Genuine', 'Forged']

        #Confusion Mat
        predictY = clf.predict(X_test)
        if fma:
            cm = confusion_matrix(y_test, predictY, labels=uniqueLabels)
        if bna:
            # predictY = convertToStr_bna(predictY)
            cm = confusion_matrix(convertToStr_bna(y_test), convertToStr_bna(predictY), labels=uniqueLabels)
        plt.figure()
        confusionMatrix.plot_confusion_matrix(cm, uniqueLabels, title='Boosted Decision Tree Confusion Matrix')
        # plt.show()

        #ROC
        probs = clf.predict_proba(X_test)
        if fma:
            skplt.metrics.plot_roc_curve(y_test, probs, title='Boosted Decision Tree ROC Curves')
        elif bna:
            skplt.metrics.plot_roc_curve(convertToStr_bna(y_test), probs[::-1], title='Boosted Decision Tree ROC Curves')
        plt.show()

### TRAIN AN SVM ###############################################################
if supVect:

    # paramSearch = False

    if supVect_paramSearch:
        # used gridsearch code from here: https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py

        # numEstimators = [10, 100, 1000]
        # maxDepths = [1, 2, 4]
        # lRates = [0.5, 1, 2]

        if fma:
            cVals = [0.2, 0.4, 0.8, 1.6, 3.2]
            kType = ['linear', 'rbf', 'sigmoid']

            #iter 2
            cVals = [1.6, 3.2, 6.4, 12.8]
            kType = ['linear', 'rbf', 'sigmoid']

        elif bna:
            cVals = [0.2, 0.4, 0.8, 1.6, 3.2]
            kType = ['linear', 'rbf', 'sigmoid']

            cVals = [1, 1.5, 2, 2.5, 3]
            kType = ['linear', 'rbf', 'sigmoid']

            cVals = [2.5, 3, 3.5, 4, 4.5]
            kType = ['linear', 'rbf', 'sigmoid']

        param_grid = {
              'C': cVals,
              'kernel': kType,
             }


        clf = skl.svm.SVC()

        # run grid search
        grid_search_SVM = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, \
                                       verbose=10)
        grid_search_SVM.fit(X_train, y_train)
        print("Best score: %0.3f" % grid_search_SVM.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search_SVM.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        scores = grid_search_SVM.cv_results_['mean_test_score'] \
                            .reshape((len(param_grid['C']), \
                                      len(param_grid['kernel'])), \
                                      order='C')

        # Save the formatted data:
        with open('svmParamSearch.pkl', 'wb') as f:
            pickle.dump([scores], f)

        scores = np.asarray(scores)
        # print(scores)
        c_kern = scores

        #colorbar params
        maxVal = np.amax(scores)
        minVal = np.amin(scores)

        # plotting
        f, ax1 = plt.subplots(1, 1)
        im = ax1.imshow(c_kern, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax1.set_ylabel('C Value')
        ax1.set_yticks(np.arange(len(cVals)))
        ax1.set_yticklabels(cVals)
        ax1.set_xlabel('kernel function')
        ax1.set_xticks(np.arange(len(kType)))
        ax1.set_xticklabels(kType)

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)

        f.suptitle('Support Vector Machine Grid Search')

        plt.show()

    if supVect_learningCurve:

        if fma:
            cVal = 6.4
            kernel = 'rbf'

            numS = 10

        elif bna:
            cVal = 3.5
            kernel = 'rbf'

            numS = 100

        cv = ShuffleSplit(n_splits=numS, test_size=0.2, random_state=0)
        plot_learning_curve(skl.svm.SVC(C=cVal, kernel=kernel), \
                            'SVM Learning Curve (C = ' + str(cVal) + ', kernel = ' + kernel + ')', \
                            X_train, y_train, cv=cv, n_jobs=-1)

        plt.show()

    if supVect_testAssess:

        if fma:
            cVal = 6.4
            kernel = 'rbf'

        elif bna:
            cVal = 3.5
            kernel = 'rbf'

        #Model Accuracy
        clf = skl.svm.SVC(C=cVal, kernel=kernel)
        clf = skl.calibration.CalibratedClassifierCV(clf)
        trainStart = time.time()
        clf = clf.fit(X_train, y_train)
        trainEnd = time.time()

        testStart = time.time()
        score = clf.score(X_test, y_test)
        testEnd = time.time()

        print('SVM accuracy: {:.2%}'.format(score))
        print('Train time: {:.5f}'.format(trainEnd - trainStart))
        print('Test time: {:.5f}'.format(testEnd - testStart))

        if fma:
            uniqueLabels = y_test.unique().tolist()
        elif bna:
            uniqueLabels = ['Genuine', 'Forged']

        #Confusion Mat
        predictY = clf.predict(X_test)
        if fma:
            cm = confusion_matrix(y_test, predictY, labels=uniqueLabels)
        if bna:
            # predictY = convertToStr_bna(predictY)
            cm = confusion_matrix(convertToStr_bna(y_test), convertToStr_bna(predictY), labels=uniqueLabels)
        plt.figure()
        confusionMatrix.plot_confusion_matrix(cm, uniqueLabels, title='Support Vector Machine Confusion Matrix')
        # plt.show()

        #ROC
        probs = clf.predict_proba(X_test)
        if fma:
            skplt.metrics.plot_roc_curve(y_test, probs, title='Support Vector Machine ROC Curves')
        elif bna:
            skplt.metrics.plot_roc_curve(convertToStr_bna(y_test), probs[::-1], title='Support Vector Machine ROC Curves')
        plt.show()

### TRAIN A KNN Classifier #####################################################
if KNN:

    # paramSearch = False

    if KNN_paramSearch:

        if fma:
            #iteration 1
            # numNeighbors = [2, 4, 8, 16, 32]

            #iteration 2
            # numNeighbors = [6, 8, 10, 12]

            #iteration 3
            numNeighbors = [6, 7, 8, 9, 10]

        elif bna:
            #iteration 1
            numNeighbors = [2, 4, 8, 16, 32]

            #iteration 2
            numNeighbors = [1, 2, 4, 8, 16]

        param_grid = {
              'n_neighbors': numNeighbors,
             }

        knnClf = KNeighborsClassifier()

         # run grid search
        grid_search_KNN = GridSearchCV(knnClf, param_grid=param_grid, n_jobs=-1, \
                                       verbose=10)
        grid_search_KNN.fit(X_train, y_train)
        print("Best score: %0.3f" % grid_search_KNN.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search_KNN.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        scores = grid_search_KNN.cv_results_['mean_test_score'] \
                            .reshape((len(param_grid['n_neighbors'])), \
                                      order='C')

        # Save the formatted data:
        with open('kmeansParamSearch.pkl', 'wb') as f:
            pickle.dump([scores], f)

        scores = np.asarray(scores)
        # print(scores)
        kVal = scores
        kVal = np.expand_dims(kVal, axis=0)

        #colorbar params
        maxVal = np.amax(scores)
        minVal = np.amin(scores)

        # plotting
        f, ax1 = plt.subplots(1, 1)
        im = ax1.imshow(kVal, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        # ax1.set_ylabel([])
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax1.set_xlabel('k Value')
        ax1.set_xticks(np.arange(len(numNeighbors)))
        ax1.set_xticklabels(numNeighbors)

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)

        f.suptitle('K-Nearest Neighbor Grid Search')

        plt.show()


    if KNN_learningCurve:

        if fma:
            numNeighb = 8
        elif bna:
            numNeighb = 2


        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        plot_learning_curve(KNeighborsClassifier(n_neighbors=numNeighb), \
                            'KNN Learning Curve (n_neighbors = 8)', \
                            X_train, y_train, cv=cv, n_jobs=-1)

        plt.show()

    if KNN_testAssess:

        if fma:
            numNeighb = 8
        elif bna:
            numNeighb = 2

        #Model Accuracy
        clf = KNeighborsClassifier(n_neighbors=numNeighb)
        trainStart = time.time()
        clf = clf.fit(X_train, y_train)
        trainEnd = time.time()

        testStart = time.time()
        score = clf.score(X_test, y_test)
        testEnd = time.time()

        print('KNN accuracy: {:.2%}'.format(score))
        print('Train time: {:.5f}'.format(trainEnd - trainStart))
        print('Test time: {:.5f}'.format(testEnd - testStart))

        if fma:
            uniqueLabels = y_test.unique().tolist()
        elif bna:
            uniqueLabels = ['Genuine', 'Forged']

        #Confusion Mat
        predictY = clf.predict(X_test)
        if fma:
            cm = confusion_matrix(y_test, predictY, labels=uniqueLabels)
        if bna:
            # predictY = convertToStr_bna(predictY)
            cm = confusion_matrix(convertToStr_bna(y_test), convertToStr_bna(predictY), labels=uniqueLabels)
        plt.figure()
        confusionMatrix.plot_confusion_matrix(cm, uniqueLabels, title='K-Nearest Neighbor Confusion Matrix')
        # plt.show()

        #ROC
        probs = clf.predict_proba(X_test)
        if fma:
            skplt.metrics.plot_roc_curve(y_test, probs, title='K-Nearest Neighbor ROC Curves')
        elif bna:
            skplt.metrics.plot_roc_curve(convertToStr_bna(y_test), probs[::-1], title='K-Nearest Neighbor ROC Curves')
        plt.show()
