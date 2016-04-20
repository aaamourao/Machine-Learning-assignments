#
#   perceptron.py
#   Class which implements perceptron algorithm
#
#   DCC 030 049 831: Machine Learning
#   Programming Assignment 1: Perceptron and SVM
#       by @aaamourao: Adriano Mourao
#
#   mourao.aaa@gmail.com
#   encrypted mail:adrianomourao@protonmail.com
#
from collections import Counter
import operator
import os
import pickle

import matplotlib.pyplot as plt

defaultAxis = [0, 405, 0, 20]

class Perceptron():
    def __init__(self, dataFile, dataFolder, threshold=30, verbose=False):
        self.trainingFile = os.path.join(dataFolder, 'spam_training.txt')
        self.validationFile = os.path.join(dataFolder, 'spam_validation.txt')
        self.dumpFile = os.path.join(dataFolder, 'perceptronDumpData')
        self.dumpFigure = os.path.join(dataFolder, 'training.png')

        # Create folder for training data, if does not exist
        if not os.path.exists(dataFolder):
            os.makedirs(dataFolder)

        # Take 20% of the data file as validation data
        if verbose:
            print 'Opening data files'
        with open(dataFile, 'r') as allDataLines:
            allData = allDataLines.readlines()
            self.trainingData = allData[:int(0.8*len(allData))]
            self.validationData = allData[len(self.trainingData):]

            with open(self.trainingFile, 'w') as fp:
                fp.write(''.join(self.trainingData))
            with open(self.validationFile, 'w') as fp:
                fp.write(''.join(self.validationData))

        self.verbose = verbose
        self.threshold = threshold;
        if verbose:
            print 'Loading data into data structures'
        self.__loadData()
        return None

    def __loadData(self):
        # isSpam = dataTable[0], mail's words counter = dict(dataTable[1])
        self.trainingSet = list(
                (int(data[0]), Counter(''.join(data[2:]).split()))
                for data in self.trainingData
        )
        self.validationSet = list(
                (int(data[0]), Counter(''.join(data[2:]).split()))
                for data in self.validationData
        )
        # Words counter of all dataset
        wordsCounter = Counter(
                ''.join(
                    [''.join(data[2:]) for data in self.trainingData]
                ).split()
        )
        # Features: all items that appear more than `threshold` times
        self.features = {k:v for k,v in wordsCounter.items() if v > self.threshold}
        # Sort Features Keys for computing
        self.featKeys = list(sorted(self.features.keys()))
        return None

    def perceptron_train(self, dataFile=None):
        if dataFile != None:
            # Take 20% of the data file as validation data
            with open(dataFile, 'r') as allDataLines:
                allData = allDataLines.readlines()
                self.trainingData = allData[:int(0.8*len(allData))]
                self.validationData = allData[len(self.trainingData):]
            self.__loadData()
        elif not dataFile and not self.trainingSet:
            raise "dataFile is required for static running"

        # Initialize weight vector, k and iter param
        self.w = dict.fromkeys(self.featKeys, 0)
        self.k = 0
        self.it = 0
        allGood = False

        # Size of data sets
        setSize = len(self.trainingSet)
        valSetSize = len(self.validationSet)
        # Errors history
        self.valError = []
        self.trainError = []
        # Create plot window
        if self.verbose:
            plt.axis(defaultAxis)
            plt.xlabel('Iteration')
            plt.ylabel('Error %')
            plt.ion()
            print 'Initializing training'
        while not allGood:
            errPerEpoch = 0
            self.it += 1

            allGood = True
            for isSpam, featVec in self.trainingSet:
                # Compute r = YXtW; If the featVec' doesnt have the key
                # the result is 0 and does not need to be computed:
                # sparce representation
                y = 1 if isSpam else -1
                yixiw = 0
                for key in self.featKeys:
                    if featVec.has_key(key):
                        yixiw += y*featVec[key]*self.w[key]
                if yixiw <= 0:
                    self.k += 1
                    allGood = False
                    errPerEpoch += 1
                    for key in self.featKeys:
                        yixi = y*featVec[key] if featVec.has_key(key) else 0
                        self.w[key] = self.w[key] + yixi
            # Get training and validation error
            val = self.__run(self.validationSet, self.w)
            self.trainError.append(100.0*errPerEpoch/float(setSize))
            self.valError.append(100.0*val[0]/float(valSetSize))
            if self.verbose:
                plt.plot(range(self.it), self.trainError, 'ro-')
                plt.plot(range(self.it), self.valError, 'b*-')
                plt.draw()
        if self.verbose:
            print 'Finished!!!'
            print 'Training error: ' + str(self.trainError[-1])
            print 'Validation error: ' + str(self.valError[-1])
        return (self.w, self.k, self.it)

    def __run(self, genSet, w):
        errorCnt = 0
        hitCnt = 0
        for isSpam, featVec in genSet:
            y = 1 if isSpam else -1
            yixiw = 0
            for key in self.featKeys:
                if featVec.has_key(key):
                    yixiw += y*featVec[key]*w[key]
            if yixiw <= 0:
                errorCnt += 1
            else:
                hitCnt += 1
        return (errorCnt, hitCnt)

    def perception_test(self, genFile, w=None):
        if not w:
            w = self.w
        with open(genFile, 'r') as f:
            genData = f.readlines()
        genSet = list(
                (int(data[0]), Counter(''.join(data[2:]).split()))
                for data in self.genSet
        )
        return self.__run(genSet, w)

    def plotResults(self, axis=defaultAxis):
        if not self.trainError or not self.valError:
            raise "No training data found"
        plt.axis(axis)
        plt.xlabel('Iteration')
        plt.ylabel('Error %')
        plt.plot(range(self.it), self.trainError, 'ro-')
        plt.plot(range(self.it), self.valError, 'b*-')
        plt.show()
        plt.draw()
        plt.savefig(self.dumpFigure, dpi=100)
        plt.close()
        return None

    def dumpData(self):
        with open(self.dumpFile, 'w') as f:
            pickle.dump([self.w, self.k, self.it, self.trainError, self.valError], f)
        self.plotResults()
        return None

    def getDumpData(self):
        with open(self.dumpFile, 'r') as f:
            self.w, self.k, self.it = pickle.load(f)
        return None
