#
#   train.py
#   Executes trainings
#
#   DCC 030 049 831: Machine Learning
#   Programming Assignment 1: Perceptron and SVM
#       by @aaamourao: Adriano Mourao
#
#   mourao.aaa@gmail.com
#   encrypted mail:adrianomourao@protonmail.com
#
from perceptron import Perceptron

trainFile = '../spam_train.txt'
testFile = '../spam_test.txt'
dataFolder = './training-data'

def main():
    neuron = Perceptron(trainFile, dataFolder, verbose=True)
    results = neuron.perceptron_train()
    neuron.dumpData()

if __name__ == "__main__":
    main()
