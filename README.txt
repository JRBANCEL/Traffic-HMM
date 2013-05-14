To generate a classifier by training:

python Traffic-Classifier.py train traces

----------------------------------------------------
To classify a trace:

python Traffic-Classifier.py classify HMM.dump traces/ftp/ftp1

Known Problems:
Sometimes, the computation of beta leads to 0 in the array and then all the
algorithm is wrong. Then the classification leads to -infinity scores...
I am planning to fix that in the next weeks.
