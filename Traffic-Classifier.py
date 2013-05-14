#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2013, Jean-Rémy Bancel
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Traffic-HMM Project nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Jean-Rémy Bancel BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
import pickle
import re
import sys
import os

import Classifier

def getObservationsFromTrace(obsPath, samples, step):
    """
    This convert a tcpdump output to a sequence of observations
    to feed Baum-Welch Algorithm
    """
    obsFile = open(obsPath, 'r')
    observations = []
    for line in obsFile.readlines():
        # Find size
        m = re.search("length ([0-9]+)", line)
        if m != None:
            # Real Size
            size = int(m.group(1))
            # Sampled Size
            size = min(int(size/step), samples - 1)
            # Find direction
            m = re.search("(ftp|http|ssh) > ", line)
            if m == None:
                observations.append(size)
            else:
                observations.append(size  + samples)
            # Time
            time = line.split(' ')[0]
    obsFile.close()
    return observations

def train(path, maxSize=1000, step=10):
    """
    This function trains a classifier using the data files at path
    The structure should be:
        path/class1/trace_file
            /class2/trace_file
    Such a directory structure generates a classifier with classes
    class1 and class2 and train them with the trace file in the corresponding
    directories
    """
    # List the directory
    os.chdir(path)
    classes = os.listdir()
    print("Training %s Classes" % len(classes))

    samples = int(maxSize/step)

    # Creating the classifier
    Q = ["Insert1", "Server1", "Client1", "Delete1",
         "Insert2", "Server2", "Client2", "Delete2"]
    E = range(2*samples)
    # Two-Match HMM
    TM = numpy.array([
                      [1, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1, 1, 0],
                      [1, 1, 1, 1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 0, 0, 0, 0],
                     ])
    # Delete states do not emit anything
    EM = numpy.array([
                      numpy.ones(2 * samples),
                      numpy.ones(2 * samples),
                      numpy.ones(2 * samples),
                      numpy.zeros(2 * samples),
                      numpy.ones(2 * samples),
                      numpy.ones(2 * samples),
                      numpy.ones(2 * samples),
                      numpy.zeros(2 * samples),
                     ])
    S = [False, False, False, True, False, False, False, True]
    classifier = Classifier.Classifier(Q, E, TM=TM, EM=EM, S=S)

    # Training
    for className in classes:
        os.chdir(className)
        if len(os.listdir()) == 0:
            os.chdir("..")
            continue
        print("Training class", className, "with", len(os.listdir()), "samples")
        # Setting random parameters before training
        classifier.addClass(className)
        classifier.resetClass(className)

        # Training the class on all the observations in the directory
        for obsPath in os.listdir():
            print("Training with", obsPath)
            classifier.trainClass(className,
                getObservationsFromTrace(obsPath, samples, step))
        os.chdir("..")

    os.chdir("..")
    # Saving the classifier
    dump = open("HMM.dump", 'bw')
    pickle.dump(classifier, dump)
    dump.close()

def classify(HMMpath, tracePath, maxSize=1000, step=10):
    samples = int(maxSize/step)
    HMMFile = open("HMM.dump", 'rb')
    classifier = pickle.load(HMMFile)
    print(classifier.classify(getObservationsFromTrace(tracePath, samples, step)))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise "Not enough arguments"

    if sys.argv[1] == "train":
        train(sys.argv[2])
    elif sys.argv[1] == "classify":
        classify(sys.argv[2], sys.argv[3])
