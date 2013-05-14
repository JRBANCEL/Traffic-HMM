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

import HiddenMarkovModel

class Classifier(object):
    def __init__(self, Q, E, PM=None, TM=None, EM=None, S=None):
        """
        Create a classifier with the following parameters
            Q: set of states (hidden) - Size n
            E: the output alphabet    - Size m
        """
        self.Q = numpy.copy(Q)
        self.E = numpy.copy(E)
        self.PM = PM
        self.TM = TM
        self.EM = EM
        self.S = S

        # Storing useful lengths for convenience
        self.n = len(Q)
        self.m = len(E)

        # Creating the dictionnary of classes
        self.classes = {}

    def removeClass(self, className):
        if className in self.classes:
            del self.classes[className]
        else:
            raise "No such class in the classifier"

    def addClass(self, className):
        if className in self.classes:
            raise Exception("Class already in the classifier")
        else:
            self.classes[className] = HiddenMarkovModel.HiddenMarkovModel(
                                      self.Q, self.E, PM=self.PM, TM=self.TM,
                                      EM=self.EM, S=self.S)

    def resetClass(self, className):
        """
        Initialize the parameters of the HMM of the class
        """
        if className in self.classes:
            self.classes[className].randomInitialization()
        else:
            raise Exception("No such class in the classifier")

    def trainClass(self, className, observations):
        """
        Train a class with the observations
        """
        if className in self.classes:
            self.classes[className].trainOnObservations(observations)
        else:
            raise Exception("No such class in the classifier")

    def classify(self, observations, threshold=-3000):
        """
        Returns the class that is most likely to have generated the observations
        """
        maxScore = -numpy.inf
        maxName = None
        for name, hmm in self.classes.items():
            score = hmm.viterbiScore(observations)
            print(name, score)
            if score > maxScore:
                maxScore = score
                maxName = name
        if maxScore < threshold:
            return None
        return maxName
