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

import HiddenMarkovModel

class Classifier(object):
    def __init__(self, Q, E, Pi, A, B):
    """
    Create a classifier with the following parameters
        Q: set of states (hidden) - Size n
        E: the output alphabet    - Size m
    """
        self.Q = numpy.copy(Q)
        self.E = numpy.copy(E)

        # Storing useful lengths for convenience
        self.n = len(Q)
        self.m = len(E)

        # Creating the dictionnary of classes
        self.classes = {}

    def removeClass(self, className):
        if self.classes.has_key(className):
            del self.classes[className]
        else:
            raise "No such class in the classifier"

    def addClass(self, className):
        if self.classes.has_key(className):
            raise "Class already in the classifier"
        else:
            self.classes[className] = HiddenMarkovModel.HiddenMarkovModel(
                                      self.Q, self.E, numpy.zeros(n),
                                      numpy.zeros((n, n)), numpy.zeros((n, m)))

    def resetClass(self, className):
        pass

    def trainClass(self, className, observations):
        pass

    def classify(self, observations):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass
