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

# Parameters of the test
training_sequences = [10, 10, 10]   # Number of training sequences
training_length = [100, 100, 100]   # Length of the training sequences
testing_sequences = [10, 10, 10]    # Number of testing sequences
testing_length = [100, 100, 100]    # Length of the testing sequences

# Set 1
A = numpy.array([
                [1/3., 1/3., 1/3.],
                [1/6., 2/3., 1/6.],
                [0, 1/2., 1/2.]
                ])
B = numpy.array([
                [1/3., 2/3., 0],
                [3/4., 1/8., 1/8.],
                [1/2., 0, 1/2.]
                ])
Q = ["0", "1", "2"]
E = ["A", "B", "C"]
Pi = numpy.array([1/2., 1/2., 0])
HMM1 = HiddenMarkovModel.HiddenMarkovModel(Q, E, Pi, A, B)
observations1 = [HMM1.generateSequence(training_length[0])
                 for _ in range(training_sequences[0])]

## Set 2
A = numpy.array([
                [1/2., 1/4., 1/4.],
                [1/3., 2/3., 0],
                [1/3., 1/3., 1/3.]
                ])
B = numpy.array([
                [1/3., 0., 2/3.],
                [1/4., 1/2., 1/4.],
                [1/4., 3/4., 0.]
                ])
Q = ["0", "1", "2"]
E = ["A", "B", "C"]
Pi = numpy.array([1/3., 1/3., 1/3.])
HMM2 = HiddenMarkovModel.HiddenMarkovModel(Q, E, Pi, A, B)
observations2 = [HMM2.generateSequence(training_length[1])
                 for _ in range(training_sequences[1])]

# Set 3
A = numpy.array([
                [.35, .15, .5],
                [.05, .05, .9],
                [.34, .26, .4]
                ])
B = numpy.array([
                [0., 1/3., 2/3.],
                [1/2., 0., 1/2.],
                [1/4., 0., 3/4.]
                ])
Q = ["0", "1", "2"]
E = ["A", "B", "C"]
Pi = numpy.array([1/6., 2/3., 1/6.])
HMM3 = HiddenMarkovModel.HiddenMarkovModel(Q, E, Pi, A, B)
observations3 = [HMM3.generateSequence(training_length[2])
                 for _ in range(training_sequences[2])]

# Training
HMMT1 = HiddenMarkovModel.HiddenMarkovModel(Q, E, Pi, A, B)
HMMT1.randomInitialization()
for i, observation in enumerate(observations1):
    HMMT1.trainOnObservations(observation)
HMMT2 = HiddenMarkovModel.HiddenMarkovModel(Q, E, Pi, A, B)
HMMT2.randomInitialization()
for i, observation in enumerate(observations2):
    HMMT2.trainOnObservations(observation)
HMMT3 = HiddenMarkovModel.HiddenMarkovModel(Q, E, Pi, A, B)
HMMT3.randomInitialization()
for i, observation in enumerate(observations3):
    HMMT3.trainOnObservations(observation)

# Testing Sets
observations1 = [HMM1.generateSequence(testing_length[0])
                 for _ in range(testing_sequences[0])]
observations2 = [HMM2.generateSequence(testing_length[1])
                 for _ in range(testing_sequences[1])]
observations3 = [HMM3.generateSequence(testing_length[2])
                 for _ in range(testing_sequences[2])]

# Testing
test = 0
success = 0
for i, observation in enumerate(observations1):
    if HMMT1.viterbiScore(observation) > \
       max(HMMT2.viterbiScore(observation), HMMT3.viterbiScore(observation)):
        success += 1
    test += 1
for i, observation in enumerate(observations2):
    if HMMT2.viterbiScore(observation) > \
       max(HMMT3.viterbiScore(observation), HMMT1.viterbiScore(observation)):
        success += 1
    test += 1
for i, observation in enumerate(observations3):
    if HMMT3.viterbiScore(observation) > \
       max(HMMT2.viterbiScore(observation), HMMT1.viterbiScore(observation)):
        success += 1
    test += 1
print("Accuracy: %.2lf%%" % (success/float(test)))
