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

class HiddenMarkovModel(object):
    """
    A Hidden Markov Chain is:
        Q: set of states (hidden)              - Size n
        E: the output alphabet                 - Size m
        Pi: the initial distribution of states - Size n
        A: transition probabilities            - Size n * n
        B: emission probabilities              - Size n * m
        TM: transition mask                    - Size n * n
        PM: Pi mask                            - Size n
        EM: emission mask                      - Size n * m
        S: silent states                       - Size m
    """

    def __init__(self, Q, E, Pi=None, A=None, B=None, TM=None, PM=None,
                 EM=None, S=None):
        self.Q = numpy.copy(Q)
        self.E = numpy.copy(E)

        # Storing useful lengths for convenience
        n = self.n = len(Q)
        m = self.m = len(E)

        if Pi != None:
            self.Pi = numpy.copy(Pi)
        else:
            self.Pi = numpy.zeros(n)
        if A != None:
            self.A = numpy.copy(A)
        else:
            self.A = numpy.zeros((n, n))
        if B != None:
            self.B = numpy.copy(B)
        else:
            self.B = numpy.zeros((n, m))
        if TM != None:
            self.TM = numpy.copy(TM)
        else:
            self.TM = numpy.ones((n, n))
        if PM != None:
            self.PM = numpy.copy(PM)
        else:
            self.PM = numpy.ones(n)
        if EM != None:
            self.EM = numpy.copy(EM)
        else:
            self.EM = numpy.ones((n, m))
        if S != None:
            self.S = numpy.copy(S)
        else:
            self.S = [False] * n


    def randomInitialization(self, a=2, b=2):
        """
        Initialize the parameters using a beta(a,b) distribution
        A Beta(2,2) is nice because generates parameters that are of the order
        of magnitude
        """
        for i in range(self.n):
            if self.PM[i] != 0:
                self.Pi[i] = numpy.random.beta(a, b)
            else:
                self.Pi[i] = 0
            for j in range(self.n):
                if self.TM[i][j] != 0:
                    self.A[i][j] = numpy.random.beta(a, b)
                else:
                    self.A[i][j] = 0
            if numpy.linalg.norm(self.A[i], 1) > 0:
                self.A[i] /= numpy.linalg.norm(self.A[i], 1)
            for j in range(self.m):
                if self.EM[i][j] != 0:
                    self.B[i][j] = numpy.random.beta(a, b)
                else:
                    self.B[i][j] = 0
            if numpy.linalg.norm(self.B[i], 1) > 0:
                self.B[i] /= numpy.linalg.norm(self.B[i], 1)
        self.Pi /= numpy.linalg.norm(self.Pi, 1)

    def trainOnObservations(self, O, iterations=5):
        """
        Re-estimate the parameters of the model using Baum-Welch Algorithm
        The algorithm elicit the parameters that maximize the likelihood of
        such a sequence of observations to occur.
            O: sequence of observations - Size T
            iterations: maximal number of iterations of the update process
        """
        # Length of the sequence of observations - Using T for convenience
        T = len(O)
        n = self.n
        m = self.m
        eta = numpy.zeros((T-1, n, n))
        gamma = numpy.zeros((T, n))

        # Forward-Backward Variables
        alpha = self.forwardVariable(O)
        beta = self.backwardVariable(O)

        # Log likelihood
        likelihood = -sum([numpy.log(c) for c in self.c])
        old_likelihood = likelihood - 1

        # Main Loop of updating until convergence of the likelihood
        while iterations > 0 and likelihood > old_likelihood:
            print("Likelihood:", likelihood)
            #print("Alpha:", alpha)
            print("Beta:", beta)
            #print("B:", self.B)
            # Gamma and Eta computation
            for t in range(T-1):
                denominator = 0
                for i in range(n):
                    for j in range(n):
                        denominator += alpha[t][i] * self.A[i][j] \
                                       * self.B[j][O[t+1]] * beta[t+1][j]

                #print("Denominator(t=", t, ")=", denominator)
                for i in range(n):
                    gamma[t][i] = 0
                    for j in range(n):
                        if denominator > 0:
                            eta[t][i][j] = alpha[t][i] * self.A[i][j] \
                                           * self.B[j][O[t+1]] * beta[t+1][j] \
                                           / denominator
                            gamma[t][i] += eta[t][i][j]
                        else:
                            eta[t][i][j] = 0.
                            gamma[t][i] = 0.

            # Parameters Updating
            self.Pi = [gamma[0][i] * self.PM[i] for i in range(self.n)]
            self.Pi /= numpy.linalg.norm(self.Pi, 1)
            for i in range(n):
                for j in range(self.n):
                    if self.TM[i][j] != 0:
                        if sum([gamma[t][i] for t in range(T-1)]) > 0:
                            self.A[i][j] = sum([eta[t][i][j] for t in range(T-1)]) \
                                           / sum([gamma[t][i] for t in range(T-1)])
                        else:
                            self.A[i][j] = 0
                    else:
                        self.A[i][j] = 0
                self.A[i] /= numpy.linalg.norm(self.A[i], 1)
                for k in range(self.m):
                    if self.EM[i][k] != 0:
                        self.B[i][k] = sum([gamma[t][i] for t in range(T) \
                                            if O[t] == k]) \
                                       / sum([gamma[t][i] for t in range(T)])
                    else:
                        self.B[i][k] = 0
                if not self.S[i]:
                    self.B[i] /= numpy.linalg.norm(self.B[i], 1)

            # Recompute Alpha, Beta and Likelihood
            alpha = self.forwardVariable(O)
            beta = self.backwardVariable(O)
            old_likelihood = likelihood
            likelihood = -sum([numpy.log(c) for c in self.c])
            iterations -= 1

    def forwardVariable(self, O):
        """
        Compute the forward variable a.k.a. as alpha in the litterature
        It relies on a recursive formula to do it in O(n^2T)
            O: sequence of observations - Size T
        """
        T = len(O)
        alpha = numpy.zeros((T, self.n))
        c = numpy.zeros(T)

        # Initialization
        alpha[0] = [self.Pi[i] * self.B[i][O[0]] for i in range(self.n)]
        c[0] = sum(alpha[0])

        # Scaling (avoid underflow)
        c[0] = 1/c[0]
        alpha[0] *= c[0]

        # Recursive computation of alpha
        for t in range(1, T):
            alpha[t] = [(self.B[j][O[t]] if not self.S[j] else 1) *
                        sum([alpha[t-1][i] * self.A[i][j]
                        for i in range(self.n)]) for j in range(self.n)]
            c[t] = sum(alpha[t])

            # Scaling
            if c[t] > 0:
                c[t] = 1/c[t]
                alpha[t] *= c[t]
            else:
                alpha[t] = [1/float(self.n)] * self.n
                c[t] = 1.

        # Exporting the scaling constants to the global scope to be used in
        # backward computation
        self.c = c
        return alpha

    def backwardVariable(self, O):
        """
        Compute the backward variable a.k.a. as beta in the litterature
        It relies on a recursive formula to do it in O(n^2T)
        Note that forwardVariable has to be called before in order for
        self.c to be valid.
            O: sequence of observations - Size T
        """
        T = len(O)
        beta = numpy.zeros((T, self.n))
        beta[T-1] = [self.c[T-1]] * self.n
        for t in range(T-2, -1, -1):
            beta[t] = [sum([self.A[i][j] * (beta[t+1][j] * self.B[j][O[t+1]] if
                       not self.S[j] else beta[t+1][j])
                       for j in range(self.n)]) for i in range(self.n)]
            # Scaling
            beta[t] *= self.c[t]

            # Dealing with empty array
            if not sum(beta[t]) > 0:
                beta[t] = [1/float(self.n)] * self.n
        return beta

    def generateSequence(self, length):
        """
        Generate a sequence of observations of length @length
            length: length of the sequence
        """
        q = []

        # Picking initial state
        state = sum([n*i for n,i in
                enumerate(numpy.random.multinomial(1, self.Pi))])
        q.append(state)

        while len(q) < length:
            # Choosing the next state
            state = sum([n*i for n,i in
                    enumerate(numpy.random.multinomial(1, self.A[state]))])
            # Drawing a symbol
            symbol = sum([n*i for n,i in
                     enumerate(numpy.random.multinomial(1, self.B[state]))])
            q.append(symbol)
        return q

    def viterbiScore(self, O):
        """
        Compute the Viterbi Score of the observation
        The Higher the score, the higher the probability of being generated by
        this hidden markov model.
        Note that the logarithmic variable is used to avoid underflow
            O: sequence of observations - Size T
        """
        T = len(O)
        delta = numpy.zeros((T, self.n))
        delta[0] = [numpy.log(self.Pi[i]) + numpy.log(self.B[i][O[0]])
                    for i in range(self.n) if self.B[i][O[0]] > 0]
        for t in range(1, T):
            delta[t] = [numpy.log(self.B[j][O[t]]) \
                        + max([delta[t-1][i] + numpy.log(self.A[i][j])
                        for i in range(self.n)]) for j in range(self.n)]
        return max(delta[T-1])

