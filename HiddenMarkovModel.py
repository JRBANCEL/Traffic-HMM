import numpy

class HiddenMarkovModel(object):
    """
    A Hidden Markov Chain is:
        Q: set of states (hidden)              - Size n
        E: the output alphabet                 - Size m
        Pi: the initial distribution of states - Size n
        A: transition probabilities            - Size n * n
        B: emission probabilities              - Size n * m
    """

    def __init__(self, Q, E, Pi, A, B):
        self.Q = numpy.copy(Q)
        self.E = numpy.copy(E)
        self.Pi = numpy.copy(Pi)
        self.A = numpy.copy(A)
        self.B = numpy.copy(B)

        # Storing useful length for convenience
        self.n = len(Q)
        self.m = len(E)

    def uniformInitialization(self):
        # Initialization of the parameters using a uniform distribution
        self.Pi = [1/float(self.n)] * self.n
        for i in range(self.n):
            for j in range(self.n):
                self.A[i][j] = 1/float(self.n)
            for k in range(self.m):
                self.B[i][k] = 1/float(self.m)

    def randomInitialization(self):
        # Initialization of the parameters using a uniform distribution
        self.Pi = numpy.random.beta(2, 2, self.n)
        self.Pi /= numpy.linalg.norm(self.Pi, 1)
        for i in range(self.n):
            self.A[i] = numpy.random.beta(2, 2, self.n)
            self.A[i] /= numpy.linalg.norm(self.A[i], 1)
            self.B[i] = numpy.random.beta(2, 2, self.m)
            self.B[i] /= numpy.linalg.norm(self.B[i], 1)

    def trainOnObservations(self, O):
        """
        Re-estimate the parameters of the model using Baum-Welch Algorithm
        The algorithm elicit the parameters that maximize the likelihood of
        such a sequence of observations to occur.
            O: sequence of observations - Size t
        """
        # Length of the sequence of observations - Using T for convenience
        T = len(O)
        n = self.n
        m = self.m
        eta = numpy.zeros((T-1, n, n))
        gamma = numpy.zeros((T, n))

        # Forward-Backward Variables
        alpha = self.ForwardVariable(O)
        beta = self.BackwardVariable(O)
        likelihood = sum(alpha[-1])
        old_likelihood = likelihood - 1

        # Main Loop of updating until convergence of the likelihood
        #c = 2
        #while c > 0:#abs(old_likelihood - likelihood) > 0.001:
        #print("Likelihood", likelihood)
        #print("Pi", self.Pi)
        #print("A", self.A)
        #print("B", self.B)
        #print("Alpha", alpha)
        #print("Beta", beta)
        # Gamma and Eta computation
        for t in range(T-1):
            for i in range(n):
                for j in range(n):
                    eta[t][i][j] = alpha[t][i] * self.A[i][j] \
                                   * self.B[j][O[t+1]] * beta[t+1][j] \
                                   / likelihood
        #print("Eta", eta)
        for t in range(T):
            for i in range(n):
                gamma[t][i] = alpha[t][i] * beta[t][i] / likelihood
        #print(gamma)

        # Parameters Updating
        self.Pi = gamma[0]
        for i in range(n):
            for j in range(self.n):
                self.A[i][j] = sum([eta[t][i][j] for t in range(T-1)]) \
                               / sum([gamma[t][i] for t in range(T-1)])
            for k in range(self.m):
                self.B[i][k] = sum([gamma[t][i] for t in range(T) \
                                    if O[t] == k]) \
                               / sum([gamma[t][i] for t in range(T)])

        # Recompute Alpha, Beta and Likelihood
        alpha = self.ForwardVariable(O)
        beta = self.BackwardVariable(O)
        #print(old_likelihood, likelihood)
        old_likelihood = likelihood
        likelihood = sum(alpha[-1])
        #print("Pi", self.Pi)
        #print("A", self.A)
        #print("B", self.B)
        #    c -= 1

    def ForwardVariable(self, O):
        T = len(O)
        alpha = numpy.zeros((T, self.n))
        alpha[0] = [self.Pi[i] * self.B[i][O[0]] for i in range(self.n)]
        for t in range(1, T):
            alpha[t] = [self.B[j][O[t]] * sum([alpha[t-1][i] * self.A[i][j]
                        for i in range(self.n)]) for j in range(self.n)]
        return alpha

    def BackwardVariable(self, O):
        T = len(O)
        beta = numpy.zeros((T, self.n))
        beta[T-1] = [1.] * self.n
        for t in range(T-2, -1, -1):
            beta[t] = [sum([beta[t+1][j] * self.A[i][j] * self.B[j][O[t+1]]
                       for j in range(self.n)]) for i in range(self.n)]
        return beta

    def generateSequence(self, length):
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
        T = len(O)
        delta = numpy.zeros((T, self.n))
        delta[0] = [self.Pi[i] * self.B[i][O[0]] for i in range(self.n)]
        for t in range(1, T):
            delta[t] = [self.B[j][O[t]] * max([delta[t-1][i] * self.A[i][j]
                        for i in range(self.n)]) for j in range(self.n)]
        return max(delta[T-1])

