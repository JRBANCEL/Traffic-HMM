import numpy

import HiddenMarkovModel

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
observations1 = [HMM1.generateSequence(100) for _ in range(10)]

# Set 2
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
observations2 = [HMM2.generateSequence(100) for _ in range(10)]

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
observations3 = [HMM3.generateSequence(100) for _ in range(10)]

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
observations1 = [HMM1.generateSequence(100) for _ in range(500)]
observations2 = [HMM2.generateSequence(100) for _ in range(500)]
observations3 = [HMM3.generateSequence(100) for _ in range(500)]

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
print(success/float(test))
#print("Dist Pi:", numpy.linalg.norm(HMM2.Pi - Pi))
#print("Dist A:", numpy.linalg.norm(HMM2.A - A))
#print("Dist B:", numpy.linalg.norm(HMM2.B - B))