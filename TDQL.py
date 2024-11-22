import numpy as np
import os
import sys

class td_qlearning:

  alpha = 0.10
  gamma = 0.95
  Q = {}
  moves = {
      'A': ['N', 'B', 'D'],
      'B': ['N'],
      'C': ['N', 'B', 'E'],
      'D': ['N', 'F', 'E', 'A'],
      'E': ['N', 'D', 'C', 'F'],
      'F': ['N', 'D', 'E']
    }

  def __init__(self, directory):
    # directory is the path to a directory containing trials through state space
    # Return nothing

    qDict = {}

    #Creating all possible states and initializing Q-values
    nodes = ['A','B','C','D','E','F']
    states=[]
    for m in nodes:
      for c in nodes:
        currState = m+c
        states.append(currState)

        #Puts every (state, action) combo into qDict
        for move in self.moves[m]:
          qDict[(currState, move)] = self.reward(currState)

    #Loading trials
    trialData = []
    for filename in os.listdir(directory):
      path = directory+"\\"+filename
      #print(path)
      f = open(path, "r")
      currTrial = []

      for line in f:
        d = line.split(',')
        currTrial.append((d[0],d[1][0]))

      trialData.append(currTrial)

    #Update Q-Values until convergence
    converged = False
    iterations = 0
    while not converged and iterations < 10000:
      iterations += 1
      converged = True
      oldQDict = qDict.copy()

      for trial in trialData:
        #print(trial)
        for i, line in enumerate(trial):
          state = line[0]
          action = line[1]

          if i + 1 >= len(trial):
            break

          nextState = trial[i+1][0]
          reward = self.reward(state)

          #This next bit is the Q learning function broken down
          maxNextQ = max(qDict.get((nextState, m)) for m in self.moves.get(nextState[0]))

          oldQ = qDict[(state,action)]

          qDict[(state, action)] += self.alpha * ( reward + self.gamma * float(maxNextQ) - oldQ)

      #Check if still converging
      for pair in oldQDict:
        if abs(qDict[pair] - oldQDict[pair]) > 0.0005:
          converged = False
          break

    self.Q = qDict

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is a string representation of an action

    q = self.Q.get((state,action))

    # Return the q-value for the state-action pair
    return q

  def policy(self, state):
    # state is a string representation of a state

    # Calculate highest action
    action_q_values = [(self.Q.get((state, m)), m) for m in self.moves.get(state[0])]
    _, a = max(action_q_values, key=lambda x: x[0])

    # Return the optimal action under the learned policy
    return a
  
  def reward(self, state):
    #Setting initial Q-Values according to r(s)
        if state[0]=='B':
          return 10
        elif state[0] == state[1]:
          return -10
        else:
          return -1
