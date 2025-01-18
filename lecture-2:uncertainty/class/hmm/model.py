from pomegranate.distributions import Categorical,ConditionalCategorical
from pomegranate.hmm import DenseHMM
from pomegranate.markov_chain import *
import numpy as np

# Probabulity of carrying umbrella when its sunny
probability_during_Sunny=torch.tensor([[0.2,0.8]])

sun=Categorical(probs=probability_during_Sunny)

# Probabulity of carrying umbrella when its rainy
probability_during_Rainy=torch.tensor([[0.9,0.1]])

rain=Categorical(probs=probability_during_Rainy)

states=[sun,rain]

transition=[
    [0.8,0.2], # Sunny--->Sunny or Rain
    [0.3,0.7] # Rain--->Sunny or Rain
]

start=[0.5,0.5]


# Create the model
model=DenseHMM(states,edges=transition,starts=start)

