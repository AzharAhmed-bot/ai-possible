import numpy as np
from torch import nn
from torch.masked import MaskedTensor
from pomegranate.distributions import Categorical, ConditionalCategorical
from pomegranate.bayesian_network import *


probability_Of_rain=torch.tensor([[0.7,0.2,0.1]]) # none,light,heavy

rain = Categorical(probs=probability_Of_rain)

probability_Of_maintenance=torch.tensor([
    [0.4,0.6],
    [0.2,0.8],
    [0.1,0.9]
]) # none:yes,no,light:yes,no,heavy:yes,no

maintenance = ConditionalCategorical(probs=[probability_Of_maintenance])


# Define the probabilities for the train variable
probability_Of_train=torch.tensor([
    [[0.8,0.2],[0.9,0.1]],
    [[0.6,0.4],[0.7,0.3]],
    [[0.4,0.6],[0.5,0.5]]
])

train = ConditionalCategorical(probs=[probability_Of_train])

probability_Of_appointment=torch.tensor([
    [0.9,0.1],
    [0.6,0.4]
])

appointment = ConditionalCategorical(probs=[probability_Of_appointment])

model=BayesianNetwork()

model.add_distributions([rain,maintenance,train,appointment])
model.add_edge(rain,maintenance)
model.add_edge(rain,train)
model.add_edge(maintenance,train)
model.add_edge(train,appointment)
