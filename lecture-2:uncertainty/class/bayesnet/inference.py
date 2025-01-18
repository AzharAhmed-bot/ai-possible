# Import necessary libraries
from model import model
from pomegranate.bayesian_network import *
from torch.masked import MaskedTensor

# Calculate predictions
# Define the observation (evidence) and mask for the Bayesian network
# The observation represents the values of the variables we have observed
# The mask represents which variables are observed (True) and which are not (False)
observation = torch.tensor([[0, 0, 1, 0]])  # Given evidence (observation)
mask = torch.tensor([[False, False, True, False]])  # Which variables are observed?

# Create a MaskedTensor with the observation and mask
X = MaskedTensor(observation, mask)

# Use the model to predict the probabilities of the unobserved variables
predictions = model.predict_proba(X)

# Define the names of the nodes in the Bayesian network
node_names = ["Rain", "Maintenance", "Train", "Appointment"]

# Define the possible states for each node
states = {
    'Rain': ['none', 'light', 'heavy'],
    'Maintenance': ['yes', 'no'],
    'Train': ['On time', 'Delayed'],
    'Appointment': ['Attended', 'Miss']
}

# Print the predictions for each node
for node_name, prediction in zip(node_names, predictions):
    print(f"Predictions for {node_name}:")
    # Get the probabilities and associated states for the node
    for i, prob in enumerate(prediction[0]):
        print(f"  {states[node_name][i]}: {prob.item():.4f}")
    print("\n")