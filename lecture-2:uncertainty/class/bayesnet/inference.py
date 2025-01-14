from model import model
from pomegranate.bayesian_network import *
from torch.masked import MaskedTensor

# Calculate predictions

observation = torch.tensor([[0, 0, 1, 0]])  # Given evidence (observation)
mask = torch.tensor([[False, False, True, False]])  # Which variables are observed?

X = MaskedTensor(observation, mask)  # Create MaskedTensor with observation and mask


predictions=model.predict_proba(X)

node_names=["Rain","Maintenance","Train","Appointment"]

states={
    'Rain':['none','light','heavy'],
    'Maintenance':['yes','no'],
    'Train':['On time','Delayed'],
    'Appointment':['Attended','Miss']
}



for node_name, prediction in zip(node_names, predictions):
    print(f"Predictions for {node_name}:")
    # Get the probabilities and associated states for the node
    for i, prob in enumerate(prediction[0]):
        print(f"  {states[node_name][i]}: {prob.item():.4f}")
    print("\n")