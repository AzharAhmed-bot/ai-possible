from model import model

# Calculate probability for a given observation
# ["none", "yes", "delayed", "attend"]
probability = model.probability([[0,1,0,0]])

print(probability)
