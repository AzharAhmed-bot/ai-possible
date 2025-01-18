from model import model
from collections import Counter

N=1000
data=[]

for i in range(N):
    sample=model.sample(1)
    # If in the sample, the variable of train has the value delayed, save the sample
    # Since we are interested in the probability of train delayed we discard train was on time when getting the number of times I have attended the appointment
    if sample[:,2]==1:
        data.append(sample[:,3].item())



# Get the count of the number of times I have attended the appointment
count=Counter(data)
probability_Of_attending=count[0]/ sum(count.values())
print(count)
print(f"probability that you attend given train is on time: {probability_Of_attending:.4f}")