from pomegranate.distributions import Categorical,ConditionalCategorical
from pomegranate.markov_chain import *
from collections import Counter 


probabiliy_Of_weather=torch.tensor([[0.5,0.5]]) # Sunny rainy

weather=Categorical(probs=probabiliy_Of_weather)

probabiliy_Of_Transition=torch.tensor([
    [0.8,0.2], # Sunny--->Sunny or Rain
    [0.3,0.7] # Rain--->Sunny orRain
])

transition=ConditionalCategorical(probs=[probabiliy_Of_Transition])

model=MarkovChain([weather,transition])

N=100
sample=[]
for i in range(N):
    samples=model.sample(1)
    if samples[:,0]==1:
        sample.append(samples[:,1].item())

weather_map={0:"Sunny",1:"Rainy"}
descriptive_sample=[weather_map[s] for s in sample]
weather_counts=Counter(descriptive_sample)

# Print the results descriptively
print("Weather on the second day when the first day is Rainy")
for weather,count in weather_counts.items():
    print(f"{weather}:{count} times")

total_samples=sum(weather_counts.values())

# Calculate and print probabilities
for weather,count in weather_counts.items():
    probability=count/total_samples
    print(f"Probability of  {weather}:{probability:.4f}")
