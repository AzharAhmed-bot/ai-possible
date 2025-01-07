from pomegranate.distributions import DiscreteDistribution

rain = DiscreteDistribution({
    "none": 0.7,
    "light": 0.2,
    "heavy": 0.1
})
print(rain)
