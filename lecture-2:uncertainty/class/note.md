# Uncertainty
When AI is deriving knowledge from its environment in reality its only partial knowledge of the world. This leaves room for unvertainty. However, when the agent is making choices we need it to make choices at 100% accuracy. This is where the probabilistic nature of agents come in hand.

## Probabbility
Every world we live in probability has a  denotation of ω of events to possibly occur. <br>
For example in our own earth the porbability of day and night are as follows
```bash
P(day)=0.5
P(night)=0.5
P(any other situation)=0
```
There are different kinds of probability models
### 1. Unconditional probability
This model presents the probability of an event occuring independently. For example when rolling a dice twice, its independent. The previous roll does not depend on the current roll.

### 2. Conditional probability
This model emphasises on an event occuring dependent on another event. For example picking a number of balls from a back without returning them
There are formulas to get the conditional proability of an event occuring given a certain event occurs often denoted as **P(a|b)**. Whats the probability of a occuring given b.
```bash
P(a|b)=P(a,b) / P(b)
```
This also means that
```bash
P(a^b)=P(a|b) * P(b)
      or
P(a,b)=P(b|a)* P(a)
```
**Random Variables** in probability is the possible values the model can take. For example in weather the r.v are 
- Sunny
- Windy
- Rainy
- Cloudy

### Bayes Rule
This rule is the most commonly used rule in theory of probability 
```bash
P(b|a)=(P(b)* P(a|b))/P(a)
```

### Joint probability
This the probability of multiple events occuring simultaneously.
-N/B:**I have draw the tables on book so here I'll just be abstract**

So to present our probability models algorithmically, we use one of the following:
- Bayesian Network
- Markov model
- Hidden markov model

## Bayesian Network
Invented by Judea Pearl its actually a data structure that represents the dependencies among random variables. Bayesian networks have the following properties
- Are directed graphs
- each node represents a random variable
- Arrows from node to node represent parent child relationship
- Each node represents a probability of P[X| Parent(x)]




