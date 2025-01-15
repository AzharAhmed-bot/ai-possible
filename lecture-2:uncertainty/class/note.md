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

So how does the agent deduce information from this probability model. This is where inferencing comes in

### Inferencing in Probabilistic
It's the conclusion of new information based on the information we already have. Inferencing has the following properties:
- Query X:The variable which we want the probability distribution
- Evidence E: The set of observed events.
- Hidden Y: This is the event we havent observed
- Goal: Finally I'll want to know P[X|e]
For example:
```bash
Compute: P[Appointment| light,no]
e:light rain,no maintenance
x:appointment
```
Thus we use inference by enumeration to find the probability of X given the observed e and hidden y.<br>
However, using bayesian network can be inefficient,especially when there are multiple random variables. Thus, a good way is to abandon **exact inferencing** and go with **approxiamte inferencing**. We will lose precision but acheive scalability.

## Sampling
It is a technique for approximate inferencing, where each random variable is sampled for a variable according to its probability distibution. But sometimes instead of discarding samples, we adjust the importance or rather weight onf an evidence, making it more efficient.<br>
Likelihood weighthood uses the following steps:
- Start by fixing values for evidence variables
- Sample the non evidence variables using conditional probablity
- Weight each sample by its likelihood 

So sampling is essential when working with complex distribution especially when its difficult to calculate distribution and handle dependencies,

## Markov Model
In this model it  brings the dimension of time. Where we not only include some information that we observed but predict how a variable changes with time.<br>
**The markov assumption**- States that the current state depends on only a finite number of previous states. For example, how many days am I going to consider when prediction tomorrows weather.<br>
**The markov chain**- Its a chain of random variables that follow the markov assumption. By construting the transition model we can come up with the markov chain. (ps: I did draw this in my notes)

## Hidden Markov Model
It's a type of markov model for a system with hidden states(Whats happening in the real world) that generate an observable state(what the agent discovers). Sometimes the AI has some measurement of the world but has no access to the precide state of the world. For example:
- In measuring a web page user engagement, the hidden state is the user engagement the observable state is the web analytics.<br>

**The sensor markov assumption**-States that the evidence variable only depends on the observable states. Sometimes it can ignore other factors such as personality of the person which is had to predict.<br>
Based on the Hidden Markov model, multiple tasks can be achieved:
- Filtering: given the observation from start to now,calculate the probability distribution for the current state.
- Prediction: given the observation from start to now, calculate the probability distribution for the next state.
- Smoothing: given the observation from start to now, calculate the probability distibuion for the past state.
- Most Likely explanation: give the observation from start to now, calculate the most likely sequence of actions.