# Knowledge
As human beings we reason based on existing knowledge and we are able to pontificate upon at length to draw conclusions.
knowledge based Agents in AI are those agents that reason by operating on internal representation of knowledge.
Key terms:
- A **sentence** is an assertion about the world in a knowledge representation language.
- A **Proposition logic** are statements about the world that can either be true or false.
- A **Model** its an agent that is used to assign truth value to every proposition. Important thing to note is that the number of models depends on the number of propostions,if we have n propositions then we 2^n models eg.
```bash
P:It's raining & Q:It's Tuesday
Model: {P:True, Q:False}
```
- **Knowledge Base(KB)** is a set of sentences known by the agent. We use the KB to build conclusions about something.
- **Entailment** is a simple way of saying if let's say subject A is true the it means Subject B is true. Technically, if α entails β then it means, in any world where α is true then β is also true. e.g
```bash
α:Its Tuesday in January
β:It's January
```
- **Inference** is the process of drawing conclusions from the knowledge base. There are multiple ways of drawing inferences. This is the backbone of this topic. Some of the ways to draw inferences include:
    - Model checking
    - Inference ruling
    - Resolution
    - First Order logic
I'll be pontificating upon this methods after the logical connectives below



## Logical connectives
Logical connectives are used to combine propositions to form new propositions.
Key terms:
- **Negation** is a logical connective that takes a proposition as input and produces a proposition
- **Conjunction** is a logical connective that takes two propositions as input and produces a proposition
- **Disjunction** is a logical connective that takes two propositions as input and produces a proposition
- **Implication** is a logical connective that takes two propositions as input and produces a proposition
- **Equivalence** is a logical connective that takes two propositions as input and produces a proposition
- **Biconditional** is a logical connective that takes two propositions as input and produces
- **Tautology** is a proposition that is always true

## How AI gets its knowledge
### 1. Model checking
In model checking the AI enumarates over all possible models and checks where if any of the models where the Knowledge base it true then KB ⊨(entails) α. Don't worry about the symbols they'll be easy to understand as we go along. <br>
In model checking, we try to conclude that α is true based on our knowledge base.Consider this  image below:

![WhatsApp Image 2024-10-10 at 14 37 28](https://github.com/user-attachments/assets/8fe63c4f-2703-4722-896d-2a963aaa45ad)

Try understanding the proposition logic in this picture they you'll get to know that there's only 1 possibility where KB is true.
This leads us to something we call Knowledge engineering.
####    Knowledge Engineering
This is the process of figuring out how to represent propositionas and logics in AI. It deals with capturing human knowledge and putting it into computers

### 2. Inference Rules
Model checking is not an efficient algorithm because it has to consider  every possible model before making a conclusion. Consider the image above, you'll see that we had to enumerate over all combinations to know the conclusion.<br>
Inference rules are a set of rules that are used to draw conclusions from the knowledge base. <br>
Inference rules are usually reprented using a horizontal line.
```bash
{premesis}
-----------
{conclusion}
```
Consider the example:<br>
"If its raining, Azhar is inside the house"<br>
The premisis is: "Its raining"<br>
The conclusion is:"Azhar is inside"<br>
```bash
Its raining
----------------
Azhar is inside
```
In the examples below I'll be using α and β to represent propositional statements.
Some of the inference rules include:
- **And elimiation** which states that if an AND proposition is true, then any atomic proposition within it is true as well
```bash
    α ∧ β
 -------------
      α
```
- **Double negation elimiation**
```bash
    (α')'
 -------------
      α
```
-**Implication elimination** 
```bash
    α <---> β
 -------------
      (α-->β) ∧(β-->α)
```
-**De morgans law**
```bash
(α V β)'
--------
α' ∧ β'
```
-**Distributive law**
```bash
α ∧ (β V δ)
------------
(α ∧ β) V (α ∧ δ)
```
- **Modus ponens**
Modus Ponens is a fundamental inference rule in propositional logic. It is a Latin phrase that translates to "mode that affirms." This rule allows us to draw a conclusion from a conditional statement and a premise that matches the condition.
```bash
α --> β
    α
---------
    β
```

Inferencing can be viewed as a search problem with this properties:
- Initial state = KB
- Action = Inference rules
- Transitional model = New knowledge as a result of inferencing
- Goal test= checking if KB ⊨α
- Path cost= Number of steps to improve