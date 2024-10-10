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
### Model checking
In model checking the AI enumarates over all possible models and checks where if any of the models where the Knowledge base it true then KB ⊨(entails) α. Don't worry about the symbols they'll be easy to understand as we go along. <br>
In model checking, we try to conclude that α is true based on our knowledge base.Consider this  image below: