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
- **Implication elimination** 
```bash
    α <---> β
 -------------
      (α-->β) ∧(β-->α)
```
- **De morgans law**
```bash
(α V β)'
--------
α' ∧ β'
```
- **Distributive law**
```bash
  α ∧ (β V δ)
-----------------
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

### 3. Resolution
Resolution is just a fancy method or rather improved method(excuse me)  of inference. This is another method that is used to prove whether KB ⊨ α. <br>
It relies on complementary literals. In  a fancy way, it's a powerful inference rul that states that **if one or two atomic propositions in an OR proposition is false, the other has to be true**. Tak the example

```bash
P V Q
    P'
---------
    Q
```

What can we use to generate new sentences using inference by resolution? Well that's where clauses come in.<br>
Clauses are basically disjuction of literals(if this is greek search it out). Thus inference algorithms locate complementary literals to generate new knowledge.<br>
Take this example:
```bash
P V Q & Q V R
then:
P' v R
------
Q V R
```
Clauses enable us to convert and logic statement into a conjuctive Normal Form(CNF). If this is greek to search it out its infact easy.<br>
Take this example:
```bash
(A v B) ∧ (D V E) ∧ (F V G)
(clause)∧ (clause)∧ (clause)
```
If you see above CNF can be defined as disjunction of clauses
So what are the steps for converting a logical statement into CNF:
- Eliminate biconditionals
- Eliminate implications
- Use de morgans law where necessary 
- Use distributive law where necessary
At the point of converting into CNF it maybe necessary to factor out some literals because of duplication.<br>
Some times an empty clause can occur when we are resolving literlas and its negation eg

```bash
P
then:
    P'
-------
    ()-this is an empty clause
```

So what is the steps for inference by resolution?
- To determine whether KB ⊨α
    - Convert (KB ∧  α') into CNF 
    - Keep checking if we can use resolution to produce new clauses
    - If we produce an empty clause then KB ⊨α
    - Else no entailment

Example:
```bash
 Does  (A V B) ∧ ( B' V C) ∧ (C')  ⊨ α
We pick 2 clauses and resolve . I pick ( B' V C) ∧ (C')
( B' V C) ∧ (C')
----------------
        B'- conclusion
Then we set B' Aside with
We pick 2 clauses and resolve again. I pick (A V B) and B'
(A V B) ∧ B'
-----------
    A
Now our new set of clauses are  (A') (B) (A)
We resolve A
we get:
    A'
----------
    ()
Then we conclude KB ⊨α

```

### 4. First order logic
Finally the last one. First order logic allows us to express statements more succinctly that propositional logic. First order logic uses 2 types of symbols:
- Constants symbols:This basically represents objects like a persons name or name of a house
- Predicate symbols: This is a function actually that takes an argument and returns true or false. Example:
```bash
Person(Azhar) => Means Azhar is a person
House(Castle) => Means Castle is a house.
BelongsTo(Castle,Azhar)=> Castle belongs to Azhar
```
There are 2 types of First Order Logic:
#### Universal Quantification
Fancy way of saying the word "for all" (∀).<br>
Example:
```bash
∀x Belongto(x,Azhar) --> - BelongsTo(x,Takoy)

```
This means: "For all x that belong to Azhar it doesnt belong to Takoy".

## Existential Quantification
Fancy way of saying the word "there exists" ∃. <br>
Example:
```bash
∃x House(x) ∧ Belongs(Castle,x)
```
This means:"There exists a house x and the house x belongs to a castle and its a house". In short: "Castle belongs to a house".

Finally, thats all is about to know in **Knowledge Engineering**!Byeeeee


