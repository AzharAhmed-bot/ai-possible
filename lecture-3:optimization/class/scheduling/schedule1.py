from constraint import *

problem=Problem()

course=["A","B","C","D","E","F","G"]
exam_day=["Monday","Tuesday","Wednesday"]
CONSTRAINTS = [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("B", "D"),
    ("B", "E"),
    ("C", "E"),
    ("C", "F"),
    ("D", "E"),
    ("E", "F"),
    ("E", "G"),
    ("F", "G")
]

problem.addVariables(course,exam_day)

for (x,y) in CONSTRAINTS:
    problem.addConstraint(lambda x,y:x!=y,(x,y))

for solution in problem.getSolutions():
    print(solution) 



