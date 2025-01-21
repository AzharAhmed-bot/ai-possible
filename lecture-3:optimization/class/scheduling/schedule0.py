

VARIABLES=["A","B","C","D","E","F","G"]
CONSTRAINTS=[
    ("A","B"),
    ("A","C"),
    ("B","C"),
    ("A","C"),
    ("B","D"),
    ("B","E"),
    ("C","E"),
    ("C","F"),
    ("D","E"),
    ("E","F"),
    ("E","G")
]



def backTracking(assignment):
    if len(assignment)==len(VARIABLES):
        return assignment
    
    var=select_unassigned_variable(assignment)
    domain=["Monday","Tuesday","Wednesday"]
    for value in domain:
        new_assignment=assignment.copy()
        new_assignment[var]=value
        if consistency(new_assignment):
            result=backTracking(new_assignment)
            if result is not None:
                return result
    
    return None


def select_unassigned_variable(assignment):
    for variable in VARIABLES:
        if variable not in assignment:
            return variable
    return None

def consistency(assignment):
    for (constraint1,constraint2) in CONSTRAINTS:
        if constraint1 not in assignment or constraint2 not in assignment:
            continue
        if assignment[constraint1]==assignment[constraint2]:
            return False
    
    return True


solution=backTracking(dict())
print(solution)