import scipy.optimize


# Objective function: 50x+80y
# Constraint 1: 5x+2y<=20
# Constraint 2: (-10x)+(-12y)<=90

result=scipy.optimize.linprog(
    [50,80],
    A_ub=[[5,2],[-10,-12]], # Coefficent of x and y
    b_ub=[20,-90] # Constraint of x and y
)

if result.success:
    print(f"X:{round(result.x[0],2)} hours")
    print(f"Y:{round(result.x[1],2)} hours")
else:
    print("No solution found")

