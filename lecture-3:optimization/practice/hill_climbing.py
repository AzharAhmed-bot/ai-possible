import random

class Classroom:
    def __init__(self,grid_size,students,disturbance_matrix):
        self.grid_size=grid_size
        self.students=students
        self.disturbance_matrix=disturbance_matrix

    def generate_random_grid(self):
        grid=[]
        random.shuffle(self.students)
        index=0
        for _ in range(self.grid_size):
            grid.append(self.students[index:index+self.grid_size])
            index+=self.grid_size
        return grid
    
    def get_cost(self,grid):
        cost=0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                student=grid[i][j]
                neighbours=[
                    (i-1,j),(i+j,i),
                    (i,j-1),(i,j+1)
                ]
                for ni,nj in neighbours:
                    if 0<=ni<self.grid_size and 0<=nj<self.grid_size:
                        neighbour=grid[ni][nj] 
                        cost+=self.disturbance_matrix[student-1][neighbour-1]
        return cost
    

grid_size=2
students=[0,1,2,3]
disturbance_score=[
    [0,2,5,3],
    [2,0,6,4],
    [5,6,0,5],
    [3,4,5,0]
]

class4Yellow=Classroom(grid_size,students,disturbance_score)

grid=class4Yellow.generate_random_grid()
print(grid)
cost=class4Yellow.get_cost(grid)
print(cost)


