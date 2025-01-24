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
                    (i-1,j),(i+1,i),
                    (i,j-1),(i,j+1)
                ]
                for ni,nj in neighbours:
                    if 0<=ni<self.grid_size and 0<=nj<self.grid_size:
                        neighbour=grid[ni][nj] 
                        cost+=self.disturbance_matrix[student-1][neighbour-1]
        return cost
    
    def get_neighbours(self, grid):
        neighbours = []
        positions=[
            (i,j) for i in range(self.grid_size) for j in range(self.grid_size)
        ]
        for idx1,(i1,j1) in enumerate(positions):
            for i2,j2 in positions[idx1+1:]:
                new_grid=grid.copy()
                new_grid[i1][j1],new_grid[i2][j2]=new_grid[i2][j2],new_grid[i1][j1]
                neighbours.append(new_grid)
            
        return neighbours
    
    def hill_climbing(self):
        current_grid=self.generate_random_grid()
        current_cost=self.get_cost(current_grid)

        while True:
            neighbours=self.get_neighbours(current_grid)
            best_neighbour=None
            best_cost=float('inf')
            for neighbour in neighbours:
                neighbour_cost=self.get_cost(neighbour)
                if neighbour_cost < current_cost:
                    best_neighbour=neighbour
                    best_cost=neighbour_cost
            # If no better neighbour is found, stop
            if best_cost >= current_cost:
                return current_grid, current_cost
                
            current_grid=best_neighbour
            current_cost=best_cost   
        
    def random_restart(self,maximum):
        best_cost=None
        best_grid=None

        for i in range(maximum):
            final_grid,final_cost=self.hill_climbing()
            if best_cost is None or final_cost < best_cost:
                best_cost=final_cost
                best_grid=final_grid
            else:
                continue

        return best_grid,best_cost

grid_size=2
students=[0,1,2,3]
disturbance_score=[
    [0,2,5,3],
    [2,0,6,4],
    [5,6,0,5],
    [3,4,5,0]
]

class4Yellow=Classroom(grid_size,students,disturbance_score)
final_grid,final_cost=class4Yellow.hill_climbing()
best_grid,best_cost=class4Yellow.random_restart(10)

print("Final seating arrangement of Hill climbing")
for row in final_grid:
    print(row)
print(f"Final cost: {final_cost}")

print("Final seating arrangement of Random Restart")
for row in best_grid:
    print(row)
print(f"Final cost: {best_cost}")

