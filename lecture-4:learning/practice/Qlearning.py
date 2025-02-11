from collections import defaultdict
import numpy as np
import random

class TrafficLightQLearner:
    def __init__(self):
        # N,S,E,W
        self.directions=4 
        #Action to be taken
        self.actions=["NS-green","EW-green"] 
         # Default q-table for unseen state-action pair
        self.q_table=defaultdict(lambda:[0.0,0.0])
        # Hyper parameters
        self.alpha=0.1
        self.gamma=0.9
        self.epsilon=0.2
        self.epsilon_decay=0.995

        # Traffic parameters
        self.arrival_prob=0.3
        self.max_cars_depart=4


        # Threshold for discretizing traffic states for example:
        # 0 means no cars
        # 1 means 1-3 cars
        # 2 means 4-6 cars
        # 3 means 7+ cars
        self.car_thresholds=[0,3,6]
    
    def discretize_state(self,cars):
        return tuple(np.digitize(c, self.car_thresholds) for c in cars)
    
    def choose_action(self,state):
        # If no known action pick randon
        if random.random() < self.epsilon:
            return random.choice([0,1])
        else:
            return np.argmax(self.q_table[state])
    
    def update_epsilon(self):
        self.epsilon= max(0.01,self.epsilon*self.epsilon_decay)
    
    def simulate_traffic(self,cars,action):
        new_cars=cars.copy()
        if action==0:
            new_cars[0]=max(0,new_cars[0]-self.max_cars_depart)
            new_cars[1]=max(0,new_cars[1]-self.max_cars_depart)
        else:
            new_cars[2]=max(0,new_cars[2]-self.max_cars_depart)
            new_cars[3]=max(0,new_cars[3]-self.max_cars_depart)
        
        # Simulate random arrival
        for i in range(4):
            if random.random() < self.arrival_prob:
                new_cars[i]+=1
        
        return new_cars

    def calculate_reward(self,cars):
        return -sum(cars)

    def train(self, episodes=1000, steps_per_episode=60):
        for episode in range(episodes):
            cars=[0,0,0,0]
            total_reward=0
            for step in range(steps_per_episode):
                state=self.discretize_state(cars)
                action=self.choose_action(state)
                new_cars=self.simulate_traffic(cars,action)
                reward=self.calculate_reward(new_cars)
                total_reward+=reward

                # Get future rewards
                new_state=self.discretize_state(new_cars)
                old_value=self.q_table[state][action]
                future_reward=max(self.q_table[new_state])
                # Update q-table
                new_value=old_value + self.alpha * (reward + self.gamma * future_reward - old_value)
                self.q_table[state][action]=new_value

                cars=new_cars

            self.update_epsilon()

            if(episode+1)%100==0:
                print(f"Episode {episode+1} | Total reward: {total_reward} | Epsilon: {self.epsilon:.3f}")
            

    def test(self, steps=60):
        cars = [0, 0, 0, 0]
        print("\nTesting trained agent:")
        print("Step | Cars (N, S, E, W) | Action | Total Cars")

        for step in range(steps):
            state = self.discretize_state(cars)
            action = np.argmax(self.q_table[state])  
            cars = self.simulate_traffic(cars, action)

            print(f"{step+1:4} | {cars} | {self.actions[action]:8} | {sum(cars):3}")


t1_agent=TrafficLightQLearner()
t1_agent.train(1000)
t1_agent.test(60)









