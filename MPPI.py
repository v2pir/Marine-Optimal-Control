import random
import numpy as np
import sys
import time as t

class model:

    #initialize environment
    def __init__(self, Cd, p, voltage, Nc,
                 dim_x, dim_y, dim_z, mass, I):
        '''
        Cd: coefficient of drag
        p: density of environment
        A: cross sectional area of object
        voltage: voltage being passed to thrusters
        Nc: noise coefficient in the environment

        dim_x: dimension of robot in x direction (length)
        dim_y: dimension of robot in y direction (width)
        dim_z: dimension of robot in z direction (depth)
        mass: mass of robot
        I: rotational inertia of robot at center of mass
        '''

        # environment data
        self.Cd = Cd
        self.p = p
        self.voltage = voltage
        self.Nc = Nc

        # submarine specific data
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.mass = mass
        self.I = I

    def create_function(self):
        def create_function_from_points(points):
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            coefficients = np.polyfit(x_values, y_values, 7)  # 6th degree polynomial for 7 points

            def function(x):
                return np.polyval(coefficients, x)

            return function

        # Example points
        points = [(1892, 5.18), (1828, 3.96), (1729, 2.12), (1500, 0), (1388, 0.58), (1276, 1.7), (1180, 2.99), (1108, 4.02), (1292, 1.51), (1228, 2.35), (1756, 2.73)]
        my_function = create_function_from_points(points)
        return my_function

    def drag(self, v, A):
        '''
        calculates drag force in the environment given environment parameters (defined when class initialized)
        '''
        return self.Cd * self.p * (v**2)/2 * A #drag formula

    def thrust_force(self, pwms):
        '''
        calculates the thrust force based on voltage and current --> t200 bluerobotics thrusters
        voltage = 16
        current depends on number of thrusters `n` --> 120/n --> plan is to have 8 thrusters so current = 15
        '''
        thrust_array = []
        func = self.create_function()

        for pwm in pwms:
            force = func(pwm)
            if pwm < 1500:
                force *= -1
            force = 9.80665 * force # convert to newtons
            thrust_array.append(force)
        return np.array(thrust_array)
    
    def pythag(self, a, b):
        return np.sqrt(a**2 + b**2)
    
    def random_noise(self):
        '''
        simulate random environment noise given thrust
        '''
        noise = np.random.randint(-10, 10) #calculate noise with 10% buffer
        return noise

    def movement(self, thrusts, time, dvl_data):
        # horizontals are 4, 5, 6, 7
        # verticals are 0, 1, 2, 3
        x_v, y_v, z_v = dvl_data
        thrust_0, thrust_1, thrust_2, thrust_3, thrust_4, thrust_5, thrust_6, thrust_7 = thrusts

        x_force = thrust_4 * np.sin(np.pi/4) + thrust_5 * np.sin(np.pi/4) - thrust_6 * np.sin(np.pi/4) - thrust_7 * np.sin(np.pi/4) - self.drag(abs(x_v), self.dim_y*self.dim_z) + self.random_noise()
        y_force = -thrust_4 * np.cos(np.pi/4) + thrust_5 * np.cos(np.pi/4) - thrust_6 * np.cos(np.pi/4) + thrust_7 * np.cos(np.pi/4) - self.drag(abs(y_v), self.dim_x*self.dim_z) + self.random_noise()
        z_force = thrust_0 - thrust_1 + thrust_2 - thrust_3 - self.drag(abs(z_v), self.dim_x*self.dim_y) + self.random_noise()

        x_dist = 0.5 * (x_force/self.mass) * np.power(time, 2) + x_v * time
        y_dist = 0.5 * (y_force/self.mass) * np.power(time, 2) + y_v * time
        z_dist = 0.5 * (z_force/self.mass) * np.power(time, 2) + z_v * time

        x_v += x_force/self.mass * time
        y_v += y_force/self.mass * time
        z_v += z_force/self.mass * time

        return np.array([x_dist, y_dist, z_dist]), np.array([0, 0, 0])
    
    def get_state_dim(self):
        return 3
    
    def get_control_dim(self):
        return 8

class MPPI:

    def __init__(self, horizon, num_rollouts, model, time_step, lamb):
        self.horizon = horizon
        self.num_rollouts = num_rollouts
        self.model = model
        self.state_dim = model.get_state_dim()
        self.control_dim = model.get_control_dim()
        self.control_mu = np.zeros(self.control_dim)
        self.time_step = time_step
        self.lamb = lamb
        self.vel = np.zeros(self.state_dim)

    def integral(self, dvl_data, time:np.array):
        '''
        Output: float
        '''
        dist = 0
        for i in range(len(time) - 1):
            width = time[i + 1] - time[i]
            height = (dvl_data[i] + dvl_data[i + 1]) / 2 #midpoint reimann sum
            dist += width * height
        return dist #returns position since dvl_data gives velocity at some time

    def get_current_pos(self, dvl_data, setpoint, time):
        '''
        Output: np.ndarray of shape (self.state_dim)
        '''
        return setpoint - self.integral(dvl_data, time)

    def sample_controls(self, max_power):
        '''
        Input: max power; float
        Output: np.ndarray of shape (self.num_rollouts, self.horizon, self.control_dim) / (N, H, U)
        '''
        return np.random.uniform(1500-400*max_power,1500+400*max_power, (self.num_rollouts,self.horizon,self.control_dim))
    
    def next_state(self, cntrl, current):
        thrusts = self.model.thrust_force(cntrl) # calculate the power given to each thruster
        movements, self.vel = self.model.movement(thrusts, self.time_step, self.vel) # model the movement given the thrusts
        return np.add(current, movements)
        
    def rollout(self, x0, U):
        '''
        Input: Initial State, Control (of shape (self.horizon, self.control_dim)
        Output: np.ndarray of shape (self.horizon, self.state_dim)
        '''

        states = []
        for traj in U:
            curr_state = x0
            state_array = []
            for move in traj:
                curr_state = self.next_state(move, curr_state)
                state_array.append(curr_state.tolist())
            states.append(state_array)

        return np.asarray(states)

    def cost(self, states, setpoint):
        '''
        Input: Rollouts of shape (N, H, S)
        Output: Cost of each state of shape (N, H)
        '''

        cost = 0
        for state in range(len(states)):
            cost_pos = np.linalg.norm(setpoint-states[state])
            cost = cost + cost_pos

        return cost

    def optimize(self, initial, setpoint):
        '''
        Input: Initial State
        Output: Optimized control sequence of shape (H, U) 
        '''

        '''with weights'''
        # self.control_mu = np.zeros(self.control_dim)
        # U = self.sample_controls(0.6)
        # states = self.rollout(initial, U)
        # total_cost = 0
        # for state in states:
        #     total_cost += np.exp(self.lamb * -self.cost(state, setpoint))

        # weights = []
        # for trajectory in states:
        #     tr_cost = np.nan_to_num(np.exp(self.lamb * -self.cost(trajectory, setpoint)))
        #     weights.append(np.nan_to_num(tr_cost/total_cost))

        # weights = np.array(weights)

        # for t in range(self.num_rollouts):
        #     self.control_mu += weights[t] * U[t, 0]

        # return self.control_mu

        '''without weights'''
        self.control_mu = np.zeros(self.control_dim)
        U = self.sample_controls(0.6)
        least_cost = sys.maxsize
        states = self.rollout(initial, U)
        for state in range(len(states)):
            cost = self.cost(states[state], setpoint)
            if cost < least_cost:
                least_cost = cost
                self.control_mu = U[state][0]

        return self.control_mu

        # Compute MPPI update with exponential utility np.exp(costs)