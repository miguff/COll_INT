from Algorithms.AlgorithmAbs import Algorithm
import carla
from Agents.PPOagents import PPOAgent
from Environment import Environment
import math
import time
import torch as T
from Agents import ActorNetwork, CriticNetwork, CarEncoder
device = T.device("cuda" if T.cuda.is_available() else "cpu")
import numpy as np
import torch.nn.functional as F
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
import os
import datetime



class PPOOptimizer():
    def __init__(self,
                clip_epsilon: float = 0.2,
                gamma: float = 0.99,
                lmbda: float = 0.9,
                entropy_eps: float = 1e-4,
                embed_dim: int = 64,
                car_feature_dim: int = 5,
                epochs: int = 2,
                batch_size: int = 64,
                value_coef: float = 0.5,
                entropy_coef: float = 0.0,
                policy_lr: float = 3e-4,
                value_lr: float = 1e-3):

        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.epochs = epochs
        self.batch_size = batch_size

        #// To store the values
        self.buffer = {
            "states" : [],
            "actions" : [],
            "logprobs" : [],
            "values" : [],
            "rewards" : [],
            "dones" : [],
            "next_values" : []
        }
        
        self.encoder = CarEncoder(in_dim=car_feature_dim,
                                  hidden_dim=64,
                                  embed_dim=embed_dim).to(device)
        self.actor = ActorNetwork(input_dims=embed_dim, n_actions=1).to(device)
        self.critic = CriticNetwork(input_dims=embed_dim).to(device)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef


        self.actor_opt = T.optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.critic_opt = T.optim.Adam(self.critic.parameters(), lr=value_lr)


    def _encode_state(self, state):
        state = T.tensor(state, dtype=T.float32, device=device)
        state = state.unsqueeze(0)
        with T.no_grad():
            emb = self.encoder(state)
        return emb
    
    def choose_action(self, state):
        emb = self._encode_state(state)


        #// Add the current state to the buffer
        self.buffer["states"].append(emb)


        with T.no_grad():
            action, logprob = self.actor.act(emb)
            value = self.critic(emb).item()
        value = T.tensor(value)
        self.buffer["actions"].append(action)
        self.buffer["logprobs"].append(logprob)
        self.buffer["values"].append(value)
        return action
    
    def next_value(self, state):
        emb = self._encode_state(state)
        #value = self.critic(emb).item()
        value = self.critic(emb)
        value = T.tensor(value)
        self.buffer["next_values"].append(value)


    def _compute_gae(self, rewards, dones, values, next_values):
        """
        Compute GAE-Lambda advantages and returns.
        rewards: (T,)
        dones: (T,)  boolean or 0/1
        values: (T,)
        next_values: (T,) value at next state, for each step
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lmbda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns


    def update(self):
        states = self.buffer["states"]
        actions = self.buffer["actions"]
        old_logprobs = self.buffer["logprobs"]
        rewards = self.buffer["rewards"]
        sum_rewards = T.stack(rewards, dim=0).sum(dim=0).sum(dim=0)
        dones = self.buffer["dones"]
        values = self.buffer["values"]
        next_values = self.buffer["next_values"]


        advantages, returns = self._compute_gae(rewards, dones, values, next_values)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages_t = T.tensor(advantages, dtype=T.float32, device=device)
        returns_t = T.tensor(returns, dtype=T.float32, device=device)
        states_t = T.stack(states)
        actions_t = T.stack(actions)
        old_logprobs_t = T.stack(old_logprobs)

        dataset_size = len(states)
        indices = np.arange(dataset_size)


        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_logprobs = old_logprobs_t[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]
                # print("batch_states")
                # print(b_states)
                new_logprobs, entropy = self.actor.evaluate_actions(b_states, b_actions)
                values_pred = self.critic(b_states)
                

                ratio = T.exp(new_logprobs - b_old_logprobs)

                surr1 = ratio * b_advantages
                surr2 = T.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                policy_loss = -T.min(surr1, surr2).mean()


                value_loss = F.mse_loss(values_pred, b_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.actor.zero_grad()
                self.critic.zero_grad()
                loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()


        self.buffer = {
            "states" : [],
            "actions" : [],
            "logprobs" : [],
            "values" : [],
            "rewards" : [],
            "dones" : [],
            "next_values" : []
        }

        return sum_rewards.item()

class PPO(Algorithm):
    def __init__(self, world, simulation_time: int,
                 env: Environment,
                 spawn_interval = 0.5,
                 max_vehicles = 20,
                 DELTA = 0.05,
                 clip_epsilon: float = 0.2,
                 gamma: float = 0.99,
                 lmbda: float = 0.9,
                 entropy_eps: float = 1e-4,
                 MaxBufferSize: int = 100,
                 logdir: str = "logs/simulations/"):
        super().__init__(world, simulation_time, env, spawn_interval, max_vehicles, DELTA)
        self.WaitDict: dict = {}

        self.CentralPPO = PPOOptimizer(clip_epsilon, gamma, lmbda, entropy_eps)
        self.MaxBufferSize = MaxBufferSize
        self.speed_list = []
        self.simulation_reward = 0
        self.log_dir = logdir

    def train(self, Number_of_Simulations):
        now = datetime.datetime.now()
        comment = f'PPO_{now.strftime("%Y-%m-%d %H_%M_%S")}_BufferSize_{self.MaxBufferSize}_Vehicles_{self.max_vehicles}'
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, comment), comment=comment)

        for i in range(Number_of_Simulations):
            print(f"{i+1} Simulation")
            self.success_count = 0
            self.collision_count = 0
            self.speed_list = []
            success_count, collision_count, waitTime= self.simulation()
            avg_speed = mean(self.speed_list)
            max_speed = max(self.speed_list)

            print("-----------------------------------------------------")
            print(f"Summary of simulation:")
            print(f"Running time: {self.simulation_time} sec")
            print(f"Number of Successful journeys: {success_count}")
            print(f"Number of Collisions: {collision_count}")
            print(f"Wait time: {waitTime} sec")
            print(f"Average speed: {avg_speed}")
            print(f"Max speed in simulation: {max_speed}")
            print(f"Simulation reward: {self.simulation_reward}")
            print("-----------------------------------------------------")


            step = i
            writer.add_scalar("success_count", success_count, step)
            writer.add_scalar("collision_count", collision_count, step)
            writer.add_scalar("average_speed", avg_speed, step)
            writer.add_scalar("max_speed", max_speed, step)
            writer.add_scalar("simulation_reward", self.simulation_reward, step)
            writer.add_histogram("speed_histogram", T.tensor(self.speed_list, dtype=T.float32), step)

        writer.close()

    def simulation(self):
        self.simulation_reward = 0
        self.running_simulation_time = 0
        actor_dict = self.env.actors_list   
        #// Initialize time for spawning
        snapshot = self.world.get_snapshot()
        if snapshot is None:
            print("No snapshot received from world, exiting.")
            return
        last_spawn_time = snapshot.timestamp.elapsed_seconds
        #// Spawn a starting actor
        self.env.spawns_actor(PPOAgent)
        #// Run simulation
        running_sim = True
        while running_sim and self.running_simulation_time <= self.simulation_time:
            #// move the world
            self.world.tick()

            #// Add the simulation time
            self.running_simulation_time += self.DELTA
            
            #// Get a snaphot of the word, to compare with the previous timestamp, needed to decide wheter to spanw a new vehicle or not
            snapshot = self.world.get_snapshot()
            if snapshot is None:
                continue
            current_time = snapshot.timestamp.elapsed_seconds


            #// Spawn new vehicles at fixed time intervals
            if current_time - last_spawn_time >= self.spawn_interval:
                #// Check wheter the number of agents in the environment is less than the max number
                #// if yes, then create a new
                if len(actor_dict) < self.max_vehicles:
                    pass
                    self.env.spawns_actor(PPOAgent)
                last_spawn_time = current_time


            #// Stepping Phase
            for actor_id, data in list(actor_dict.items()):
                #// Get the values
                agent: PPOAgent = data["agent"]
                vehicle = data["vehicle"]
                sensor = data["sensor"]


                #// Get the current state of the agent
                current_state = self.get_state(vehicle, agent, actor_dict, actor_id)
                if len(current_state) == 0:
                    continue

                #// This is the PPO step
                data["last_route_index"] = agent._route_index
                action = self.CentralPPO.choose_action(current_state)
                control = agent.run_step(action)
                vehicle.apply_control(control)
                #// It is needed, becuase it will be stored to calucate reward from it
                data["last_control"] = control
                data["new_route_index"] = agent._route_index

            #// Giving Reward Phase
            for actor_id, data in list(actor_dict.items()):
                reward = 0
                agent: PPOAgent = data["agent"]
                vehicle = data["vehicle"]
                sensor = data["sensor"]

                #// Get the next state of the agent
                next_state = self.get_state(vehicle, agent, actor_dict, actor_id)
                if len(next_state) == 0:
                    continue
                self.CentralPPO.next_value(next_state)


                #// Check if the agent is done or collided
                if agent.done() or data["collided"]:
                        if data["collided"]:
                            #// If collided, than print out that fact, and add 0.5 to the collision count
                            #// It is 0.5, because one collision includes 2 cars but just 1 collision.
                            self.collision_count += 0.5
                            reward -= 1
                            self.CentralPPO.buffer["dones"].append(T.tensor(1))
                        else:
                            #// If it reached the goal print out that
                            #print(f"Vehicle {actor_id} reached its destination, destroying actor.")
                            reward += 1
                            #// Make the success count bugger
                            self.success_count += 1
                            #self.env.already_through.remove(actor_id)
                            #self.WaitDict[actor_id] = round(agent.waitTime, 2)
                            self.CentralPPO.buffer["dones"].append(T.tensor(1))
                        #// Delete the vehicles and sensors that are done
                        vehicle.destroy()
                        sensor.destroy()
                        #// Add the actors to the deletable list
                        #keys_to_delete.append(actor_id)
                        reward = T.tensor(reward)
                        self.CentralPPO.buffer["rewards"].append(reward)
                        #// Delete every id from that list, that are done
                        del actor_dict[actor_id]
                        continue
                else:
                    reward -= 0.02
                    self.CentralPPO.buffer["dones"].append(T.tensor(0))                

                #// Give reward based on Velocity. If it is between 25 and 30 give it 1, greater than 30 give it -1, else 0
                #// If speed equal 0, agent the throttle was positive, than +1.5 point
                vel = vehicle.get_velocity()
                vehicle_speed = math.sqrt(vel.x ** 2 + vel.y ** 2)
                self.speed_list.append(vehicle_speed)
                control = data["last_control"]
                if vehicle_speed > agent.max_speed:
                    reward -= 1
                # elif agent.max_speed - 2 <= vehicle_speed <= agent.max_speed:
                #     reward += 0.5

                #// Maybe we need a reward for distance to other vehicles
                #// Maybe reward for reaching the next waypoint
                if data["new_route_index"] > data["last_route_index"]:
                    print("Bigger, give points")
                    reward += 0.03


                # if (vehicle_speed < agent.max_speed - 2) and (control.throttle > control.brake):
                #     #// Try to connect it to the speed, of the car. To Encourage going faster
                #     reward += 0.02 * vehicle_speed
                self.simulation_reward += reward
                reward = T.tensor(reward)
                self.CentralPPO.buffer["rewards"].append(reward)
                
            

            #// Check if we should train or just rollout
            if len(self.CentralPPO.buffer["next_values"]) > self.MaxBufferSize:
                self.CentralPPO.update()
            
        #// When simulation ends, remove every vehicle and sensor
        print("Cleaning up remaining actors...")
        for actor_id, data in list(actor_dict.items()):
            vehicle = data["vehicle"]
            sensor = data["sensor"]
            try:
                if sensor and sensor.is_alive:
                    sensor.stop()     # stop listening
                    sensor.destroy()
            except:
                pass
            try:
                if vehicle and vehicle.is_alive:
                    vehicle.destroy()
            except:
                pass


            self.env.actors_list.clear()
            self.world.tick()
        #// Return the success, failure and wait time counts.

        return self.success_count, self.collision_count, 0
    

    def get_state(self, vehicle, agent: PPOAgent, actor_dict: dict, actor_id: int):
        
        state = [] 
        agent_state = []

        #// Get the required values for the state, that is an input to the PPO.
        vehicle_pos_x = vehicle.get_transform().location.x
        vehicle_pos_y = vehicle.get_transform().location.y

        #// This is needed, because when a car is spawn it starts to fall down, and it is not needed, to track its parameters
        if round(abs(vehicle.get_velocity().z),3) > 0:
            return state
        vel = vehicle.get_velocity()
        vehicle_speed = (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2))/3.6
        next_waypoint_x = agent.next_waypoint_x
        next_waypoint_y = agent.next_waypoint_y
        agent_state.append(round(vehicle_pos_x, 3))
        agent_state.append(round(vehicle_pos_y, 3))
        agent_state.append(round(vehicle_speed, 3))
        agent_state.append(round(next_waypoint_x, 3))
        agent_state.append(round(next_waypoint_y, 3))
        state.append(agent_state)
        for another_actor_id, another_data in list(actor_dict.items()):
            another_state = []
            if actor_id == another_actor_id:
                continue
            
            anotheragent: PPOAgent = another_data["agent"]
            anothervehicle = another_data["vehicle"]

            #// This is needed, because when anothercar is spawn it starts to fall down, and it is not needed, to track its parameters
            if round(abs(anothervehicle.get_velocity().z),3) > 0:
                continue
            #// Get the required values for the state, that is an input to the PPO.
            anothervehicle_pos_x = anothervehicle.get_transform().location.x
            anothervehicle_pos_y = anothervehicle.get_transform().location.y
            anothervel = anotheragent._vehicle.get_velocity()
            anothervehicle_speed = (3.6 * math.sqrt(anothervel.x ** 2 + anothervel.y ** 2))/3.6
            anothernext_waypoint_x = anotheragent.next_waypoint_x
            anothernext_waypoint_y = anotheragent.next_waypoint_y
            another_state.append(anothervehicle_pos_x)
            another_state.append(anothervehicle_pos_y)
            another_state.append(anothervehicle_speed)
            another_state.append(anothernext_waypoint_x)
            another_state.append(anothernext_waypoint_y)
            state.append(another_state)

        return state