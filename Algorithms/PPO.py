from Algorithms.AlgorithmAbs import Algorithm
import carla
from Agents.PPOagents import PPOAgent
from Environment import Environment
import math
import time
import torch as T

class PPO(Algorithm):
    def __init__(self, world, simulation_time: int , env: Environment, spawn_interval = 0.5, max_vehicles = 20, DELTA = 0.05):
        super().__init__(world, simulation_time, env, spawn_interval, max_vehicles, DELTA)
        self.WaitDict: dict = {}

    def simulation(self):
        actor_dict = self.env.actors_list   
        # Initialize time for spawning
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

            #// Create a list for deletable actors
            keys_to_delete = []
            #// Every tick, move the vehicles

            for actor_id, data in list(actor_dict.items()):
                #// Get the values
                print(actor_id)
                state = [] 
                agent_state = []
                agent: PPOAgent = data["agent"]
                vehicle = data["vehicle"]
                sensor = data["sensor"]


                #// Get the required values for the state, that is an input to the PPO.
                vehicle_pos_x = vehicle.get_transform().location.x
                vehicle_pos_y = vehicle.get_transform().location.y

                #// This is needed, because when a car is spawn it starts to fall down, and it is not needed, to track its parameters
                if round(abs(vehicle.get_velocity().z),3) > 0:
                    continue
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

                #// This is the PPO step
                control = agent.run_step(state)
                vehicle.apply_control(control)


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

        return 0, 0, 0