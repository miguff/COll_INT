from Algorithms.AlgorithmAbs import Algorithm
from Environment import Environment
import time
import carla
import math
from Agents import BasicAgent
import numpy as np


class FIFO(Algorithm):
    def __init__(self, world, simulation_time, env, spawn_interval = 0.5, max_vehicles = 20, DELTA = 0.05):
        super().__init__(world, simulation_time, env, spawn_interval, max_vehicles, DELTA)
        self.distance_threshold = 6
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
        self.env.spawns_actor(BasicAgent)

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
                    self.env.spawns_actor(BasicAgent)
                last_spawn_time = current_time

            #// Create a list for deletable actors
            keys_to_delete = []

            #// Every tick, move the vehicles
            for actor_id, data in list(actor_dict.items()):
                #// Get the values
                agent: BasicAgent = data["agent"]
                vehicle = data["vehicle"]
                sensor = data["sensor"]
                
                #// This is a PID control step, where it returns wheter it should brake or throttle
                control = agent.run_step()

                #// get the vehicles location, x and y coordinates to measure how far it is from line
                vehicle_pos_x = vehicle.get_transform().location.x
                vehicle_pos_y = vehicle.get_transform().location.y


                #// Here there is a decision step, 
                #// len(self.env.wait_queue) == 0 --> If there are no previous vehicle in the waiting queue, first vehicle to be put into the list
                #// self.env.moving_id == actor_id --> In this scenario, only 1 actor can move through the intersection, so the moving id checks that
                
                #// Calculate distance to line
                distance = 1000
                for id, (A, B, C) in self.env.line_equations.items():
                    calculated_distance = abs(A*vehicle_pos_x+B*vehicle_pos_y+C) / math.sqrt(A*A+B*B)
                    if calculated_distance < distance:
                        distance = calculated_distance

                if len(self.env.wait_queue) == 0 or self.env.moving_id == actor_id or distance > self.distance_threshold or actor_id in self.env.middle_queue or actor_id in self.env.already_through:
                #if distance > self.distance_threshold:
                    #// If it is the car that the recieved control is applied to the vehicle
                    vehicle.apply_control(control)
                else:
                    #// If there are multiple vehicles in the queue or the moving actor is not that vehicle, it stops.
                    #// It will be changed, so that it would be able to move to a given point.
                    vehicle.apply_control(carla.VehicleControl(brake=1))
                
                #// get the vehicles location, x and y coordinates
                vehicle_pos_x = vehicle.get_transform().location.x
                vehicle_pos_y = vehicle.get_transform().location.y
                
                #// Loop through the predifined bounding boxes, to check if the car is inside any four of it
                for i in self.env.bounding_boxes[:-1]:
                    # //Check if car is in Bounding box and not already in the waiting queue
                    if (i[0].x <= vehicle_pos_x and i[0].y <= vehicle_pos_y and
                        i[1].x >= vehicle_pos_x and i[1].y <= vehicle_pos_y and
                        i[2].x >= vehicle_pos_x and i[2].y >= vehicle_pos_y and
                        i[3].x <= vehicle_pos_x and i[3].y >= vehicle_pos_y) and actor_id not in self.env.wait_queue:
                        

                        # // Append every agent that is inside the bounding box
                        self.env.wait_queue.append(actor_id)
                #// If there are no previous car in the than the moving actor will be the current actor
                if len(self.env.wait_queue) == 0:
                    self.env.moving_id = actor_id
                else:
                    #// If there are agents in the queu, that in that case, the first one will be the moving object, the others will wait
                    self.env.moving_id = self.env.wait_queue[0]
                
                
                if actor_id in self.env.wait_queue and actor_id != self.env.moving_id:
                    agent.waitTime += self.DELTA
                
                
                
                #//Middle value if the middle square where the cars passes through. If a car, gets into the middle square,
                #// then steps out, it would give the green light to another vehicles.
                middle_value = self.env.bounding_boxes[-1]


                #// Check if the actor is already in the waiting queue, and 
                #// Not already in the middle queue
                #// And it is inside the bounding box of the middle square
                if actor_id in self.env.wait_queue and actor_id not in self.env.middle_queue and (middle_value[0].x <= vehicle_pos_x and middle_value[0].y <= vehicle_pos_y and
                                                        middle_value[1].x >= vehicle_pos_x and middle_value[1].y <= vehicle_pos_y and
                                                        middle_value[2].x >= vehicle_pos_x and middle_value[2].y >= vehicle_pos_y and
                                                        middle_value[3].x <= vehicle_pos_x and middle_value[3].y >= vehicle_pos_y):
                    #// If it is inside, add it to the middle queue
                    self.env.middle_queue.append(actor_id)
                    

                #// Checks if actor already in the middle queue and
                #// Checks if it is not inside the bounding box anymore
                if actor_id in self.env.middle_queue and (middle_value[0].x >= vehicle_pos_x or middle_value[0].y >= vehicle_pos_y or 
                                                            middle_value[1].x <= vehicle_pos_x or middle_value[1].y >= vehicle_pos_y or
                                                            middle_value[2].x <= vehicle_pos_x or middle_value[2].y <= vehicle_pos_y or
                                                            middle_value[3].x >= vehicle_pos_x or middle_value[3].y <= vehicle_pos_y):
                    #// remove the id from the two queue
                    self.env.wait_queue.remove(actor_id)
                    self.env.middle_queue.remove(actor_id)
                    self.env.already_through.append(actor_id)

                #// Check if the agent is done or collided
                if agent.done() or data["collided"]:
                        if data["collided"]:
                            #// If collided, than print out that fact, and add 0.5 to the collision count
                            #// It is 0.5, because one collision includes 2 cars but just 1 collision.
                            print(f"Vehicle {actor_id} Collided, destroying car")
                            self.collision_count += 0.5
                            if actor_id in self.env.wait_queue:
                                self.env.wait_queue.remove(actor_id)
                            if actor_id in self.env.middle_queue:
                                self.env.middle_queue.remove(actor_id)
                            if actor_id in self.env.already_through:
                                self.env.already_through.remove(actor_id)
                            self.WaitDict[actor_id] = round(agent.waitTime, 2)
                        else:
                            #// If it reached the goal print out that
                            print(f"Vehicle {actor_id} reached its destination, destroying actor.")
                            #// Make the success count bugger
                            self.success_count += 1
                            self.env.already_through.remove(actor_id)
                            self.WaitDict[actor_id] = round(agent.waitTime, 2)
                        #// Delete the vehicles and sensors that are done
                        vehicle.destroy()
                        sensor.destroy()
                        #// Add the actors to the deletable list
                        keys_to_delete.append(actor_id)
                #time.sleep(0.05)
            #// Delete every id from that list, that are done
            for key in keys_to_delete:
                del actor_dict[key]
                if len(actor_dict) == 0:
                    running_sim = False
                    self.world.tick()

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


        #// Average out the waiting times
        self.waitTime = round(np.mean(list(self.WaitDict.values())))
        return (self.success_count, self.collision_count, self.waitTime)