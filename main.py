import carla
from Environment import Environment
from Algorithms import Baseline, FIFO, PPO  
import time
from statistics import mean
import matplotlib.pyplot as plt

#// Setup to be reproductible
import torch
import random
import numpy as np

DELTA = 0.05
baseline = False
fifo = False
ppo = True





def main(world):
    
    settings = world.get_settings()
    settings.fixed_delta_seconds = DELTA
    settings.synchronous_mode = True
    world.apply_settings(settings)

    SIMULATION_TIME = 100 
    env = Environment(world, life_time=0)
    env.DrawPointsFor30Sec()
    #Setup the environment
    
    MaxBuffer = 1_000
    max_vehicles = 20

    if baseline:

        success_base = []
        coll_base = []
        waitTime_base = []
        speed_list_max_base = []
        speed_list_mean_base = []
        for i in range(20):
            #// Set random seeds to be different in each run
            torch.manual_seed(i)
            random.seed(i)
            np.random.seed(i)

            algorithm = Baseline(world, SIMULATION_TIME, env, DELTA=DELTA, max_vehicles=max_vehicles)
            success_count, collision_count, waitTime, speed_list = algorithm.simulation()
            success_base.append(success_count)
            coll_base.append(collision_count)
            waitTime_base.append(waitTime)
            speed_list_max_base.append(max(speed_list))
            speed_list_mean_base.append(mean(speed_list))

        all_data = [success_base, coll_base, speed_list_max_base, speed_list_mean_base]
        titles = ["Succesful journey", "Collision number", "Max speed", "Average speed"]
        
        fig,axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, d, title in zip(axes.flatten(), all_data, titles):
            ax.hist(d, bins=10)
            ax.set_title(title)
            ax.set_xlabel(title)
            ax.set_ylabel("Simulation")

        plt.tight_layout()
        plt.show()


    if fifo:

        success_base = []
        coll_base = []
        waittime_max_base = []
        waittime_mean_base = []
        for i in range(20):
            #// Set random seeds to be different in each run
            torch.manual_seed(i)
            random.seed(i)
            np.random.seed(i)
            env = Environment(world, life_time=0)
            env.DrawPointsFor30Sec()
            algorithm = FIFO(world, SIMULATION_TIME, env, DELTA=DELTA, max_vehicles=max_vehicles)
            success_count, collision_count, waitTime, speed_list = algorithm.simulation()
            success_base.append(success_count)
            coll_base.append(collision_count)
            waittime_max_base.append(max(waitTime))
            waittime_mean_base.append(mean(waitTime))

        all_data = [success_base, coll_base, waittime_max_base, waittime_mean_base]
        titles = ["Succesful journey", "Collision number", "Max waitTime", "Average waitTime"]
        
        fig,axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, d, title in zip(axes.flatten(), all_data, titles):
            ax.hist(d, bins=10)
            ax.set_title(title)
            ax.set_xlabel(title)
            ax.set_ylabel("Simulation")

        plt.tight_layout()
        plt.show()
        
    if ppo:
        torch.manual_seed(1)
        random.seed(1)
        np.random.seed(1)
        algorithm = PPO(world, SIMULATION_TIME, env,DELTA=DELTA, max_vehicles=max_vehicles, MaxBufferSize=MaxBuffer)
        algorithm.train(500)
    


    #// Just for after easier navigation in the CARLA
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)


if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    main(world)