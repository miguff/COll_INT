import carla
from Environment import Environment
from Algorithms import Baseline, FIFO, PPO  
import time
from statistics import mean

DELTA = 0.05
baseline = False
fifo = False

#Csak személygépjármű legyen

#Maximum résztvevő szám, adott pillanatban maximum x vehet részt
#Első pozvióban láthat a saját x,y pozicióját, sebességét, választott action, (méret)
#maradék helyre az összes többi járműnek ugyanezek a paraméterek

#Minden ágens egymástól független, minden megkap minden információt, minden ágensnek saját tanuló algoritmusa van

#MLP és PPO alap



def main(world):
    
    settings = world.get_settings()
    settings.fixed_delta_seconds = DELTA
    settings.synchronous_mode = True
    world.apply_settings(settings)

    SIMULATION_TIME = 100 #20 sec simulation time
    env = Environment(world, life_time=0)
    env.DrawPointsFor30Sec()
    #Setup the environment
    
    if baseline:
        algorithm = Baseline(world, SIMULATION_TIME, env, DELTA=DELTA, max_vehicles=1)
    if fifo:
        algorithm = FIFO(world, SIMULATION_TIME, env, DELTA=DELTA, max_vehicles=2)
    

    start = time.time()
    algorithm = PPO(world, SIMULATION_TIME, env,DELTA=DELTA, max_vehicles=7, MaxBufferSize=300)

    # #// Do some simulations
    algorithm.train(10)

    # success_count, collision_count, waitTime, speed_list = algorithm.simulation()
    # end = time.time()
    # print("Running Time")
    # print(end - start)

    # print("-----------------------------------------------------")
    # print(f"Summary of simulation:")
    # print(f"Running time: {SIMULATION_TIME} sec")
    # print(f"Number of Successful journeys: {success_count}")
    # print(f"Number of Collisions: {collision_count}")
    # print(f"Wait time: {waitTime} sec")
    # print(f"Average speed: {mean(speed_list)}")
    # print(f"Max speed in simulation: {max(speed_list)}")
    # print("-----------------------------------------------------")

    # #// Just for after easier navigation in the CARLA
    # settings = world.get_settings()
    # settings.synchronous_mode = False
    # world.apply_settings(settings)


if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    main(world)