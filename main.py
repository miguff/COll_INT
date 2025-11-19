import carla
from Environment import Environment
from Algorithms import Baseline, FIFO

DELTA = 0.05
baseline = False


def main(world):
    
    settings = world.get_settings()
    settings.fixed_delta_seconds = DELTA
    settings.synchronous_mode = True
    world.apply_settings(settings)

    SIMULATION_TIME = 120 #20 sec simulation time
    env = Environment(world, life_time=10)
    env.DrawPointsFor30Sec()
    #Setup the environment
    
    if baseline:
        algorithm = Baseline(world, SIMULATION_TIME, env, DELTA=DELTA)
 
        
    algorithm = FIFO(world, SIMULATION_TIME, env, DELTA=DELTA, max_vehicles=2)
    algorithm.simulation()
    success_count, collision_count, waitTime = algorithm.simulation()

    print("-----------------------------------------------------")
    print(f"Summary of simulation:")
    print(f"Running time: {SIMULATION_TIME} sec")
    print(f"Number of Successful journeys: {success_count}")
    print(f"Number of Collisions: {collision_count}")
    print(f"Wait time: {waitTime} sec")
    print("-----------------------------------------------------")



if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    main(world)