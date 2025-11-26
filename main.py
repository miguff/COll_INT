import carla
from Environment import Environment
from Algorithms import Baseline, FIFO, PPO  

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

    SIMULATION_TIME = 20 #20 sec simulation time
    env = Environment(world, life_time=10)
    env.DrawPointsFor30Sec()
    #Setup the environment
    
    if baseline:
        algorithm = Baseline(world, SIMULATION_TIME, env, DELTA=DELTA)
    if fifo:
        algorithm = FIFO(world, SIMULATION_TIME, env, DELTA=DELTA, max_vehicles=2)
    
    algorithm = PPO(world, SIMULATION_TIME, env,DELTA=DELTA, max_vehicles=2)
    
    success_count, collision_count, waitTime = algorithm.simulation()

    print("-----------------------------------------------------")
    print(f"Summary of simulation:")
    print(f"Running time: {SIMULATION_TIME} sec")
    print(f"Number of Successful journeys: {success_count}")
    print(f"Number of Collisions: {collision_count}")
    print(f"Wait time: {waitTime} sec")
    print("-----------------------------------------------------")

    #// Just for after easier navigation in the CARLA
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)


if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    main(world)