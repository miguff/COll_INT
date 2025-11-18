from Algorithms.AlgorithmAbs import Algorithm
from Environment import Environment
import time

class FIFO(Algorithm):
    def __init__(self, world, simulation_time, env, spawn_interval = 0.5, max_vehicles = 20, DELTA = 0.05):
        super().__init__(world, simulation_time, env, spawn_interval, max_vehicles, DELTA)



    def simulation(self):
        
        actor_dict = self.env.actors_list
        # Initialize time for spawning
        snapshot = self.world.get_snapshot()
        if snapshot is None:
            print("No snapshot received from world, exiting.")
            return
        last_spawn_time = snapshot.timestamp.elapsed_seconds
        self.env.spawns_actor()
        self.world.tick()
        keys_to_delete = []
        for actor_id, data in list(actor_dict.items()):
            vehicle = data["vehicle"]
            sensor = data["sensor"]
            keys_to_delete.append(actor_id)
            time.sleep(10)
            vehicle.destroy()
            sensor.destroy()

        for key in keys_to_delete:
            del actor_dict[key]
            if len(actor_dict) == 0:
                running_sim = False
                self.world.tick()

        return (self.success_count, self.collision_count, self.waitTime)