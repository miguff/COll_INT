from Algorithms.AlgorithmAbs import Algorithm
from Environment import Environment

class Baseline(Algorithm):
    def __init__(self, world, simulation_time: int, env: Environment,
                 spawn_interval: float = 0.5, max_vehicles: int = 20, DELTA: float = 0.05):
        super().__init__(world, simulation_time, env, spawn_interval, max_vehicles, DELTA)


    def simulation(self) -> tuple:
        
        actor_dict = self.env.actors_list

        # Initialize time for spawning
        snapshot = self.world.get_snapshot()
        if snapshot is None:
            print("No snapshot received from world, exiting.")
            return
        last_spawn_time = snapshot.timestamp.elapsed_seconds

        running_sim = True
        while running_sim and self.running_simulation_time <= self.simulation_time:
        # Advance simulation (use wait_for_tick() if you're in async mode)
            self.world.tick()
            self.running_simulation_time += self.DELTA
            snapshot = self.world.get_snapshot()
            if snapshot is None:
                continue
            current_time = snapshot.timestamp.elapsed_seconds

            # 1) Spawn new vehicles at fixed time intervals
            if current_time - last_spawn_time >= self.spawn_interval:
                if len(actor_dict) < self.max_vehicles:
                    pass
                    self.env.spawns_actor()
                last_spawn_time = current_time

            # 2) Move existing vehicles and delete ones that reached destination
            keys_to_delete = []

            # list(...) so we don't modify the dict while iterating
            for actor_id, data in list(actor_dict.items()):
                agent = data["agent"]
                vehicle = data["vehicle"]
                sensor = data["sensor"]
                # Let the agent compute the next control
                control = agent.run_step()
                vehicle.apply_control(control)
                # If the agent reached the goal
                if agent.done() or data["collided"]:
                    if data["collided"]:
                        print(f"Vehicle {actor_id} Collided, destroying car")
                        self.collision_count += 0.5
                    else:
                        print(f"Vehicle {actor_id} reached its destination, destroying actor.")
                        self.success_count += 1
                    vehicle.destroy()
                    sensor.destroy()
                    keys_to_delete.append(actor_id)

            # Actually remove them from the dict
            for key in keys_to_delete:
                del actor_dict[key]
                if len(actor_dict) == 0:
                    running_sim = False
                    self.world.tick()

            super().draw_counters(self.world, self.success_count, self.collision_count, life_time=0.1)

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

        # Clear dictionary
        self.env.actors_list.clear()
        self.world.tick()

        return (self.success_count, self.collision_count, self.waitTime)