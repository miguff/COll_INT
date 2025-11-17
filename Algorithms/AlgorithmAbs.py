from abc import ABC, abstractmethod
import carla

class Algorithm(ABC):
    pass

    @abstractmethod
    def simulation():
        raise NotImplementedError("It was not yet implemented")
    
    def draw_counters(self, world, success_count, collision_count, life_time=0.1):
        """
        Draws a floating text in front of the spectator camera with the counters.
        life_time should be small and this function called every tick.
        """
        debug = world.debug
        spectator = world.get_spectator()
        transform = spectator.get_transform()

        # Position text a bit in front of the camera and slightly up
        forward_vec = transform.get_forward_vector()
        loc = transform.location + forward_vec * 2.0  # 2 meters in front
        loc.z += 1.5                                 # 1.5 meters above

        text = f"Success: {success_count}  |  Collisions: {collision_count}"

        debug.draw_string(
            loc,
            text,
            draw_shadow=False,
            color=carla.Color(r=255, g=255, b=255),
            life_time=life_time,
            persistent_lines=False
        )