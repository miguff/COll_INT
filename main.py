import carla
import time
from collections import deque
import random
import sys
sys.path.append(r'C:\Users\local_user\Documents\Programozás\SelfDrivingCar\CarlaRun\PythonAPI\carla')
from agents.navigation.basic_agent import BasicAgent

DELTA = 0.02


def main(world):

    settings = world.get_settings()
    settings.fixed_delta_seconds = DELTA
    settings.synchronous_mode = False
    settings = world.get_settings()
    world.apply_settings(settings)

    #First lane - ID 126 for spawn - Possible end IDs: 2, 26, 55
    center1 = carla.Location(x=-51, y=-8, z=0.0)

    #Second lane - ID 28 for spawn - Possible end IDs: 55, 2, 104
    center2 = carla.Location(x=-43, y=48, z=0.0)

    #Third lane - ID 85 for spawn - Possible end IDs: 26, 55, 104
    center3 = carla.Location(x=-70.5, y=26, z=0.0)

    #Furth Lane - ID 49 for spawn - Possible end IDs: 2, 26, 104
    center4 = carla.Location(x=-23, y=15, z=0.0)
    draw_ground_rectangle(world, center1, width=7, height=15, life_time=10)
    draw_ground_rectangle(world, center2, width=7, height=15, life_time=10)
    draw_ground_rectangle(world, center3, width=15, height=7, life_time=10)
    draw_ground_rectangle(world, center4, width=15, height=7, life_time=10)

    spawn_points = world.get_map().get_spawn_points()

    DrawPointsFor30Sec(world, spawn_points)

    spawn_pont1 = spawn_points[126]
    end_location = spawn_points[104].location

    
    drive_from_to_ignore_lights(world, spawn_pont1, end_location)



def drive_from_to_ignore_lights(world, spawn_point, end_location):
    # 1) spawn
    vehicle = spawn_random_vehicle(world, spawn_point)

    # 2) create agent
    agent = BasicAgent(vehicle, target_speed=30)   # km/h
    agent.ignore_traffic_lights(True)             # <-- key line
    agent.set_destination(end_location)
    agent.ignore_stop_signs(True)

    # 3) main loop
    while True:                   # assuming synchronous mode
        world.tick()
        control = agent.run_step()
        vehicle.apply_control(control)
        if agent.done():
            vehicle.destroy()
            break

    return vehicle



def spawn_random_vehicle(world, spawn_point):
    bp_lib = world.get_blueprint_library()
    vehicle_blueprints = bp_lib.filter('vehicle.*')

    # optionally filter out bikes, etc.
    vehicle_blueprints = [bp for bp in vehicle_blueprints
                          if not bp.id.endswith('bh.crossbike')]

    blueprint = random.choice(vehicle_blueprints)
    blueprint.set_attribute('role_name', 'ego')

    vehicle = world.try_spawn_actor(blueprint, spawn_point)
    if vehicle is None:
        raise RuntimeError("Could not spawn vehicle at that spawn point")
    return vehicle



def DrawPointsFor30Sec(world, spawn_points):
    drawn_points = []
    for index, waypoint in enumerate(spawn_points):
        # Draw a string with an ID at the location of each spawn point
        point_id = f'ID: {index}'
        point = world.debug.draw_string(
            waypoint.location,
            point_id,
            draw_shadow=False,
            color=carla.Color(r=255, g=255, b=255),
            life_time=30,  # Set to 0 to make it persist indefinitely
            persistent_lines=True
        )
        drawn_points.append(point)


def draw_ground_rectangle(world, center, width, height, z_offset=0.2, life_time=0.0):
    """
    Draws a visible rectangle on the ground using debug lines.
    Arguments:
      - center: carla.Location
      - width, height: rectangle size in meters
      - z_offset: height above ground
      - life_time: how long the rectangle stays (0 = forever)
    """

    # Rectangle corners relative to center
    half_w = width / 2
    half_h = height / 2

    p1 = carla.Location(center.x - half_w, center.y - half_h, center.z + z_offset)
    p2 = carla.Location(center.x + half_w, center.y - half_h, center.z + z_offset)
    p3 = carla.Location(center.x + half_w, center.y + half_h, center.z + z_offset)
    p4 = carla.Location(center.x - half_w, center.y + half_h, center.z + z_offset)

    color = carla.Color(255, 0, 0)  # red lines

    world.debug.draw_line(p1, p2, thickness=0.2, color=color, life_time=life_time)
    world.debug.draw_line(p2, p3, thickness=0.2, color=color, life_time=life_time)
    world.debug.draw_line(p3, p4, thickness=0.2, color=color, life_time=life_time)
    world.debug.draw_line(p4, p1, thickness=0.2, color=color, life_time=life_time)


if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    tm = client.get_trafficmanager()
    world = client.get_world()
    actors = world.get_actors()
    main(world)
