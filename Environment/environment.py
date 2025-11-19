
import carla
import random
import sys
sys.path.append(r'C:\Users\local_user\Documents\Programozás\SelfDrivingCar\CarlaRun\PythonAPI\carla')
from Agents import BasicAgent
from collections import deque


class Environment(object):

    def __init__(self, world, 
                 center1: carla.Location = carla.Location(x=-51, y=-8, z=0.0),
                 center2: carla.Location = carla.Location(x=-43, y=48, z=0.0),
                 center3: carla.Location = carla.Location(x=-70.5, y=26, z=0.0),
                 center4: carla.Location = carla.Location(x=-23, y=15, z=0.0),
                 life_time: int = 0,
                 start_positions: list = [126, 28, 85, 49],
                 endpositions: dict = {126: [2, 26, 55], 28: [55,2,104], 85: [26, 55, 104], 49: [2, 26, 104]}):
        self.world = world

        #First lane - ID 126 for spawn - Possible end IDs: 2, 26, 55
        self.center1 = center1

        #Second lane - ID 28 for spawn - Possible end IDs: 55, 2, 104
        self.center2 = center2

        #Third lane - ID 85 for spawn - Possible end IDs: 26, 55, 104
        self.center3 = center3

        #Furth Lane - ID 49 for spawn - Possible end IDs: 2, 26, 104
        self.center4 = center4

        color = carla.Color(255, 0, 0)

        self.center5 = carla.Location(x=-47, y=21, z=0.0)
        middle_color = carla.Color(255, 255, 0)

        self.life_time = life_time
        self.actors_list = {}
        self.bounding_boxes = []
        self.line_equations = {}

        self.draw_ground_rectangle(self.center1, width=7, height=15, life_time=10, color=color, intersection_number=1)
        self.draw_ground_rectangle(self.center2, width=7, height=15, life_time=10, color=color, intersection_number=2)
        self.draw_ground_rectangle(self.center3, width=15, height=7, life_time=10, color=color, intersection_number=3)
        self.draw_ground_rectangle(self.center4, width=15, height=7, life_time=10, color=color, intersection_number=4)
        self.draw_ground_rectangle(self.center5, width=25, height=25, life_time=10, color=middle_color)

        self.spawn_points = self.world.get_map().get_spawn_points()
        self.start_positions = start_positions
        self.end_positions = endpositions
        self.number_of_agents = 0
        self.wait_queue = deque()
        self.middle_queue = deque()
        self.already_through = deque()
        self.moving_id = 0
        self.world.tick()
        




    def draw_ground_rectangle(self, center, width, height, z_offset=0.2, life_time=0.0, color: carla.Color = carla.Color(255, 0, 0), intersection_number: int = -1):
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

        self.bounding_boxes.append([p1, p2, p3, p4])
        
        if intersection_number == 1:
            A, B, C = self._line_from_points(p3, p4)
            self.line_equations[intersection_number] = (A, B, C)
        elif intersection_number == 2:
            A, B, C = self._line_from_points(p1, p2)
            self.line_equations[intersection_number] = (A, B, C)
        elif intersection_number == 3:
            A, B, C = self._line_from_points(p2, p3)
            self.line_equations[intersection_number] = (A, B, C)
        elif intersection_number == 4:
            A, B, C = self._line_from_points(p4, p1)
            self.line_equations[intersection_number] = (A, B, C)
        


        self.world.debug.draw_line(p1, p2, thickness=0.2, color=color, life_time=life_time)
        self.world.debug.draw_line(p2, p3, thickness=0.2, color=color, life_time=life_time)
        self.world.debug.draw_line(p3, p4, thickness=0.2, color=color, life_time=life_time)
        self.world.debug.draw_line(p4, p1, thickness=0.2, color=color, life_time=life_time)

    def DrawPointsFor30Sec(self):
        drawn_points = []
        for index, waypoint in enumerate(self.spawn_points):
            # Draw a string with an ID at the location of each spawn point
            point_id = f'ID: {index}'
            point = self.world.debug.draw_string(
                waypoint.location,
                point_id,
                draw_shadow=False,
                color=carla.Color(r=255, g=255, b=255),
                life_time=60,  # Set to 0 to make it persist indefinitely
                persistent_lines=True
            )
            drawn_points.append(point)

    def spawns_actor(self):

        #Spawn to random start and random ends
        startid = random.choice(self.start_positions)
        endid = random.choice(self.end_positions[startid])
        spawn = self.spawn_points[startid]
        end_location = self.spawn_points[endid].location

        vehicle = self.spawn_random_vehicle(spawn)
        if vehicle == None:
            return self.actors_list
        agent = BasicAgent(vehicle, spawn.location, end_location, target_speed=10)
        
        self.number_of_agents += 1
        actor_id = f"{self.number_of_agents}"
        sensor = self._attach_collision_sensor(vehicle, actor_id)

        self.actors_list[actor_id] = {
                "agent": agent,
                "vehicle": vehicle,
                "sensor": sensor,
                "collided": False,
            }

        return self.actors_list
        
    def spawn_random_vehicle(self, spawn_point):
        bp_lib = self.world.get_blueprint_library()
        vehicle_blueprints = bp_lib.filter('vehicle.*')

        # optionally filter out bikes, etc.
        vehicle_blueprints = [bp for bp in vehicle_blueprints
                            if bp.has_attribute("number_of_wheels") 
                            and int(bp.get_attribute("number_of_wheels").as_int()) == 4]

        blueprint = random.choice(vehicle_blueprints)

        
        vehicle = self.world.try_spawn_actor(blueprint, spawn_point)

        if vehicle == None:
            return None

        return vehicle
    

    def _line_from_points(self, p1, p2):
        """
        Calculates the coefficients A, B, and C for the line equation Ax + By = C.
        """
        A = (p1.y - p2.y)
        B = (p2.x - p1.x)
        C = (p1.x * p2.y - p2.x * p1.y)
        return A, B, C


    def _attach_collision_sensor(self, vehicle, actor_id: str):
        bp_lib = self.world.get_blueprint_library()
        sensor_bp = bp_lib.find('sensor.other.collision')

        # Attach to the vehicle (no offset, but you can move it if you want)
        sensor_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.0)
        )
        sensor = self.world.spawn_actor(
            sensor_bp,
            sensor_transform,
            attach_to=vehicle
        )

        # Collision callback: just mark the actor as collided
        def _on_collision(event, env=self, actor_id=actor_id):
            print(f"[COLLISION] actor {actor_id}")
            if actor_id in env.actors_list:
                env.actors_list[actor_id]["collided"] = True

        sensor.listen(_on_collision)
        return sensor