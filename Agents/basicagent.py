import sys
sys.path.append(r'C:\Users\local_user\Documents\Programozás\SelfDrivingCar\CarlaRun\PythonAPI\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import LocalPlanner
import carla
import math
from shapely.geometry import Polygon
import numpy as np

class BasicAgent(object):
    def __init__(self, vehicle, spawn_point: carla.Location, endlocation: carla.Location, target_speed=30, MAX_STEER_DEGREES = 40):

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()

        self.MAX_STEER_DEGREES = MAX_STEER_DEGREES
        self._map = self._world.get_map()
        self._offset = 0

        self.settings = self._world.get_settings()
        self._base_vehicle_threshold = 8 # meters
        self._speed_ratio = 1
        self.dt = self.settings.fixed_delta_seconds or 0.05  # fallback if None
        # PID parameters
        self.Kp = 0.1
        self.Ki = 0.0
        self.Kd = 0.1
        self.integral_error = 0.0
        self.last_error = 0.0
        self.max_speed = target_speed
        # Route planner
        self.sampling_resolution = 1
        self.grp = GlobalRoutePlanner(self._world.get_map(), self.sampling_resolution)
        self.spawn_point = spawn_point
        self.endlocation = endlocation
        self.route = self.grp.trace_route(self.spawn_point, self.endlocation)

        # where we are on the route
        self._route_index = 0
        self.draw_route()
        self.waitTime = 0

    # -------------------------------------------------
    #   VISUALIZE ROUTE
    # -------------------------------------------------
    def draw_route(self, life_time=30.0):
        debug = self._world.debug
        color_wp = carla.Color(0, 255, 0)   # green route points
        color_line = carla.Color(0, 0, 255) # blue lines between them

        prev_loc = None
        for wp, road_option in self.route:
            loc = wp.transform.location

            # draw a small point at each waypoint
            debug.draw_point(
                loc,
                size=0.1,
                color=color_wp,
                life_time=life_time
            )

            # draw line from previous waypoint
            if prev_loc is not None:
                debug.draw_line(
                    prev_loc,
                    loc,
                    thickness=0.05,
                    color=color_line,
                    life_time=life_time
                )

            prev_loc = loc

    @property
    def speed(self):
        vel = self._vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


    def update_control(self, desired_speed):
        current_speed = self.speed

        print("Current Speed")
        print(current_speed)
        print("Desired Speed")
        print(desired_speed)


        speed_error = desired_speed - current_speed
        self.integral_error += speed_error * self.dt
        derivative_error = (speed_error - self.last_error) / self.dt
        self.last_error = speed_error

        # PID computation
        control_output = self.Kp * speed_error + self.Ki * self.integral_error + self.Kd * derivative_error


        # Map control output to throttle and brake command
        if control_output > 0:
            throttle = min(control_output, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(control_output), 1.0)
        
        return throttle, brake
    
    def done(self):
        return self._route_index >= len(self.route)


     # --- Internal: advance along route when we are close to a waypoint ---
    def _update_route_progress(self, distance_threshold=3):
        if self.done():
            return

        veh_loc = self._vehicle.get_location()

        # move forward while we are close enough to the next point
        while self._route_index <= len(self.route) - 1:
            wp, _ = self.route[self._route_index]
            dist = wp.transform.location.distance(veh_loc)
            if dist < distance_threshold:
                self._route_index += 1
            else:
                break

    # --- Internal: choose a target waypoint ahead ---
    def _get_target_waypoint(self, look_ahead=1):
        if self.done():
            return None
        idx = min(self._route_index + look_ahead, len(self.route) - 1)
        wp, _ = self.route[idx]
        return wp
    
        # --- Internal: compute steering towards a waypoint ---
    def _compute_steering(self, target_wp, steer_gain=1.0):
        if target_wp is None:
            return 0.0
        veh_transform = self._vehicle.get_transform()
        veh_loc = veh_transform.location
        veh_yaw = veh_transform.rotation.yaw * math.pi / 180.0

        # Vehicle forward vector in 2D
        forward = carla.Vector3D(
            x=math.cos(veh_yaw),
            y=math.sin(veh_yaw),
            z=0.0
        )

        # Vector from vehicle to target waypoint
        target_loc = target_wp.transform.location
        to_target = carla.Vector3D(
            x=target_loc.x - veh_loc.x,
            y=target_loc.y - veh_loc.y,
            z=0.0
        )

        # Normalize
        f_norm = math.sqrt(forward.x**2 + forward.y**2) + 1e-6
        t_norm = math.sqrt(to_target.x**2 + to_target.y**2) + 1e-6
        forward.x /= f_norm
        forward.y /= f_norm
        to_target.x /= t_norm
        to_target.y /= t_norm

        # Signed angle between forward and to_target
        dot = forward.x * to_target.x + forward.y * to_target.y
        det = forward.x * to_target.y - forward.y * to_target.x
        angle = math.atan2(det, dot)  # radians, left:+, right:-

        steer = steer_gain * angle      # simple proportional controller
        steer = max(-1.0, min(1.0, steer))  # clamp to [-1, 1]

        return steer


    def angle_between(self, v1, v2):
        return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))



    def run_step(self):
        hazard_detected = False

        # Retrieve all relevant actors
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vel = self._vehicle.get_velocity()
        vehicle_speed = (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))/3.6

        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        control = carla.VehicleControl()
        # If route finished, stop
        if self.done():
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0
            return control
        
        # 1) Update where we are on the route
        self._update_route_progress()
        # 2) Get a look-ahead waypoint
        target_wp = self._get_target_waypoint()

        # 3) Steering towards it
        steer = self._compute_steering(target_wp)

        # 4) Speed control (you can make this smarter if you want)
        desired_speed = self.max_speed
        throttle, brake = self.update_control(desired_speed)

        # 5) Build control
        control.throttle = throttle
        control.brake = brake
        control.steer = steer
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control
    

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

        """
        control.throttle = 0.0
        control.brake = 1
        control.hand_brake = False
        return control
    

    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            #for wp, _ in self._local_planner.get_plan():
            for wp, _ in self.route:
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (target_wpt.is_junction) and route_polygon:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (True, target_vehicle, self._compute_distance(target_vehicle.get_location(), ego_location))

            # # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    steps=0
                    next_index = min(self._route_index + steps, len(self.route) - 1)
                    if len(self.route) > next_index:
                        next_wpt, _ = self.route[next_index]
                
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

            target_forward_vector = target_transform.get_forward_vector()
            target_extent = target_vehicle.bounding_box.extent.x
            target_rear_transform = target_transform
            target_rear_transform.location -= carla.Location(
                x=target_extent * target_forward_vector.x,
                y=target_extent * target_forward_vector.y,
            )

            if self._is_within_distance(target_rear_transform, ego_front_transform, max_distance):# [low_angle_th, up_angle_th]):
                return (True, target_vehicle, self._compute_distance(target_transform.location, ego_transform.location))

        return (False, None, -1)
    def _is_within_distance(self, target_transform, reference_transform, max_distance, angle_interval=None):
        """
        Check if a location is both within a certain distance from a reference object.
        By using 'angle_interval', the angle between the location and reference transform
        will also be tkaen into account, being 0 a location in front and 180, one behind.

        :param target_transform: location of the target object
        :param reference_transform: location of the reference object
        :param max_distance: maximum allowed distance
        :param angle_interval: only locations between [min, max] angles will be considered. This isn't checked by default.
        :return: boolean
        """
        target_vector = np.array([
            target_transform.location.x - reference_transform.location.x,
            target_transform.location.y - reference_transform.location.y
        ])
        norm_target = np.linalg.norm(target_vector)

        # If the vector is too short, we can simply stop here
        if norm_target < 0.001:
            return True

        # Further than the max distance
        if norm_target > max_distance:
            return False

        # We don't care about the angle, nothing else to check
        if not angle_interval:
            return True

        min_angle = angle_interval[0]
        max_angle = angle_interval[1]

        fwd = reference_transform.get_forward_vector()
        forward_vector = np.array([fwd.x, fwd.y])
        angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

        return min_angle < angle < max_angle
    
    def _compute_distance(self, location_1, location_2):
        """
        Euclidean distance between 3D points

            :param location_1, location_2: 3D points
        """
        x = location_2.x - location_1.x
        y = location_2.y - location_1.y
        z = location_2.z - location_1.z
        norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
        return norm