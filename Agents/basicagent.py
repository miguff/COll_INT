import sys
sys.path.append(r'C:\Users\local_user\Documents\Programozás\SelfDrivingCar\CarlaRun\PythonAPI\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner
import carla
import math

class BasicAgent(object):
    def __init__(self, vehicle, spawn_point: carla.Location, endlocation: carla.Location, target_speed=30):

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.settings = self._world.get_settings()

        self.dt = self.settings.fixed_delta_seconds or 0.05  # fallback if None
        # PID parameters
        self.Kp = 0.3
        self.Ki = 0.0
        self.Kd = 0.1
        self.integral_error = 0.0
        self.last_error = 0.0
        self.max_speed = target_speed

        # Route planner
        self.sampling_resolution = 5
        self.grp = GlobalRoutePlanner(self._world.get_map(), self.sampling_resolution)
        self.spawn_point = spawn_point
        self.endlocation = endlocation
        self.route = self.grp.trace_route(self.spawn_point, self.endlocation)

        # where we are on the route
        self._route_index = 0
        self.draw_route()

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




    def run_step(self):
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

        return control