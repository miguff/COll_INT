from Agents.basicagent import BasicAgent
import math
import carla
import torch as T
device = T.device("cuda" if T.cuda.is_available() else "cpu")
import torch.nn.functional as F



class PPOAgent(BasicAgent):
    def __init__(self, 
                vehicle,
                spawn_point,
                endlocation,
                target_speed=30,
                fixed_length_state=False,
                speed_learn_booster = 2,
                lookahead = 4,
                distance_threshold = 3,
                need_safety_brake = 0):
        super().__init__(vehicle, spawn_point, endlocation, target_speed, need_safety_brake=need_safety_brake)
        self.fixed_length_state = fixed_length_state

        self.speed_learn_booster = speed_learn_booster
        self.lookahead = lookahead
        self.next_waypoint_x = spawn_point.x
        self.next_waypoint_y = spawn_point.y
        #// So that the next waypoint are updated, and then the algorithm can get the current value
        target_wp = self._get_target_waypoint(look_ahead=self.lookahead)
        self.next_waypoint_x = target_wp.transform.location.x
        self.next_waypoint_y = target_wp.transform.location.y
        self.distance_threshold = distance_threshold
        
        
    def run_step(self, action):

        control = carla.VehicleControl()
        #// If route finished, stop
        if self.done():
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0
            return control
        
        # 1) Update where we are on the route
        self._update_route_progress(self.distance_threshold)
        # 2) Get a look-ahead waypoint
        target_wp = self._get_target_waypoint(look_ahead=self.lookahead)
        
        if target_wp is not None:
            self.next_waypoint_x = target_wp.transform.location.x
            self.next_waypoint_y = target_wp.transform.location.y
        # 3) Steering towards it
        steer = self._compute_steering(target_wp)

        throttle = F.relu(action)
        brake = F.relu(-action)

        # 5) Build control
        control.throttle = throttle.item()
        control.brake = brake.item()
        control.steer = steer

        return control
