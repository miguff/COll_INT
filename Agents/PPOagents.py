from Agents.basicagent import BasicAgent
import math
import carla
from Agents.Networks import ActorNetwork, CriticNetwork, CarEncoder
import torch as T
device = T.device("cuda" if T.cuda.is_available() else "cpu")



#Egy nagy Buffer
#egy nagy tanuló algoritmus, amibe a buffer bele fog menni és tanul és 
#az ágensek ezt a közönes tanult algoritmust tudják majd használni, gyakorlatilkag 
#inferenciára.




class PPOAgent(BasicAgent):
#class PPOAgent():
    def __init__(self, 
                vehicle,
                spawn_point,
                endlocation,
                target_speed=30,
                fixed_length_state=False,
                speed_learn_booster = 2,
                lookahead = 4):
        super().__init__(vehicle, spawn_point, endlocation, target_speed)
        self.fixed_length_state = fixed_length_state

        self.speed_learn_booster = speed_learn_booster
        self.lookahead = lookahead
        self.next_waypoint_x = spawn_point.x
        self.next_waypoint_y = spawn_point.y
        #// So that the next waypoint are updated, and then the algorithm can get the current value
        target_wp = self._get_target_waypoint(look_ahead=self.lookahead)
        self.next_waypoint_x = target_wp.transform.location.x
        self.next_waypoint_y = target_wp.transform.location.y
        
        
    def run_step(self, action):

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
        target_wp = self._get_target_waypoint(look_ahead=self.lookahead)
        
        if target_wp is not None:
            self.next_waypoint_x = target_wp.transform.location.x
            self.next_waypoint_y = target_wp.transform.location.y
        # 3) Steering towards it
        steer = self._compute_steering(target_wp)

        #Diszkrét Action térre le lehetne korlátolni és akkor vagy full brake v full throttle
        #2 folytonos érték a throttle-re vagy brake-re, ezeket átadni a rendszernek és ezekkel menni, nem kell a 0.-ös elválasztás.


        # 4) Speed control (you can make this smarter if you want)
        #throttle = float(action[0])
        #brake = float(action[1])

        if action[0][0] >= action[0][1]:
            throttle = float(action[0][0])
            brake = 0
        else:
            throttle = 0
            brake = float(action[0][1])

        # 5) Build control
        control.throttle = throttle
        control.brake = brake
        control.steer = steer
        # if hazard_detected:
        #     control = self.add_emergency_stop(control)
        return control
