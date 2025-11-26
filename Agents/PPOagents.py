from Agents.basicagent import BasicAgent
import math
import carla
from Agents.Networks import ActorNetwork, CriticNetwork, CarEncoder
import torch as T
device = T.device("cuda" if T.cuda.is_available() else "cpu")





class PPOAgent(BasicAgent):
#class PPOAgent():
    def __init__(self, 
                vehicle,
                spawn_point,
                endlocation,
                target_speed=30,
                embed_dim=64,
                car_feature_dim=5,
                fixed_length_state=False):
        super().__init__(vehicle, spawn_point, endlocation, target_speed)
        self.fixed_length_state = fixed_length_state

        self.encoder = CarEncoder(in_dim=car_feature_dim,
                                  hidden_dim=64,
                                  embed_dim=embed_dim).to(device)

        self.actor = ActorNetwork(input_dims=embed_dim).to(device)
        self.critic = CriticNetwork(input_dims=embed_dim).to(device)
        
       
        self.next_waypoint_x = spawn_point.x
        self.next_waypoint_y = spawn_point.y
        #// So that the next waypoint are updated, and then the algorithm can get the current value
        target_wp = self._get_target_waypoint()
        self.next_waypoint_x = target_wp.transform.location.x
        self.next_waypoint_y = target_wp.transform.location.y
        

    
    def _encode_state(self, state):
        state = T.tensor(state, dtype=T.float32, device=device)
        state = state.unsqueeze(0)
        with T.no_grad():
            emb = self.encoder(state)
        return emb

    
    def choose_action(self, state):
        emb = self._encode_state(state)
        with T.no_grad():
            action, logprob = self.actor.act(emb)
            value = self.critic(emb).item()

        return action, logprob, value

    def run_step(self, state):
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
        #self._update_route_progress()
        # 2) Get a look-ahead waypoint
        target_wp = self._get_target_waypoint()
        self.next_waypoint_x = target_wp.transform.location.x
        self.next_waypoint_y = target_wp.transform.location.y
        # 3) Steering towards it
        steer = self._compute_steering(target_wp)

        # 4) Speed control (you can make this smarter if you want)
        control_value, log_prob, stateactionvalue = self.choose_action(state)
        if control_value[0] > 0.5:
            throttle = float(control_value[1])
            brake = 0
        else:
            brake = float(control_value[1])
            throttle = 0


        # 5) Build control
        control.throttle = throttle
        control.brake = brake
        control.steer = steer
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control
