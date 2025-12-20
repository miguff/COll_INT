# Collective Intelligence
Second Assignment of Collective Intelligence

This repository contains the second assignment for the **Collective Intelligence** course.

---

### Version Information
**Version:** 2.0
**Status:** Final Version for the semester.  
**Last Updated:** 2025.12.20

## Update 3:

### Parameters of the run
- DELTA = 0.05 (What is time between two ticks)
- SIMULATION_TIME = 100 (How long a simulation should last.)
- INPUT_STATE = This is a list, the first place is reserved for the current agent. And each value inside this list is made up of the followings :(vehicle_pos_x, vehicle_pos_y, vehicle_speed, next_waypoint_x, next_waypoint_y)

### State of the project
- Implemented PPO algorithm, and started testing with various point systems. In the log files the ones where the dates are before 2025-12-19 they are the one where I tried to give the model not only if the model reached an end state, but also some dense rewards as well. It can be seen that it was tested, with different hyper-parameters also. The rewards were given, when the accelerated and the speed was less the a given speed, when the speed is faster than the maximum allowed speed, and when the speed is between a given range. I found that it was not a great way. Next, after 2025-12-18, the rewards were changed in a way, that they were given, when the car reached the next waypoint, so not directly if it accelerates or brakes. And it also got rewards, when it was faster than the maximum, and also when the speed was between a given range. It got rewards, when it reached the goal state, or when it collided. 

### Challenges

- Tried to only give sparse rewards when the vehicles reaches the terminal states, but I found that since in carla you need multiple throttle values, to make the car car, it would require a lot of random consequent throttle values, which are not likely. I tried to train them for 500 episode, with an elevated number of simulation time, and the cars did not move.
- I think one of the biggest challenges is to how to make the car move, you need multiple throttle values in the beginnig, just to make it go. After that it also needs to brake, when the states says that, there are cars near you.
- Rethink the reward system. Figure out how to give points for braking. Add reward based on driving smoothness. Give rewards based on distance to other vehicles.

### BUGS
- When I created this version of the documentation, I found a bug that I do not remember being there previously. In the FIFO system, in one side of the intersection, only the first cars stop, and the others that comes after them, just crashes into them. I have not found a solution for it yet.

### Next steps
- Add wandb to the project
- Think about the reward structure
- Currently it uses Beta distribution for action sampling, It might give it a try to use Normal distribution, so that it naturally return values between -1 and 1, without any math conversion.


## UPDATE 2:

- Implemnted FIFO based system.

## UPDATE 1:
- Implemented a Baseline version, where the cars can go however they like, no need to stop and wait for others.
- Implememted 2 metrics, Successful throuput and collision
- Next is to implement the FIFO system

### Goal 1: Setup and Test Rule-Based Systems
- Implement a **rule-based system** simulating decision-making at intersections.  
- Test and validate system performance. 

---

### Goal 2: Create and Evaluate Metrics
Develop a set of quantitative metrics to assess system performance, including:
- **Waiting Time**: Measure and analyze vehicle waiting times using a suitable statistical distribution.    
- **Throughput**: Quantify how many vehicles successfully pass through the system within a given time frame.

---

### Deadline
**Baseline Tests:** 2025.11.16

