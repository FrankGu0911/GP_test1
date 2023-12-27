import numpy as np
from collections import deque

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

class LCDiff_Controller(object):
    def __init__(self,config):
        self.turn_controller = PIDController(
            K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n
        )
        self.speed_controller = PIDController(
            K_P=config.speed_KP,
            K_I=config.speed_KI,
            K_D=config.speed_KD,
            n=config.speed_n,
        )
        self.config = config
        self.max_throttle = config.max_throttle

        self.stop_steps = 0
        self.forced_forward_steps = 0

    def run_step(self, speed, waypoints, stop_reason=None):
        """
        Arguments:
            speed: current speed of the vehicle
            waypoints: waypoints to track
            stop reason: [should_break,should_slow,junction,vehicle,bike,lane_vehicle,junction_vehicle,pedestrian,red_light]
        Returns:
            throttle: throttle signal
            brake: brake signal
            steer: steer signal
        """
        if speed < 0.2:
            self.stop_steps += 1
        else:
            self.stop_steps = max(0, self.stop_steps - 10)
        # Compute the target speed
        brake = False
        aim = (waypoints[1] + waypoints[0]) / 2.0
        # aim[1] *= -1
        distance = (aim[0] ** 2 + aim[1] ** 2) ** 0.5
        target_speed = distance
        if target_speed > self.config.max_speed:
            target_speed = self.config.max_speed
        elif target_speed < 0.2:
            target_speed = 0
            brake = True
        stop_str = ''
        if stop_reason is not None:
            if stop_reason[0] > 0.7: # stop
                brake = True
                target_speed = 0
                stop_str += 'stop '
            if stop_reason[1] > 0.7: # slow
                brake = False
                target_speed = 2
                stop_str += 'slow '
            if stop_reason[2] > 0.7: # junction
                brake = False
                target_speed = 2
                stop_str += 'junction '
            if stop_reason[3] > 0.7: # vehicle
                brake = True
                target_speed = 0
                stop_str += 'vehicle '
            if stop_reason[4] > 0.7: # bike
                brake = True
                target_speed = 0
                stop_str += 'bike '
            if stop_reason[5] > 0.7: # lane_vehicle
                brake = True
                target_speed = 0
                stop_str += 'lane_vehicle '
            if stop_reason[6] > 0.7: # junction_vehicle
                brake = True
                target_speed = 0
                stop_str += 'junction_vehicle '
            if stop_reason[7] > 0.7: # pedestrian
                brake = True
                target_speed = 0
                stop_str += 'pedestrian '
            if stop_reason[8] > 0.7: # red_light
                brake = True
                target_speed = 0
                stop_str += 'red_light '
        if speed > target_speed * self.config.brake_ratio:
            brake = True

        delta = np.clip(target_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        # Compute the target steer
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90
        if speed < 0.01:
            angle = 0
        steer = self.turn_controller.step(angle)
        if steer > 0.2:
            steer *= 1.25
        steer = np.clip(steer, -1.0, 1.0)
        if self.stop_steps > 1200:
            self.forced_forward_steps = 12
            self.stop_steps = 0
        if self.forced_forward_steps > 0:
            throttle = self.config.max_throttle
            brake = False
            self.forced_forward_steps -= 1
        if brake:
            brake = 1.0
            throttle = 0.0
        else:
            brake = 0.0
        # print("waypoints: ", waypoints)
        print("aim: ", aim)
        print("stop_reason: ", stop_reason)
        print("stop_str: ", stop_str)
        print("throttle: ", throttle, "brake: ", brake, "steer: ", steer)
        return throttle, brake, steer

