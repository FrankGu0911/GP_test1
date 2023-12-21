import os
import pygame
import numpy as np
import cv2
import time
import math
import carla
from collections import deque
from leaderboard.autoagents import autonomous_agent
import torch
SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')

class HIDPannel(object):
    def __init__(self):
        self._width = 1200
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Human Agent")
    
    def run_interface(self,input_data):
        surface = np.zeros((self._height, self._width, 3),np.uint8)
        # rgb
        surface[:,:800] = cv2.resize(input_data['rgb'],(800,600))
        surface[:150,:200] = cv2.resize(input_data['rgb_left'],(200,150))
        surface[:150, 600:800] = cv2.resize(input_data['rgb_right'],(200,150))
        surface[:150, 300:500] = cv2.resize(input_data['rgb_focus'],(200,150))
        surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface[:150,198:202]=0
        surface[:150,298:302]=0
        surface[:150,498:502]=0
        surface[:150,598:602]=0
        surface[148:152, :200] = 0
        surface[148:152, 300:500] = 0
        surface[148:152, 600:800] = 0

        # lidar
        surface[0:200,800:1000] = cv2.resize(input_data['lidar'],(200,200))
        surface[0:202,998:1002] = 255
        surface[198:202,800:1200] = 255

        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()
        pygame.event.get()

def get_entry_point():
    return "LCDiffAgent"

class RoutePlanner(object):
    def __init__(self, min_distance, max_distance):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0])  # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945])  # for carla 9.10

        # self.debug = Plotter(debug_size)

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos["lat"], pos["lon"]])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        # self.debug.clear()

        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(
                self.route[i][0] - self.route[i - 1][0]
            )
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            r = 255 * int(distance > self.min_distance)
            g = 255 * int(self.route[i][1].value == 4)
            b = 255
            # self.debug.dot(gps, self.route[i][0], (r, g, b))

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        # self.debug.dot(gps, self.route[0][0], (0, 255, 0))
        # self.debug.dot(gps, self.route[1][0], (255, 0, 0))
        # self.debug.dot(gps, gps, (0, 0, 255))
        # self.debug.show()

        return self.route[1]

    def get_future_waypoints(self, num=10):
        res = []
        for i in range(min(num, len(self.route))):
            res.append(
                [self.route[i][0][0], self.route[i][0][1], self.route[i][1].value]
            )
        return res

class LCDiffAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self._hid = HIDPannel()
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -60.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 60.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb_right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 1.3,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]
    
    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def _lidar_2d(self,lidar_points:np.ndarray):
        lidar_raw = lidar_points[:,:3]
        lidar_raw[:,0] = -lidar_raw[:,0]
        lidar_raw = lidar_raw[(lidar_raw[:,0] > -22.5)]
        lidar_raw = lidar_raw[(lidar_raw[:,0] < 22.5)]
        lidar_raw = lidar_raw[(lidar_raw[:,1] < 45)]
        lidar_raw = lidar_raw[(lidar_raw[:,1] > 0)]
        lidar_down = lidar_raw[(lidar_raw[:,2] <= -2.3)]
        lidar_middle = lidar_raw[(lidar_raw[:,2] > -2.3)]
        lidar_middle = lidar_middle[(lidar_middle[:,2] < 1)]
        lidar_up = lidar_raw[(lidar_raw[:,2] > 1)]
        lidar_2d = np.zeros((3,256,256),dtype=np.uint8)
        for p in lidar_down:
            lidar_2d[0][int((45-p[1])/45*256)][int((p[0]+22.5)/45*256)] = 255
        for p in lidar_middle:
            lidar_2d[1][int((45-p[1])/45*256)][int((p[0]+22.5)/45*256)] = 255
        for p in lidar_up:
            lidar_2d[2][int((45-p[1])/45*256)][int((p[0]+22.5)/45*256)] = 255
        lidar_2d = np.transpose(lidar_2d,(1,2,0))
        return lidar_2d

    def tick(self, input_data):
        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_focus = cv2.cvtColor(input_data['rgb'][1][150:450, 200:600, :3], cv2.COLOR_BGR2RGB)
        lidar_data = input_data['lidar'][1][:, :3]
        lidar_processed = self._lidar_2d(lidar_data)
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]        
        if math.isnan(compass) == True:  # It can happen that the compass sends nan for a few frames
            compass = 0.0
        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "rgb_focus": rgb_focus,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }
        pos = self._get_position(result)
        result["gps"] = pos
        result["lidar"] = lidar_processed
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_command"] = next_cmd.value
        result['measurements'] = [pos[0], pos[1], compass, speed]
        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point
        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        self._hid.run_interface(tick_data)
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        return control
