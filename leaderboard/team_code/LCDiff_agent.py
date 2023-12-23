import os
import pygame
import numpy as np
import cv2
import time
import math
import imp
import carla
from collections import deque
from leaderboard.autoagents import autonomous_agent
import torch
from torchvision.transforms import ToTensor, Resize, InterpolationMode, CenterCrop, Normalize, Compose
import clip
from diffusers import PNDMScheduler
from team_code.models.vae import VAE
from team_code.models.unet import UNet
from team_code.models.gru import GRU
from team_code.models.controlnet import ControlNet
from team_code.LCDiff_Controller import LCDiff_Controller
SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')

seg_tag = {
    0: [0, 0, 0],  # Unlabeled
    1: [70, 70, 70],  # Building
    2: [100, 40, 40],  # Fence
    3: [55, 90, 80],  # Other
    4: [220, 20, 60],  # Pedestrian
    5: [153, 153, 153],  # Pole
    6: [157, 234, 50],  # RoadLine
    7: [128, 64, 128],  # Road
    8: [244, 35, 232],  # Sidewalk
    9: [107, 142, 35],  # Vegetation
    10: [0, 0, 142],  # Car
    11: [102, 102, 156],  # Wall
    12: [220, 220, 0],  # TrafficSign
    13: [70, 130, 180],  # Sky
    14: [81, 0, 81],  # Ground
    15: [150, 100, 100],  # Bridge
    16: [230, 150, 140],  # RailTrack
    17: [180, 165, 180],  # GuardRail
    18: [250, 170, 30],  # TrafficLight
    19: [110, 190, 160],  # Static
    20: [170, 120, 50],  # Dynamic
    21: [45, 60, 150],  # Water
    22: [145, 170, 100],  # Terrain
    23: [255, 0, 0],  # RedLight
    24: [255, 255, 0],  # YellowLight
    25: [0, 255, 0],  # GreenLight
}


def cvt_rgb_seg(seg: np.ndarray):
    # print(seg.shape)
    for i in seg_tag:
        seg = np.where(seg == [i, i, i], np.array(seg_tag[i]), seg)

    seg = seg.astype(np.uint8)
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    return seg


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

    def run_interface(self, input_data):
        surface = np.zeros((self._height, self._width, 3), np.uint8)
        # rgb
        surface[:, :800] = cv2.resize(input_data['rgb'], (800, 600))
        surface[:150, :200] = cv2.resize(input_data['rgb_left'], (200, 150))
        surface[:150, 600:800] = cv2.resize(
            input_data['rgb_right'], (200, 150))
        surface[:150, 300:500] = cv2.resize(
            input_data['rgb_focus'], (200, 150))
        surface = cv2.putText(surface, 'Left  View', (40, 135),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        surface = cv2.putText(surface, 'Focus View', (335, 135),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        surface = cv2.putText(surface, 'Right View', (640, 135),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        surface[:150, 198:202] = 0
        surface[:150, 298:302] = 0
        surface[:150, 498:502] = 0
        surface[:150, 598:602] = 0
        surface[148:152, :200] = 0
        surface[148:152, 300:500] = 0
        surface[148:152, 600:800] = 0

        # lidar
        lidar_show = input_data['lidar']
        lidar_show[lidar_show != 0] = 255
        surface[0:200, 800:1000] = cv2.resize(lidar_show, (200, 200))
        surface[0:200, 1000:1200] = cv2.resize(input_data['bev'], (200, 200))
        surface[0:202, 998:1002] = 255
        surface[198:202, 800:1200] = 255

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
        self.scale = np.array(
            [111324.60662786, 111319.490945])  # for carla 9.10

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
        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
        self._hid = HIDPannel()
        self.step = -1
        self.skip_frames = self.config.skip_frames
        self.prev_control = None
        self.prev_lidar = None
        self.wall_start = time.time()
        self.initialized = False
        # init model
        self.clip_encoder, _ = clip.load("ViT-L/14", device="cuda:0")
        self.clip_encoder.eval()
        self.clip_preprocess = Compose([
            Resize(224, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(224),
            Normalize([0.48145466, 0.4578275, 0.40821073], 
                      [0.26862954, 0.26130258, 0.27577711]),
        ])
        self.diff_step = self.config.diff_step
        # TODO: use config file
        self.vae_model = VAE(26,26).cuda()
        vae_params = torch.load('leaderboard/team_code/models/vae_model_69.pth',map_location=torch.device('cuda:0'))['model_state_dict']
        self.vae_model.load_state_dict(vae_params)
        self.vae_model.eval()

        self.UNet_model = UNet().cuda()
        unet_params = torch.load('leaderboard/team_code/models/diffusion_model_36.pth',map_location=torch.device('cuda:0'))['model_state_dict']
        self.UNet_model.load_state_dict(unet_params)
        self.UNet_model.eval()
        self.scheduler = PNDMScheduler(
                        num_train_timesteps=1000, 
                        beta_end=0.012, 
                        beta_start=0.00085,
                        beta_schedule="scaled_linear",
                        prediction_type="epsilon",
                        set_alpha_to_one=False,
                        skip_prk_steps=True,
                        steps_offset=1,
                        trained_betas=None
                        )

        self.gru_model = GRU(with_lidar=True,with_rgb=True).cuda()
        gru_params = torch.load('leaderboard/team_code/models/gru_model_24.pth',map_location=torch.device('cuda:0'))['model_state_dict']
        self.gru_model.load_state_dict(gru_params)
        self.gru_model.eval()

        if self.config.with_lidar:
            self.controlnet = ControlNet().cuda()
            controlnet_params = torch.load('leaderboard/team_code/models/controlnet_4.pth',map_location=torch.device('cuda:0'))['model_state_dict']
            self.controlnet.load_state_dict(controlnet_params)
            self.controlnet.eval()
        
        self.controller = LCDiff_Controller(self.config)

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

    def _lidar_2d(self, lidar_points: np.ndarray):
        lidar_raw = lidar_points[:, :3]
        lidar_raw[:, 0] = -lidar_raw[:, 0]
        lidar_raw = lidar_raw[(lidar_raw[:, 0] > -22.5)]
        lidar_raw = lidar_raw[(lidar_raw[:, 0] < 22.5)]
        lidar_raw = lidar_raw[(lidar_raw[:, 1] < 45)]
        lidar_raw = lidar_raw[(lidar_raw[:, 1] > 0)]
        lidar_down = lidar_raw[(lidar_raw[:, 2] <= -2.3)]
        lidar_middle = lidar_raw[(lidar_raw[:, 2] > -2.3)]
        lidar_middle = lidar_middle[(lidar_middle[:, 2] < 1)]
        lidar_up = lidar_raw[(lidar_raw[:, 2] > 1)]
        lidar_2d = np.zeros((3, 256, 256), dtype=np.uint8)
        for p in lidar_down:
            lidar_2d[0][int((45-p[1])/45*256)][int((p[0]+22.5)/45*256)] += 1
        for p in lidar_middle:
            lidar_2d[1][int((45-p[1])/45*256)][int((p[0]+22.5)/45*256)] += 1
        for p in lidar_up:
            lidar_2d[2][int((45-p[1])/45*256)][int((p[0]+22.5)/45*256)] += 1
        lidar_2d = np.transpose(lidar_2d, (1, 2, 0))
        return lidar_2d

    def tick(self, input_data):
        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(
            input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(
            input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_focus = cv2.cvtColor(
            input_data['rgb'][1][150:450, 200:600, :3], cv2.COLOR_BGR2RGB)
        lidar_data = input_data['lidar'][1][:, :3]
        lidar_processed = self._lidar_2d(lidar_data)
        if np.max(lidar_processed) < 8 and self.prev_lidar is not None:
            lidar_processed = self.prev_lidar
        else:
            self.prev_lidar = lidar_processed
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        # It can happen that the compass sends nan for a few frames
        if math.isnan(compass) == True:
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
        R = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
        local_command_point = np.array(
            [next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point

        # Get BEV
        image_front = ToTensor()(rgb).unsqueeze(0).cuda()
        image_left = ToTensor()(rgb_left).unsqueeze(0).cuda()
        image_right = ToTensor()(rgb_right).unsqueeze(0).cuda()
        image_focus = ToTensor()(rgb_focus).unsqueeze(0).cuda()
        image_full = torch.cat(
            (
                self.clip_preprocess(image_front),
                self.clip_preprocess(image_left),
                self.clip_preprocess(image_right),
                self.clip_preprocess(image_focus),
            ),dim=0)
        pos_clip_feature = self.clip_encoder.encode_image(image_full).unsqueeze(0).to(torch.float32)
        neg_clip_feature = self.clip_encoder.encode_image(torch.zeros_like(image_full)).unsqueeze(0)
        clip_feature = torch.cat((neg_clip_feature,pos_clip_feature),dim=0).to(torch.float32)
        clip_feature = clip_feature.to(torch.float32)
        out_vae = torch.randn(1,4,32,32).cuda()
        self.scheduler.set_timesteps(self.diff_step,device='cuda:0')
        for cur_time in self.scheduler.timesteps:
            cur_time_in = torch.cat((cur_time.unsqueeze(0),cur_time.unsqueeze(0)),dim=0)
            noise = torch.cat((out_vae,out_vae),dim=0)
            noise = self.scheduler.scale_model_input(noise, cur_time)
            if self.config.with_lidar:  # ControlNet Use config file
                lidar_in = ToTensor()(lidar_processed).unsqueeze(0).cuda()
                lidar_in = torch.cat((lidar_in,lidar_in),dim=0)
                out_control_down, out_control_mid = self.controlnet(noise,clip_feature,time=cur_time_in,condition=lidar_in)
            else:
                out_control_down, out_control_mid = None, None
            pred_noise = self.UNet_model(out_vae=noise,
                                out_encoder=clip_feature,time=cur_time_in,
                                down_block_additional_residuals=out_control_down,
                                mid_block_additional_residual=out_control_mid)
            pred_noise = pred_noise[0] + 2 * (pred_noise[1] - pred_noise[0])
            out_vae = self.scheduler.step(pred_noise,cur_time,out_vae).prev_sample
        result['vae'] = out_vae.clone()
        bev = self.vae_model.decoder(out_vae).squeeze(0)
        bev = torch.nn.functional.softmax(bev,dim=0)
        bev = torch.argmax(bev,dim=0).unsqueeze(0)
        bev = (torch.cat((bev,bev,bev))).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        bev = cvt_rgb_seg(bev)
        bev = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)
        result['bev'] = bev
        # Get Control
        onehot_command = torch.zeros(6,dtype=torch.float32)
        onehot_command[result["next_command"] - 1] = 1
        local_command_point
        measurements = torch.cat([torch.tensor(local_command_point,dtype=torch.float32),onehot_command]).cuda()
        pred_wp = self.gru_model(out_vae.unsqueeze(0),measurements.unsqueeze(0),
                                 rgb_feature = pos_clip_feature,
                                 lidar_feature=ToTensor()(lidar_processed).unsqueeze(0).cuda())
        pred_wp = pred_wp.squeeze(0).cpu().detach().numpy()
        result['wp'] = pred_wp
        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
            return self.prev_control
        tick_data = self.tick(input_data)
        self._hid.run_interface(tick_data)
        control = carla.VehicleControl()
        control.throttle, control.brake, control.steer = self.controller.run_step(tick_data['speed'],tick_data['wp'])
        self.prev_control = control
        return control
