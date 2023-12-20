import os
import pygame
import numpy as np
import cv2
import time
# from leaderboard.autoagents import autonomous_agent
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
        
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()
        pygame.event.get()

# class LCDiffAgent(autonomous_agent.AutonomousAgent):
#     def setup(self, path_to_conf_file):
#         self.track = autonomous_agent.Track.SENSORS
#         pass

if __name__ == '__main__':
    a = HIDPannel()
    path = "E:\\dataset-val\\weather-0\\data\\routes_town01_long_w0_06_23_04_44_38"
    length = len(os.listdir(os.path.join(path,'rgb_full')))
    for i in range(length):
        input_data = {}
        rgb_full = cv2.imread(os.path.join(path,'rgb_full','%04d.jpg' %i))
        rgb_full = cv2.cvtColor(rgb_full, cv2.COLOR_BGR2RGB)
        input_data['rgb'] = rgb_full[:600,:]
        input_data['rgb_left'] = rgb_full[600:1200,:]
        input_data['rgb_right'] = rgb_full[1200:1800,:]
        input_data['rgb_focus'] = input_data['rgb'][150:450,200:600]
        a.run_interface(input_data)
        time.sleep(0.2)
        # break
    while(True):
        pass