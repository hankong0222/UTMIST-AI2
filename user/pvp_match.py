import os
# MUST set this BEFORE any pygame imports
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pygame
pygame.init()

from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent
from user.my_agent import SubmittedAgent

my_agent = UserInputAgent()

# Input your file path here in SubmittedAgent if you are loading a model:
opponent = SubmittedAgent(file_path=None)

match_time = 99999

# Run a single real-time match
run_real_time_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * 999990000,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)