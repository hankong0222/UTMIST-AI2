# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
#
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
#
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
#
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional

import numpy as np
import torch
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO

from environment.environment import Player # Importing an LSTM

# To run the sample TTNN model, you can uncomment the 2 lines below:
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
            self,
            file_path: Optional[str] = None,
            learning_rate: float = 5e-4,
            n_steps: int = 1024,
            batch_size: int = 128,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            ent_coef: float = 0.01,
            clip_range: float = 0.2,
            verbose: int = 1,
            max_grad_norm: float = 0.5,
            device: str = "auto",
    ):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.verbose = verbose
        self.max_grad_norm = max_grad_norm
        self.device = device
        super().__init__(file_path)

        # To run a TTNN model, you must maintain a pointer to the device and can be done by
        # uncommmenting the line below to use the device pointer
        # self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))

    def _initialize(self) -> None:
        if self.file_path is None:
            # Deep MLP architecture optimized for fighting games
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[64, 128, 256, 128, 64],  # Deep policy network
                    vf=[64, 128, 256, 128, 64],  # Deep value network
                ),
                ortho_init=True,
            )

            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                ent_coef=self.ent_coef,
                clip_range=self.clip_range,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                normalize_advantage=True,
                max_grad_norm=self.max_grad_norm,
                device=self.device,
            )
            del self.env
        else:
            self.model = PPO.load(
                self.file_path, n_steps=30 * 90 * 3, batch_size=128, device=self.device
            )

        # To run the sample TTNN model during inference, you can uncomment the 5 lines below:
        # This assumes that your self.model.policy has the MLPPolicy architecture defined in `train_agent.py` or `my_agent_tt.py`
        # mlp_state_dict = self.model.policy.features_extractor.model.state_dict()
        # self.tt_model = TTMLPPolicy(mlp_state_dict, self.mesh_device)
        # self.model.policy.features_extractor.model = self.tt_model
        # self.model.policy.vf_features_extractor.model = self.tt_model
        # self.model.policy.pi_features_extractor.model = self.tt_model

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
        # action, _ = self.model.predict(obs)
        # obs.player = self.env.objects["player"]
        # obs.opponent = self.env.objects["opponent"]

        # # Player and opponent positions
        # player_x = obs.player.body.position.x
        # player_y = obs.player.body.position.y
        # opponent_x = obs.opponent.body.position.x
        # opponent_y = obs.opponent.body.position.y

        # # Calculate distances
        # distance_x = opponent_x - player_x
        # distance_y = opponent_y - player_y
        # distance = (distance_x**2 + distance_y**2)**0.5
        # obs.distance = distance

        # # Determine opponent's position relative to player
        # if distance_x > 0:
        #     obs.opponent_position = "right"
        # else:
        #     obs.opponent_position = "left"

        # # If the opponent is within a certain range, and on a specific side, modify action
        # if distance < 50:
        #     if obs.opponent_position == "right":
        #         # If close to opponent on the right, choose aggressive action
        #         action = self.act.helper.press_keys(['d'], action) + self.act.helper.press_keys(['w'], action) + self.act.helper.press_keys(['j'], action) + self.act.helper.press_keys(['s'], action) + self.act.helper.press_keys(['j'], action) + self.act.helper.press_keys(['k'], action)
        #     elif obs.opponent_position == "left":
        #         # If close to opponent on the left, choose aggressive action
        #         action = self.act.helper.press_keys(['a'], action) + self.act.helper.press_keys(['w'], action) + self.act.helper.press_keys(['j'], action) + self.act.helper.press_keys(['s'], action) + self.act.helper.press_keys(['j'], action) + self.act.helper.press_keys(['k'], action)
        # else:
        #     # If far from opponent, choose defensive action
        #     action = self.defensive_action(obs)

        # if obs.player.health < 20:
        #     # If health is low, prioritize evasion
        #     action = self.evasive_action(obs)

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.verbose = 1

        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
