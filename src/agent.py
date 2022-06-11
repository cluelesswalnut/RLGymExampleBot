# import os

from stable_baselines3 import PPO
import pathlib
from rlgym.utils.action_parsers.discrete_act import DiscreteAction


class Agent:
    def __init__(self):
        # If you need to load your model from a file this is the time to do it
        # You can do something like:
        #
        # self.actor = # your Model
        #
        # cur_dir = os.path.dirname(os.path.realpath(__file__))
        # with open(os.path.join(cur_dir, 'model.p'), 'rb') as file:
        #     model = pickle.load(file)
        # self.actor.load_state_dict(model)
        # pass

        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": 0.2,
            "n_envs": 1,
            "device": "cpu"
        }


        # self.actor = PPO.load(str(_path) + '/exit_save', custom_objects=custom_objects)
        self.actor = PPO.load(str(_path) + '/rl_model_25273420_steps', custom_objects=custom_objects)
        self.parser = DiscreteAction()

    def act(self, state):
        # # Evaluate your model here
        # action = [1, 0, 0, 0, 0, 0, 0, 0]
        # return action

        action = self.actor.predict(state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)

        return x[0]
