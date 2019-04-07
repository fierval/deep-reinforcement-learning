from model import SharedPolicy
import torch
import numpy as np

from unityagents import UnityEnvironment

if if __name__ == "__main__":
    env = UnityEnvironment(file_name="Tennis_Win/Tennis")
    env_info = env.reset(train_mode=True)[brain_name]

    