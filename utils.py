import numpy as np
import torch
import torch.nn as nn
from typing import Iterable


def get_parameters(modules: Iterable[nn.Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


# def preprocess_obs(obs):
#     """
#     画像の変換. [0,255] -> [-0.5,0.5]
#     """

#     obs = obs.astype(np.float32)
#     normalized_obs = obs / 255.0 - 0.5
#     return normalized_obs


# def lambda_target(rewards, values, gamma, lambda_):
#     """
#     価値関数の学習の為のλ-return
#     """
#     V_lambda = torch.zeros_like(rewards, device=rewards.device)

#     H = rewards.shape[0] - 1
#     V_n = torch.zeros_like(rewards, device=rewards.device)
#     V_n[H] = values[H]

#     for n in range(1, H + 1):
#         # n-step return
#         # 系列が途中で終わったら、可能な中で最大のnを用いたn-step
#         V_n[:-n] = (gamma ** n) * values[n:]
#         for k in range(1, n + 1):
#             if k == n:
#                 V_n[:-n] += (gamma ** (n - 1)) * rewards[k:]
#             else:
#                 V_n[:-n] += (gamma ** (k - 1)) * rewards[k : -n + k]

#         # lambda_でn-step returnを重みづけてλ-returnを計算
#         if n == H:
#             V_lambda += (lambda_ ** (H - 1)) * V_n
#         else:
#             V_lambda += (1 - lambda_) * (lambda_ ** (n - 1)) * V_n

#     return V_lambda
