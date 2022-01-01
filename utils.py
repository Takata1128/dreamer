import numpy as np
import torch


def preprocess_obs(obs):
    """
    画像の変換. [0,255] -> [-0.5,0.5]
    """

    obs = obs.astype(np.float32)
    normalized_obs = obs / 255.0 - 0.5
    return normalized_obs


def lambda_target(rewards, values, gamma, lambda_):
    """
    価値関数の学習の為のλ-return
    """
    V_lambda = torch.zeros_like(rewards, device=rewards.device)

    H = rewards.shape[0] - 1
    V_n = torch.zeros_like(rewards, device=rewards.device)
    V_n[H] = values[H]

    for n in range(1, H + 1):
        # n-step return
        # 系列が途中で終わったら、可能な中で最大のnを用いたn-step
        V_n[:-n] = (gamma ** n) * values[n:]
        for k in range(1, n + 1):
            if k == n:
                V_n[:-n] += (gamma ** (n - 1)) * rewards[k:]
            else:
                V_n[:-n] += (gamma ** (k - 1)) * rewards[k : -n + k]

        # lambda_でn-step returnを重みづけてλ-returnを計算
        if n == H:
            V_lambda += (lambda_ ** (H - 1)) * V_n
        else:
            V_lambda += (1 - lambda_) * (lambda_ ** (n - 1)) * V_n

    return V_lambda
