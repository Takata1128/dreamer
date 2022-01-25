import numpy as np
import torch


class PrioritizedReplayBuffer:
    """
    Prioritized Expericence Replay
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.priorities[0] = 1.0

    def __len__(self):
        return len(self.buffer)

    def push(self, experience):
        """
        経験をリプレイバッファに保存
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience

        # 優先度は最初は大きな値で初期化、サンプルされたときに更新する
        self.priorities[self.index] = self.priorities.max()
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size, alpha=0.6, beta=0.4):
        """
        バッファから優先度に従いサンプル
        """

        # 優先度からサンプル確率を計算
        priorities = self.priorities[
            : self.buffer_size if len(self.buffer) == self.buffer_size else self.index
        ]
        priorities = priorities ** alpha
        prob = priorities / priorities.sum()

        # 確率に従ってサンプル
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)

        # ロスを重みづけするための重み
        weights = (len(self.buffer) * prob[indices]) ** (-beta)
        weights = weights / weights.max()

        obs, action, reward, next_obs, done = zip(*[self.buffer[i] for i in indices])

        return (
            torch.stack(obs),
            torch.as_tensor(action),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.stack(next_obs),
            torch.as_tensor(done, dtype=torch.uint8),
            indices,
            torch.as_tensor(weights, dtype=torch.float32),
        )

    def update_priorities(self, indices, priorities):
        """
        優先度の更新(最低保証の微小値を加算)
        """
        self.priorities[indices] = priorities + 1e-4
