from PIL import Image
from black import re
import torch
import os
import numpy as np
from agent import Agent

from dqn.qnet import QNetwork
from models.rssm import RecurrentStateSpaceModel
from models.observation_model import ObsEncoder, ObsDecoder
from models.action_model import ActionModel


class DreamerV2Evaluator(object):
    """
    モデルの評価を行うためのクラス
    """

    def __init__(self, env, config, device):
        self.device = device
        self.env = env
        self.config = config
        self.action_size = config.action_size

    def load_model(self, model_path):
        """
        保存していたモデルをロード
        """
        saved_dict = torch.load(model_path)
        obs_shape = self.config.obs_shape
        action_size = self.config.action_size
        rnn_hidden_dim = self.config.rnn_hidden_dim
        category_size = self.config.category_size
        class_size = self.config.class_size
        state_dim = category_size * class_size

        embedding_size = self.config.embedding_size
        rssm_node_size = self.config.rssm_node_size

        self.rssm = (
            RecurrentStateSpaceModel(
                rssm_node_size,
                embedding_size,
                action_size,
                state_dim,
                rnn_hidden_dim,
                category_size,
                class_size,
            )
            .to(self.device)
            .eval()
        )
        self.action_model = (
            ActionModel(
                state_dim=state_dim,
                rnn_hidden_dim=rnn_hidden_dim,
                action_dim=action_size,
            )
            .to(self.device)
            .eval()
        )

        self.encoder = ObsEncoder(obs_shape, embedding_size).to(self.device).eval()
        self.decoder = (
            ObsDecoder(obs_shape, state_dim + rnn_hidden_dim).to(self.device).eval()
        )

        self.rssm.load_state_dict(saved_dict["RSSM"])
        self.encoder.load_state_dict(saved_dict["Encoder"])
        self.decoder.load_state_dict(saved_dict["Decoder"])
        self.action_model.load_state_dict(saved_dict["ActionModel"])

    def close(self):
        self.env.close()

    def evalueate_sequence(self, result_dir, model_dir):
        """
        時系列に沿って複数モデルの評価
        モデルのindexとスコアを返す
        """
        indices = []
        model_files = []
        for f in os.listdir(model_dir):
            if f == "models_best.pth":
                continue
            index = int(re.sub(r"[^0-9]", "", f))
            indices.append(index)
            model_files.append(f)

        eval_scores = []
        for i, f in sorted(zip(indices, model_files)):
            if f == "models_best.pth":
                continue
            eval_score = self.evaluate(os.path.join(model_dir, f))
            eval_scores.append(eval_score)
        return sorted(indices), eval_scores

    def view_episode(self, model_path):
        """
        1エピソード描画してImageのリストを返却
        """

        def bool_state_to_image(state):
            """
            booleanで表された状態を描画
            """
            image = Image.new(
                "RGB", (self.config.obs_shape[1] * 40, self.config.obs_shape[2] * 40)
            )
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
            for i in range(self.config.obs_shape[0]):
                for w in range(self.config.obs_shape[1]):
                    for h in range(self.config.obs_shape[2]):
                        if state[i, w, h] == True:
                            for j in range(40):
                                for k in range(40):
                                    image.putpixel((h * 40 + j, w * 40 + k), colors[i])
            return image

        def tensor_state_to_image(state):
            """
            テンソルで表された状態を描画
            """
            image = Image.new(
                "RGB", (self.config.obs_shape[1] * 40, self.config.obs_shape[2] * 40)
            )
            clamp = lambda x: min(255, max(x, 0))

            max_c = torch.argmax(state, dim=0)
            for w in range(self.config.obs_shape[1]):
                for h in range(self.config.obs_shape[2]):
                    color = [0, 0, 0]
                    for i in range(self.config.obs_shape[0]):
                        if i > 3:
                            color[:] = clamp(int(255 * state[i, w, h]))
                        else:
                            color[i] = clamp(int(255 * state[i, w, h]))
                    for j in range(40):
                        for k in range(40):
                            image.putpixel((h * 40 + j, w * 40 + k), tuple(color))
            return image

        self.load_model(model_path)
        agent = Agent(self.encoder, self.rssm, self.action_model, self.config)
        obs, done = self.env.reset(), False
        score = 0
        images = []
        predicted_images = []

        # 最初の10ステップ行動して系列情報をrnnに蓄積
        for _ in range(10):
            action, _ = agent(obs, not done, 0, training=False)
            obs, rew, done, _ = self.env.step(action.squeeze().cpu().numpy())

        embedded_obs = self.encoder(
            torch.as_tensor(obs).float().unsqueeze(0).to(self.device)
        )
        rnn_hidden = agent.rnn_hidden
        state = self.rssm.get_stoch_state(self.rssm.posterior(rnn_hidden, embedded_obs))
        step = 10
        while not done and step <= 300:
            print("step: ", step)
            action, _ = agent(obs, not done, 0, training=False)
            next_obs, rew, done, _ = self.env.step(action.squeeze().cpu().numpy())
            score += rew
            obs = next_obs
            step += 1

            state_prior_logits, rnn_hidden = self.rssm.prior(state, action, rnn_hidden)
            state = self.rssm.get_stoch_state(state_prior_logits)
            pred_obs_dist = self.decoder(state, rnn_hidden)
            pred_obs = pred_obs_dist.mean.squeeze(0)

            images.append(bool_state_to_image(obs))
            predicted_images.append(tensor_state_to_image(pred_obs))

        return images, predicted_images

    def evaluate(self, model_path):
        """
        評価するエピソード分プレイさせてモデルを評価
        """
        self.load_model(model_path)
        eval_episodes = self.config.eval_episode
        agent = Agent(self.encoder, self.rssm, self.action_model, self.config)
        eval_scores = []
        for e in range(eval_episodes):
            obs, done = self.env.reset(), False
            score = 0
            while not done:
                action, _ = agent(obs, not done, 0, training=False)
                next_obs, rew, done, _ = self.env.step(action.squeeze().cpu().numpy())
                score += rew
                obs = next_obs
            eval_scores.append(score)
        print(
            "average evaluation score for model at "
            + model_path
            + " = "
            + str(np.mean(eval_scores))
        )
        return np.mean(eval_scores)


class DQNEvaluator(object):
    """
    モデルの評価を行うためのクラス
    """

    def __init__(self, env, config, device):
        self.device = device
        self.env = env
        self.config = config
        self.action_size = config.action_size

    def close(self):
        self.env.close()

    def load_model(self, model_path):
        """
        保存していたモデルをロード
        """
        self.net = (
            QNetwork(self.env.observation_space.shape, n_action=self.env.action_space.n)
            .to(self.device)
            .eval()
        )
        self.net.load_state_dict(torch.load(model_path))

    def evalueate_sequence(self, result_dir, model_dir):
        """
        時系列に沿って複数モデルの評価
        モデルのindexとスコアを返す
        """
        indices = []
        model_files = []
        for f in os.listdir(model_dir):
            if f == "models_best.pth":
                continue
            index = int(re.sub(r"[^0-9]", "", f))
            indices.append(index)
            model_files.append(f)

        eval_scores = []
        for i, f in sorted(zip(indices, model_files)):
            if f == "models_best.pth":
                continue
            eval_score = self.evaluate(os.path.join(model_dir, f))
            eval_scores.append(eval_score)

        return sorted(indices), eval_scores

    def evaluate(self, model_path):
        """
        評価するエピソード分プレイさせてモデルを評価
        """
        self.load_model(model_path)
        eval_episodes = self.config.eval_episode
        eval_scores = []
        for e in range(eval_episodes):
            obs, done = self.env.reset(), False
            score = 0
            while not done:
                obs = obs.float().to(self.device)
                action = self.net.act(obs, 0)
                next_obs, rew, done, _ = self.env.step(action)
                score += rew
                obs = next_obs
            eval_scores.append(score)
        print(
            "average evaluation score for model at "
            + model_path
            + " = "
            + str(np.mean(eval_scores))
        )
        return np.mean(eval_scores)
