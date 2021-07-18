import os

import torch

from cython_func.mcts_utils import CyphonNode
from .mcts import MCTS
from .utils import select_action
import multiprocessing

def _test(config, model, ep_i, device, render, save_video, save_path, ep_data):
    with torch.no_grad():
        env = config.new_game(save_video=save_video, save_path=save_path,
                              video_callable=lambda episode_id: True, uid=ep_i)
        done = False
        ep_reward = 0
        obs = env.reset()
        while not done:
            if render:
                env.render()
            root = CyphonNode(0)
            # obs = torch.FloatTensor(obs).to(device).unsqueeze(0)
            obs = tuple([torch.FloatTensor(_).to(device).unsqueeze(0) for _ in obs])
            root.expand(env.to_play(), env.legal_actions(), model.initial_inference(obs))
            MCTS(config).run(root, env.action_history(), model)
            action, _ = select_action(root, temperature=1, deterministic=True)
            obs, reward, done, __ = env.step(action)
            ep_reward += reward
        env.close()

    ep_data[ep_i] = ep_reward

def test(config, model, episodes, device, render, save_video=False):
    model.to(device)
    model.eval()
    save_path = os.path.join(config.exp_path, 'recordings')

    manager = multiprocessing.Manager()
    ep_data = manager.dict()
    jobs = []
    for ep_i in range(episodes):
        p = multiprocessing.Process(target=_test, args=(config, model, ep_i, device, render, save_video, save_path,
                                                        ep_data))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    test_reward = sum(ep_data.values())

    return test_reward / episodes
