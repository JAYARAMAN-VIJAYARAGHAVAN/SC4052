import argparse
import collections
import time
import numpy as np
import torch
from . import config as C
from .environment import CorridorEnv
from .model.networks import QuantumCombatActorCritic
from .algo.ppo import policy_step

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--delay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=800)
    args = p.parse_args()
    torch.set_num_threads(1)
    model = QuantumCombatActorCritic()
    with torch.no_grad(): model(torch.zeros(1, C.FRAME_STACK, C.SCREEN_SIZE, C.SCREEN_SIZE))
    ckpt = torch.load(C.CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    print(f'Experiment={ckpt.get("experiment")} | trained steps={ckpt.get("steps")}')
    results = []; counts = collections.Counter()
    for episode in range(args.episodes):
        env = CorridorEnv(visible=not args.headless, seed=args.seed + episode)
        try:
            obs, _ = env.reset(); total_reward = 0.; kills = 0.; detections = []
            while True:
                action, _, _, confidence = policy_step(model, obs, args.deterministic)
                counts[C.ACTIONS[action]] += 1
                detections.append(confidence)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward; kills += info['kills']
                if args.delay: time.sleep(args.delay)
                if done or truncated: break
            results.append((kills, total_reward, info['dead']))
            print(f'Episode {episode+1} | kills={kills:.0f} | '
                  f'return={total_reward:+.2f} | dead={info["dead"]} | '
                  f'actions={info["steps"]} | mean enemy confidence={np.mean(detections):.2f}')
        finally: env.close()
    print(f'Mean kills: {np.mean([x[0] for x in results]):.2f}')
    print(f'Mean return: {np.mean([x[1] for x in results]):+.2f}')
    print('Action distribution:', dict(counts))

if __name__ == '__main__': main()
