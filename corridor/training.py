import argparse
import random
import numpy as np
import torch
from . import config as C
from .environment import CorridorEnv
from .model.networks import QuantumCombatActorCritic
from .algo.ppo import policy_step, tensor_image, update

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=C.TOTAL_STEPS)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    torch.set_num_threads(1)
    C.CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    model = QuantumCombatActorCritic()
    with torch.no_grad(): model(torch.zeros(1, C.FRAME_STACK, C.SCREEN_SIZE, C.SCREEN_SIZE))
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LR)
    total = 0
    if args.resume:
        ckpt = torch.load(C.CHECKPOINT, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        total = int(ckpt['steps'])
    elif C.CHECKPOINT.exists():
        raise FileExistsError(f'{C.CHECKPOINT} already exists; use --resume')
    env = CorridorEnv()
    try:
        obs, _ = env.reset(); episode = 0; episode_return = 0.0; episode_kills = 0
        while total < args.steps:
            batch = {k: [] for k in ('images', 'enemy_targets', 'actions', 'logps',
                'values', 'rewards', 'next_values', 'ends', 'terminated')}
            for _ in range(min(C.ROLLOUT_STEPS, args.steps-total)):
                action, logp, value, detection = policy_step(model, obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                with torch.no_grad():
                    _, next_value, _ = model(tensor_image(next_obs))
                for k, v in (
                    ('images', obs['image']), ('enemy_targets', obs['enemy_target']),
                    ('actions', action), ('logps', logp), ('values', value),
                    ('rewards', reward), ('next_values', float(next_value.item())),
                    ('ends', float(terminated or truncated)),
                    ('terminated', float(terminated))):
                    batch[k].append(v)
                total += 1; episode_return += reward; episode_kills += info['kills']
                obs = next_obs
                if terminated or truncated:
                    episode += 1
                    print(f'Episode {episode} | steps={info["steps"]} | '
                          f'kills={episode_kills:.0f} | dead={info["dead"]} | '
                          f'return={episode_return:+.2f}', flush=True)
                    obs, _ = env.reset(); episode_return = 0.0; episode_kills = 0
            pg, vf, ent, det = update(model, optimizer, batch)
            print(f'Update {total}/{args.steps} | policy={pg:+.4f} '
                  f'value={vf:.4f} entropy={ent:.4f} detector={det:.4f}', flush=True)
            if total % C.SAVE_EVERY == 0 or total >= args.steps:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'steps': total, 'experiment': C.EXPERIMENT}, C.CHECKPOINT)
                print(f'Saved {C.CHECKPOINT}', flush=True)
    finally: env.close()

if __name__ == '__main__': main()
