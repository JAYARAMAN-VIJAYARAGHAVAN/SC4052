import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from .. import config as C

SHOOT_ACTION = 5
ENEMY_THRESHOLD = 0.5


def tensor_image(obs):
    return torch.as_tensor(obs['image'], dtype=torch.float32)


def mask_shoot(logits, enemy_visible):
    """Disable ATTACK for observations without a visible enemy."""
    masked = logits.clone()
    visible = torch.as_tensor(enemy_visible, device=logits.device, dtype=torch.bool)
    visible = visible.reshape(masked.shape[:-1])
    masked[..., SHOOT_ACTION] = masked[..., SHOOT_ACTION].masked_fill(~visible, -1e9)
    return masked


@torch.no_grad()
def policy_step(model, obs, deterministic=False, use_detector=False):
    logits, value, enemy_logit = model(tensor_image(obs))
    probability = torch.sigmoid(enemy_logit)
    if use_detector:
        visible = probability >= ENEMY_THRESHOLD
    else:
        visible = torch.as_tensor(obs['enemy_target'] >= 0.5, device=logits.device)
    logits = mask_shoot(logits, visible)
    dist = Categorical(logits=logits)
    action = logits.argmax(dim=-1) if deterministic else dist.sample()
    return (int(action.item()), float(dist.log_prob(action).item()),
            float(value.item()), float(probability.item()))


def update(model, optimizer, batch):
    rewards = np.asarray(batch['rewards'], np.float32)
    values = np.asarray(batch['values'], np.float32)
    next_values = np.asarray(batch['next_values'], np.float32)
    ends = np.asarray(batch['ends'], np.float32)
    terminated = np.asarray(batch['terminated'], np.float32)
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + C.GAMMA * (1 - terminated[t]) * next_values[t] - values[t]
        gae = delta + C.GAMMA * C.GAE_LAMBDA * (1 - ends[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    images = torch.as_tensor(np.stack(batch['images']), dtype=torch.float32)
    actions = torch.as_tensor(batch['actions'], dtype=torch.long)
    old_logps = torch.as_tensor(batch['logps'], dtype=torch.float32)
    returns = torch.as_tensor(returns, dtype=torch.float32)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    targets = torch.as_tensor(batch['enemy_targets'], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n = len(actions)
    metrics = []
    for _ in range(C.EPOCHS):
        for ids in torch.randperm(n).split(C.MINIBATCH):
            logits, v, detector = model(images[ids])
            logits = mask_shoot(logits, targets[ids] >= 0.5)
            dist = Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(actions[ids]) - old_logps[ids])
            pg = -torch.min(ratio * advantages[ids],
                            torch.clamp(ratio, 1-C.CLIP, 1+C.CLIP) * advantages[ids]).mean()
            vf = .5 * F.mse_loss(v, returns[ids])
            entropy = dist.entropy().mean()
            detect_loss = F.binary_cross_entropy_with_logits(detector, targets[ids])
            loss = pg + C.VALUE_COEF * vf - C.ENTROPY_COEF * entropy + C.DETECTOR_COEF * detect_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.MAX_GRAD_NORM)
            optimizer.step()
            metrics.append((float(pg.item()), float(vf.item()), float(entropy.item()), float(detect_loss.item())))
    return np.mean(metrics, axis=0)
