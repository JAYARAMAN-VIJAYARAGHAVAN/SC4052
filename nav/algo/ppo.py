"""Semi-Markov PPO: discount by the duration of each room-navigation option."""
import numpy as np
import torch
from torch.distributions import Categorical
from .. import config as C

def tensor_obs(obs):return {k:torch.as_tensor(v) for k,v in obs.items()}

def policy_step(model,obs,deterministic=False):
    with torch.no_grad():
        logits,value=model(tensor_obs(obs));dist=Categorical(logits=logits)
        action=logits.argmax(-1) if deterministic else dist.sample()
        return int(action.item()),float(dist.log_prob(action).item()),float(value.item())

def update(model,optimizer,batch):
    rewards=np.asarray(batch['rewards'],np.float32)
    values=np.asarray(batch['values'],np.float32)
    next_values=np.asarray(batch['next_values'],np.float32)
    terminated=np.asarray(batch['terminated'],np.float32)
    ends=np.asarray(batch['episode_ends'],np.float32)
    durations=np.asarray(batch['durations'],np.float32)
    advantages=np.zeros_like(rewards);gae=0.
    for t in reversed(range(len(rewards))):
        gamma=C.GAMMA**durations[t]
        delta=rewards[t]+gamma*next_values[t]*(1-terminated[t])-values[t]
        gae=delta+gamma*C.GAE_LAMBDA*(1-ends[t])*gae
        advantages[t]=gae
    returns=torch.from_numpy(advantages+values)
    adv=torch.from_numpy(advantages)
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    old_logp=torch.tensor(batch['logps'],dtype=torch.float32)
    actions=torch.tensor(batch['actions'],dtype=torch.long)
    stats=[];indices=np.arange(len(rewards))
    for _ in range(C.UPDATE_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0,len(indices),C.MINIBATCH_SIZE):
            idx=indices[start:start+C.MINIBATCH_SIZE]
            obs={k:torch.as_tensor(np.stack([batch['obs'][j][k] for j in idx]))
                 for k in ('image','nav','mask')}
            logits,val=model(obs);dist=Categorical(logits=logits)
            logp=dist.log_prob(actions[idx]);ratio=(logp-old_logp[idx]).exp()
            pg=-torch.minimum(ratio*adv[idx],ratio.clamp(1-C.CLIP_EPS,1+C.CLIP_EPS)*adv[idx]).mean()
            vf=(val-returns[idx]).square().mean();ent=dist.entropy().mean()
            loss=pg+C.VALUE_COEF*vf-C.ENTROPY_COEF*ent
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),C.MAX_GRAD_NORM)
            optimizer.step()
            stats.append((pg.item(),vf.item(),ent.item()))
    return np.mean(stats,axis=0)
