import argparse,random
import numpy as np
import torch
from . import config as C
from .environment import RoomNavigationEnv
from .model.networks import QuantumActorCritic
from .algo.ppo import policy_step,tensor_obs,update

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=C.TOTAL_TIMESTEPS,help='High-level decisions')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    random.seed(C.SEED);np.random.seed(C.SEED);torch.manual_seed(C.SEED);torch.set_num_threads(1)
    C.CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    path=C.CHECKPOINT_DIR/'latest.pt';model=QuantumActorCritic()
    with torch.no_grad():model(tensor_obs({'image':np.zeros((C.FRAME_STACK,C.SCREEN_SIZE,C.SCREEN_SIZE),np.float32),
        'nav':np.zeros(C.N_NAV_FEATURES,np.float32),'mask':np.ones(C.N_OPTIONS,np.bool_)}))
    optimizer=torch.optim.Adam(model.parameters(),lr=C.LEARNING_RATE)
    total=0
    if args.resume:
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer']);total=int(ckpt['steps'])
    elif path.exists():raise FileExistsError(f'{path} exists; use --resume')
    env=RoomNavigationEnv()
    try:
        obs,_=env.reset();ep=0;ep_return=0.
        while total<args.steps:
            batch={k:[] for k in ('obs','actions','logps','values','rewards','durations',
                'terminated','episode_ends','next_values')}
            for _ in range(min(C.ROLLOUT_STEPS,args.steps-total)):
                action,logp,value=policy_step(model,obs)
                nxt,reward,terminated,truncated,info=env.step(action)
                with torch.no_grad():_,nv=model(tensor_obs(nxt))
                for k,v in [('obs',obs),('actions',action),('logps',logp),('values',value),
                    ('rewards',reward),('durations',max(1,info['duration'])),('terminated',float(terminated)),
                    ('episode_ends',float(terminated or truncated)),('next_values',float(nv.item()))]:batch[k].append(v)
                total+=1;ep_return+=reward;obs=nxt
                if terminated or truncated:
                    ep+=1;print(f'Episode {ep} | options={env.options} game_actions={info["game_actions"]} '
                        f'rooms={info["rooms"]} cells={info["cells"]} success={info["success"]} '
                        f'return={ep_return:+.2f}',flush=True)
                    obs,_=env.reset();ep_return=0.
            pg,vf,ent=update(model,optimizer,batch)
            print(f'Update {total}/{args.steps} | policy={pg:+.4f} value={vf:.4f} entropy={ent:.4f}',flush=True)
            if total%C.CHECKPOINT_INTERVAL==0 or total>=args.steps:
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'steps':total,'experiment':C.EXPERIMENT_NAME},path)
                print(f'Saved {path}',flush=True)
    finally:env.close()
if __name__=='__main__':main()
