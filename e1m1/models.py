"""Strict checkpoint loading; no silently initialized pretrained weights."""
from pathlib import Path
import torch

def load_combat(path):
    from corridor.model.networks import QuantumCombatActorCritic
    if not Path(path).is_file(): raise FileNotFoundError(f'Combat checkpoint missing: {path}')
    model = QuantumCombatActorCritic()
    with torch.no_grad(): model(torch.zeros(1,4,84,84))
    data = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(data['model'], strict=True)
    return model.eval(), data

def load_nav(path):
    from nav.model.networks import QuantumActorCritic
    if not Path(path).is_file(): raise FileNotFoundError(f'Navigation checkpoint missing: {path}')
    model = QuantumActorCritic()
    data = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(data['model'], strict=True)
    return model.eval(), data
