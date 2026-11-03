from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BUTTON_NAMES = ('TURN_LEFT','TURN_RIGHT','MOVE_FORWARD','MOVE_LEFT','MOVE_RIGHT','ATTACK','USE')
TICS = 4
MAX_STEPS = 2500
NAV_BUDGET = 32
ENEMY_THRESHOLD = 0.65
ENEMY_RELEASE_THRESHOLD = 0.35
USE_INTERVAL = 16
DEFAULT_NAV_CHECKPOINT = ROOT / 'nav' / 'checkpoints' / 'hierarchical_room_qppo_v1' / 'latest.pt'
DEFAULT_COMBAT_CHECKPOINT = ROOT / 'corridor' / 'checkpoints' / 'corridor_quantum_combat_v1' / 'latest.pt'
