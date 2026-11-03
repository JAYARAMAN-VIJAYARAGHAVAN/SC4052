"""Single shared E1M1 DoomGame. Both policies must act on this same instance."""
import math
from pathlib import Path
import vizdoom as vzd
from . import config as C

class E1M1Game:
    def __init__(self, iwad=None, visible=False, seed=800):
        # Match the full-game loader: default to ViZDoom's Freedoom Phase 1 IWAD.
        default = Path(vzd.install_path) / 'freedoom1.wad'
        self.iwad = Path(iwad).expanduser() if iwad else default
        if not self.iwad.is_file():
            raise FileNotFoundError(
                f'E1M1 IWAD not found: {self.iwad}. '
                'Install Freedoom Phase 1 or supply --iwad /path/to/DOOM.WAD'
            )
        self.game = vzd.DoomGame()
        # The bundled config establishes full-game defaults; our explicit
        # settings below preserve the existing nav/combat action interface.
        cfg = Path(vzd.scenarios_path) / 'freedoom1.cfg'
        if cfg.is_file():
            self.game.load_config(str(cfg))
        self.game.set_doom_game_path(str(self.iwad.resolve()))
        self.game.set_doom_map('E1M1')
        self.game.set_window_visible(visible)
        self.game.set_seed(seed)
        self.buttons = [getattr(vzd.Button, x) for x in C.BUTTON_NAMES]
        self.game.set_available_buttons(self.buttons)
        self.game.set_available_game_variables([
            vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.ANGLE, vzd.GameVariable.HEALTH,
            vzd.GameVariable.KILLCOUNT, vzd.GameVariable.AMMO2,
        ])
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_sectors_info_enabled(True)
        self.game.set_episode_timeout(C.MAX_STEPS * C.TICS)
        self.game.init()
        self.steps = 0
        self.native = 0.0

    def reset(self):
        self.game.new_episode()
        self.steps = 0
        self.native = 0.0
        state = self.game.get_state()
        if state is None or state.screen_buffer is None:
            raise RuntimeError('No initial ViZDoom observation')
        return state

    def variable(self, name):
        return float(self.game.get_game_variable(getattr(vzd.GameVariable, name)))

    def pose(self):
        return (self.variable('POSITION_X'), self.variable('POSITION_Y'),
                math.radians(self.variable('ANGLE') % 360))

    def step(self, *active):
        if self.game.is_episode_finished() or self.steps >= C.MAX_STEPS:
            return None, 0.0, True
        buttons = [int(name in active) for name in C.BUTTON_NAMES]
        reward = float(self.game.make_action(buttons, C.TICS))
        self.native += reward
        self.steps += 1
        done = self.game.is_episode_finished() or self.steps >= C.MAX_STEPS
        return (None if self.game.is_episode_finished() else self.game.get_state()), reward, done

    def close(self):
        self.game.close()
