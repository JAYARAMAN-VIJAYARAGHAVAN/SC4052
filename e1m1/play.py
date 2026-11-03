"""E1M1 playback with navigation-only and hybrid modes.

Usage: uv run python -m e1m1.play --nav-only --deterministic
       uv run python -m e1m1.play --nav-only --heuristic-nav
"""
import argparse
import collections
import math
from pathlib import Path
import torch
from . import config as C
from .environment import E1M1Game
from .observations import Frames
from .models import load_combat, load_nav
from .navigation import Navigator
from .coordinator import Coordinator
from .controller import combat_buttons


class PassageFollower:
    """A stateful, conservative local waypoint follower with escape actions."""
    def __init__(self, navigator, game):
        self.nav = navigator
        self.game = game
        self.selected = None
        self.start_room = None
        self.points = []
        self.index = 0
        self.age = 0
        self.blocked = 0
        self.escape = []
        self.recoveries = 0
        self.last_action = None
        self.last_xy = game.pose()[:2]

    def clear(self):
        self.selected = None
        self.points = []
        self.index = 0
        self.age = 0
        self.blocked = 0
        self.escape = []
        self.last_action = None
        self.last_xy = self.game.pose()[:2]

    def fail(self):
        if self.selected is not None and self.start_room is not None:
            self.nav.graph.record_failure(self.start_room, self.selected.neighbor)
        self.clear()

    def select(self, frames, nav_dim, nav_count):
        self.selected = self.nav.select(frames, nav_dim, nav_count)
        self.start_room = self.nav.room
        self.points = self.nav.passage_waypoints(self.selected) if self.selected else []
        self.index = 0
        self.age = 0
        self.blocked = 0
        self.escape = []
        self.last_action = None
        self.last_xy = self.game.pose()[:2]

    def update_after_step(self, action):
        x, y = self.game.pose()[:2]
        moved = math.dist((x, y), self.last_xy)
        if action in ('MOVE_FORWARD', 'MOVE_LEFT', 'MOVE_RIGHT'):
            self.blocked = self.blocked + 1 if moved < 0.5 else 0
        elif moved >= 0.5:
            self.blocked = 0
        self.last_xy = (x, y)
        self.last_action = action
        self.age += 1

    def next_action(self, frames, nav_dim, nav_count, counts):
        if self.selected is not None and self.nav.room == self.selected.neighbor:
            self.clear()
        if self.selected is not None and self.age >= C.NAV_BUDGET:
            self.fail()
            counts['failed_passage'] += 1
        if self.selected is None:
            self.select(frames, nav_dim, nav_count)
        if self.selected is None or not self.points:
            return 'TURN_LEFT'

        x, y, _ = self.game.pose()
        while self.index < len(self.points) - 1 and math.dist((x, y), self.points[self.index]) < 12.0:
            self.index += 1
            counts['waypoint_reached'] += 1

        if self.escape:
            counts['recovery'] += 1
            return self.escape.pop(0)

        if self.blocked >= 3:
            self.recoveries += 1
            counts['stuck_events'] += 1
            if self.recoveries >= 3:
                self.fail()
                counts['failed_passage'] += 1
                return 'TURN_RIGHT'
            # Alternate lateral direction across attempts. Never require
            # MOVE_BACKWARD, which might not be enabled in the E1M1 config.
            side = 'MOVE_LEFT' if self.recoveries % 2 else 'MOVE_RIGHT'
            other = 'TURN_RIGHT' if self.recoveries % 2 else 'TURN_LEFT'
            self.escape = [side] * 4 + [other] * 4 + ['MOVE_FORWARD'] * 3
            self.blocked = 0
            counts['recovery'] += 1
            return self.escape.pop(0)

        return self.nav.steering_action(self.points[self.index])


def run(args):
    torch.set_num_threads(1)
    combat = combat_meta = None
    if not args.nav_only:
        combat, combat_meta = load_combat(args.combat_checkpoint)
        print(f'Combat checkpoint: {args.combat_checkpoint} (steps={combat_meta.get("steps", "?")})')
    nav_model = None
    nav_dim = nav_count = None
    if not args.heuristic_nav:
        nav_model, _ = load_nav(args.nav_checkpoint)
        nav_dim = nav_model.fusion[0].in_features - 128
        nav_count = nav_model.policy.out_features
        if (nav_dim, nav_count) != (22, 4):
            raise ValueError(f'Expected 22-feature/4-candidate nav V2; got {nav_dim}/{nav_count}')
    results = []
    for ep in range(args.episodes):
        game = E1M1Game(args.iwad, visible=not args.headless, seed=args.seed + ep)
        try:
            state = game.reset()
            nav_frames = Frames(96)
            nav_frames.update(state.screen_buffer, reset=True)
            combat_frames = None
            coordinator = None
            if not args.nav_only:
                combat_frames = Frames(84)
                combat_frames.update(state.screen_buffer, reset=True)
                coordinator = Coordinator()
            navigator = Navigator(state, game, nav_model)
            follower = PassageFollower(navigator, game)
            counts = collections.Counter()
            while game.steps < C.MAX_STEPS and not game.game.is_episode_finished():
                mode = 'navigation'
                if not args.nav_only:
                    with torch.no_grad():
                        logits, _, enemy_logit = combat(combat_frames.tensor())
                        confidence = float(torch.sigmoid(enemy_logit)[0].item())
                    mode = coordinator.decide(confidence, game.variable('AMMO2'))
                if mode == 'combat':
                    action_idx = (int(logits.argmax(-1).item()) if args.deterministic
                                  else int(torch.distributions.Categorical(logits=logits).sample().item()))
                    buttons = combat_buttons(action_idx)
                    counts['combat_' + str(action_idx)] += 1
                    follower.clear()
                else:
                    action = follower.next_action(nav_frames, nav_dim, nav_count, counts)
                    if action not in C.BUTTON_NAMES:
                        raise ValueError(f'{action} is missing from C.BUTTON_NAMES')
                    # Only use on an actual obstruction, rather than repeatedly
                    # activating USE while walking toward arbitrary geometry.
                    if action == 'MOVE_FORWARD' and follower.blocked >= 2 and 'USE' in C.BUTTON_NAMES:
                        action = 'USE'
                        counts['use'] += 1
                    buttons = (action,)
                    counts['navigation'] += 1
                state, _, done = game.step(*buttons)
                if state is not None and state.screen_buffer is not None:
                    nav_frames.update(state.screen_buffer)
                    if combat_frames is not None:
                        combat_frames.update(state.screen_buffer)
                navigator.observe()
                if mode == 'navigation':
                    follower.update_after_step(buttons[0])
                if done:
                    break
            result = {
                'episode': ep + 1,
                'steps': game.steps,
                'dead': bool(game.game.is_player_dead()),
                'engine_finished': bool(game.game.is_episode_finished()),
                'kills': game.variable('KILLCOUNT'),
                'health': game.variable('HEALTH'),
                'rooms': len(navigator.graph.visited),
                'cells': len(navigator.cells),
                'native_reward': game.native,
                'actions': dict(counts),
            }
            results.append(result)
            print(result, flush=True)
        finally:
            game.close()
    return results


def main():
    parser = argparse.ArgumentParser(description='E1M1 navigation and hybrid playback')
    parser.add_argument('--iwad', default=None)
    parser.add_argument('--combat-checkpoint', type=Path, default=C.DEFAULT_COMBAT_CHECKPOINT)
    parser.add_argument('--nav-checkpoint', type=Path, default=C.DEFAULT_NAV_CHECKPOINT)
    parser.add_argument('--nav-only', action='store_true', help='Do not load or run the combat model')
    parser.add_argument('--heuristic-nav', action='store_true', help='Use candidate-0 rather than the nav QNN')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=800)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
