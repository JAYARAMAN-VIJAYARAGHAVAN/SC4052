
"""Semi-Markov room-selection environment with privileged geometry.

Caveat: ViZDoom's sector geometry API varies by version. We validate geometry
at startup and fail loudly rather than silently train on fake room IDs.
"""
import math
from pathlib import Path
import numpy as np
import vizdoom as vzd
from . import config as C
from .memory import ExplorationMemory
from .room_graph import RoomGraph
from .controller import SteeringController, BUTTONS
from .preprocessing import FrameStack

OPTION_NAMES = tuple(f'CANDIDATE_{i}' for i in range(C.N_OPTIONS))


class RoomNavigationEnv:
    def __init__(self, visible=False, seed=C.SEED):
        self.game = vzd.DoomGame()
        scenario = Path(vzd.__file__).resolve().parent / 'scenarios' / C.ENV_SCENARIO
        if not scenario.exists():
            raise FileNotFoundError(scenario)
        self.game.load_config(str(scenario))
        self.game.set_window_visible(visible)
        self.game.set_seed(int(seed))
        self.game.set_available_buttons(BUTTONS)
        self.game.set_available_game_variables([
            vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.ANGLE, vzd.GameVariable.VELOCITY_X,
            vzd.GameVariable.VELOCITY_Y
        ])
        self.game.set_depth_buffer_enabled(True)
        self.game.set_sectors_info_enabled(True)
        self.game.init()
        self.controller = SteeringController(self.game)
        self.stack = FrameStack()
        self.actions = 0
        self.options = 0
        self.current_room = None

    def _pose(self):
        return self.controller.pose()

    def _game_state(self):
        return self.game.get_state()

    def _observe_step(self, action, native):
        x, y, _ = self._pose()
        room = self.graph.locate(x, y, self.current_room)
        old = self.current_room
        if room is not None:
            self.current_room = room
        if old is not None and self.current_room != old:
            self.graph.record_transition(old, self.current_room)
        new_cell, new_room = self.memory.observe(x, y, self.current_room)
        self.graph.observe(self.current_room)
        self.actions += 1
        state = self._game_state()
        if state is not None and state.screen_buffer is not None:
            self.stack.append(state.screen_buffer)
        return {
            'room': self.current_room,
            'new_cell': new_cell,
            'new_room': new_room,
            'native': native,
            'action': action
        }


    def _choices(self):
        """Generate up to four distinct candidate sector passages.

        The classical system proposes destinations.
        The quantum policy chooses between them.

        A shared boundary is only a possible passage, not proof
        that it is traversable.
        """
        x, y, heading = self._pose()
        room = self.current_room

        passages = self.graph.available(room)
        candidates = {}

        for passage in passages:
            neighbor = passage.neighbor

            if neighbor == room:
                continue

            target = self.graph.crossing_target(room, passage)
            distance = math.dist((x, y), target)

            theta = math.atan2(
                target[1] - y,
                target[0] - x
            )
            delta = math.atan2(
                math.sin(theta - heading),
                math.cos(theta - heading)
            )

            visits = self.memory.room_count(neighbor)
            failures = self.graph.failures[(room, neighbor)]
            unexplored = neighbor not in self.graph.visited

            # Ranking is only for selecting the candidate set.
            # The QNN makes the final decision.
            priority = (
                4.0 if unexplored else 0.0
            ) - (
                0.4 * min(visits, 10)
                + 0.7 * min(failures, 10)
                + 0.002 * distance
            )

            # Keep the best passage for each neighboring sector.
            existing = candidates.get(neighbor)

            if existing is None or priority > existing[0]:
                candidates[neighbor] = (
                    priority,
                    passage,
                    delta,
                    distance
                )

        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                -item[0],
                item[3],
                item[1].neighbor
            )
        )

        # A confirmed route toward an unexplored frontier can
        # also be represented by its first passage.
        path = self.graph.path_to_frontier(
            room, self.memory.rooms
        )

        if len(path) > 1:
            next_room = path[1]

            # Make sure the route's first hop is considered.
            if next_room in candidates:
                route = candidates[next_room]
                if route not in ranked[:C.N_OPTIONS]:
                    ranked.insert(0, route)

        selected = []
        used_neighbors = set()

        for _, passage, _, _ in ranked:
            if passage.neighbor in used_neighbors:
                continue

            used_neighbors.add(passage.neighbor)
            selected.append(passage)

            if len(selected) == C.N_OPTIONS:
                break

        while len(selected) < C.N_OPTIONS:
            selected.append(None)

        return selected

    def _observation(self):
        x, y, heading = self._pose()
        choices = self._choices()

        features = []

        for passage in choices:
            if passage is None:
                features.extend([0.] * 5)
                continue

            neighbor = passage.neighbor
            target = self.graph.crossing_target(
                self.current_room, passage
            )

            dx = target[0] - x
            dy = target[1] - y
            distance = math.hypot(dx, dy)

            theta = math.atan2(dy, dx)
            delta = math.atan2(
                math.sin(theta - heading),
                math.cos(theta - heading)
            )

            visits = self.memory.room_count(neighbor)
            failures = self.graph.failures[
                (self.current_room, neighbor)
            ]

            features.extend([
                math.sin(delta),
                math.cos(delta),
                min(1.0, distance / 1024.0),
                min(1.0, visits / 10.0),
                min(1.0, failures / 10.0)
            ])

        # Global exploration context.
        features.extend([
            min(
                1.0,
                self.memory.room_count(self.current_room) / 20.0
            ),
            min(1.0, len(self.memory.rooms) / 30.0)
        ])

        mask = np.array(
            [p is not None for p in choices],
            dtype=np.bool_
        )

        if not mask.any():
            raise RuntimeError(
                'No usable sector passages. '
                'Check sector geometry.'
            )

        nav = np.asarray(features, dtype=np.float32)

        assert nav.shape == (C.N_NAV_FEATURES,), (
            f'Expected {C.N_NAV_FEATURES} features, '
            f'got {nav.shape}'
        )

        return {
            'image': self.stack.value(),
            'nav': nav,
            'mask': mask
        }

    def reset(self):
        self.game.new_episode()
        state = self._game_state()
        if state is None:
            raise RuntimeError('ViZDoom returned no state at reset')

        self.graph = RoomGraph(
            list(getattr(state, 'sectors', ()) or ())
        )
        if not self.graph.polygons or not self.graph.candidates:
            raise RuntimeError(
                'Cannot build sector graph from ViZDoom geometry. '
                'Inspect state.sectors and line endpoint attributes.'
            )

        self.memory = ExplorationMemory()
        self.actions = 0
        self.options = 0
        x, y, _ = self._pose()
        self.current_room = self.graph.locate(x, y)
        if self.current_room is None:
            raise RuntimeError(
                'Cannot locate starting sector in geometry polygons'
            )
        self.memory.observe(x, y, self.current_room)
        self.graph.observe(self.current_room)
        self.stack.reset(state.screen_buffer)
        return self._observation(), {'room': self.current_room}

    def step(self, option):
        choices = self._choices()
        if option < 0 or option >= len(choices) or choices[option] is None:
            raise ValueError('Invalid/masked room option')

        chosen = choices[option]
        start = self.current_room
        remaining = C.MAX_GAME_ACTIONS - self.actions
        budget = min(C.OPTION_MAX_ACTIONS, remaining)
        self.options += 1

        def reached(info):
            return info['room'] == chosen.neighbor

        waypoint = self.graph.crossing_target(start, chosen)
        outcome, native, trace = self.controller.navigate(
            waypoint, self._observe_step, budget, reached
        )

        new_cells = sum(t['new_cell'] for t in trace)
        new_rooms = sum(t['new_room'] for t in trace)
        arrived = self.current_room == chosen.neighbor

        if not arrived and not self.game.is_episode_finished():
            self.graph.record_failure(start, chosen.neighbor)

        finished = bool(self.game.is_episode_finished())
        dead = bool(self.game.is_player_dead())
        native_total = float(self.game.get_total_reward())

        # Verify this success condition against the scenario configuration.
        success = finished and not dead and native_total > 0

        reward = (
            C.NEW_ROOM_REWARD * new_rooms
            + C.NEW_CELL_REWARD * new_cells
            + C.OPTION_TIME_PENALTY * len(trace)
            + (C.REVISIT_REWARD if arrived and not new_rooms else 0.)
            + (C.FAILED_OPTION_PENALTY
               if not arrived and not finished else 0.)
            + (C.SUCCESS_REWARD if success else 0.)
        )

        terminated = finished
        truncated = (
            not finished and (
                self.actions >= C.MAX_GAME_ACTIONS
                or self.options >= C.MAX_OPTIONS
            )
        )

        info = {
            'success': success,
            'native_total': native_total,
            'rooms': len(self.memory.rooms),
            'cells': len(self.memory.cells),
            'duration': len(trace),
            'arrived': arrived,
            'option': OPTION_NAMES[option],
            'destination': chosen.neighbor,
            'room': self.current_room,
            'game_actions': self.actions,
            'controller_outcome': outcome,
            'end_reason': None
        }

        # Do not truncate because a steering attempt failed.
        # Candidate passages remain retryable.
        if terminated:
            info['end_reason'] = (
                'game_dead' if dead else 'game_finished'
            )
        elif truncated:
            info['end_reason'] = (
                'action_limit'
                if self.actions >= C.MAX_GAME_ACTIONS
                else 'option_limit'
            )
        elif not any(self._choices()):
            truncated = True
            info['end_reason'] = 'no_sector_candidates'

        if terminated or truncated:
            # Terminal observations must remain valid for the value network.
            try:
                obs = self._observation()
            except RuntimeError:
                obs = {
                    'image': self.stack.value(),
                    'nav': np.zeros(C.N_NAV_FEATURES, np.float32),
                    'mask': np.ones(C.N_OPTIONS, np.bool_)
                }
        else:
            obs = self._observation()

        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.game.close()
