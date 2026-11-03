"""E1M1 sector-option navigation with training-compatible observations.

A sector boundary is a candidate, not proof of a walkable passage.
"""
import math
from collections import Counter
import torch
from nav.room_graph import RoomGraph, point_in_poly


class Navigator:
    def __init__(self, state, game, model=None):
        sectors = getattr(state, 'sectors', None)
        self.graph = RoomGraph(list(sectors) if sectors is not None else [])
        if not self.graph.polygons or not self.graph.candidates:
            raise RuntimeError('No usable sector graph. Check ViZDoom sectors info / IWAD.')
        self.game, self.model = game, model
        self.room = None
        self.visits = Counter()
        self.cells = set()
        self.observe()

    def observe(self):
        x, y, _ = self.game.pose()
        new_room = self.graph.locate(x, y, self.room)
        if new_room is not None:
            if self.room is not None and new_room != self.room:
                self.graph.record_transition(self.room, new_room)
            self.room = new_room
            self.graph.observe(self.room)
            self.visits[self.room] += 1
        self.cells.add((int(x // 64), int(y // 64)))

    def choices(self):
        """Match nav/environment.py's candidate ranking and de-duplication."""
        if self.room is None:
            return []
        x, y, heading = self.game.pose()
        by_neighbor = {}
        for passage in self.graph.available(self.room):
            neighbor = passage.neighbor
            if neighbor == self.room:
                continue
            target = self.graph.crossing_target(self.room, passage)
            distance = math.dist((x, y), target)
            priority = (4.0 if neighbor not in self.graph.visited else 0.0) - (
                0.4 * min(self.visits[neighbor], 10)
                + 0.7 * min(self.graph.failures[(self.room, neighbor)], 10)
                + 0.002 * distance
            )
            existing = by_neighbor.get(neighbor)
            if existing is None or priority > existing[0]:
                by_neighbor[neighbor] = (priority, passage, distance)
        ranked = sorted(by_neighbor.values(), key=lambda item: (-item[0], item[2], item[1].neighbor))
        path = self.graph.path_to_frontier(self.room, self.visits)
        if len(path) > 1 and path[1] in by_neighbor:
            route = by_neighbor[path[1]]
            if route not in ranked[:4]:
                ranked.insert(0, route)
        choices = []
        seen = set()
        for _, passage, _ in ranked:
            if passage.neighbor not in seen:
                choices.append(passage)
                seen.add(passage.neighbor)
            if len(choices) == 4:
                break
        return choices

    def nav_observation(self, frames, nav_dim, action_count, choices=None):
        if nav_dim != 22 or action_count != 4:
            raise ValueError(f'Expected nav V2 22 features/4 options, got {nav_dim}/{action_count}')
        choices = self.choices() if choices is None else choices
        x, y, heading = self.game.pose()
        features = []
        for passage in choices:
            target = self.graph.crossing_target(self.room, passage)
            delta = math.atan2(target[1] - y, target[0] - x) - heading
            delta = math.atan2(math.sin(delta), math.cos(delta))
            features.extend([
                math.sin(delta), math.cos(delta),
                min(1.0, math.dist((x, y), target) / 1024.0),
                min(1.0, self.visits[passage.neighbor] / 10.0),
                min(1.0, self.graph.failures[(self.room, passage.neighbor)] / 10.0),
            ])
        features.extend([0.0] * (20 - len(features)))
        features.extend([
            min(1.0, self.visits[self.room] / 20.0),
            min(1.0, len(self.graph.visited) / 30.0),
        ])
        mask = [i < len(choices) for i in range(action_count)]
        if not any(mask):
            return None
        return {
            'image': frames.tensor(),
            'nav': torch.tensor([features], dtype=torch.float32),
            'mask': torch.tensor([mask], dtype=torch.bool),
        }

    def select(self, frames, nav_dim, action_count):
        choices = self.choices()
        if not choices:
            return None
        if self.model is None:
            return choices[0]
        obs = self.nav_observation(frames, nav_dim, action_count, choices)
        with torch.no_grad():
            logits, _ = self.model(obs)
        # Mask here too, in case the model doesn't mask internally.
        logits = logits.masked_fill(~obs['mask'], float('-inf'))
        return choices[int(logits.argmax(-1).item())]

    def passage_waypoints(self, passage, offset=24.0):
        """Approach -> boundary midpoint -> destination side.

        Offsets are tested against polygons; these waypoints are geometric
        guesses, not collision-checked navigation meshes.
        """
        a, b = passage.segment
        mx, my = passage.point
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return [passage.point]
        nx, ny = -dy / length, dx / length

        def inside(room):
            poly = self.graph.polygons.get(room, [])
            for distance in (offset, 16.0, 8.0, 4.0):
                for sign in (1.0, -1.0):
                    candidate = (mx + sign * nx * distance, my + sign * ny * distance)
                    if point_in_poly(*candidate, poly):
                        return candidate
            return None

        approach = inside(self.room)
        destination = inside(passage.neighbor)
        points = [p for p in (approach, (mx, my), destination) if p is not None]
        # Skip points already close to the player's current position.
        x, y, _ = self.game.pose()
        while len(points) > 1 and math.dist((x, y), points[0]) < 12.0:
            points.pop(0)
        return points

    def steering_action(self, target):
        x, y, heading = self.game.pose()
        delta = math.atan2(target[1] - y, target[0] - x) - heading
        delta = math.atan2(math.sin(delta), math.cos(delta))
        if abs(delta) > 0.18:
            return 'TURN_LEFT' if delta > 0 else 'TURN_RIGHT'
        return 'MOVE_FORWARD'
