
"""Sector adjacency and confirmed traversal graph.

Candidate passages are hypotheses, NOT confirmed walkable edges. The policy sees
adjacent candidates; only physically traversed edges enter the confirmed graph.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
import math


def point_in_poly(x, y, vertices):
    if len(vertices) < 3:
        return False
    inside = False
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def endpoints(line):
    for names in [('x1', 'y1', 'x2', 'y2'), ('start_x', 'start_y', 'end_x', 'end_y')]:
        vals = [getattr(line, n, None) for n in names]
        if all(v is not None for v in vals):
            return tuple(float(v) for v in vals)
    for a, b in [('v1', 'v2'), ('vertex1', 'vertex2')]:
        va, vb = getattr(line, a, None), getattr(line, b, None)
        if va is not None and vb is not None:
            try:
                return float(va.x), float(va.y), float(vb.x), float(vb.y)
            except (AttributeError, TypeError):
                pass
    return None


def polygon_from_lines(lines):
    segments = [endpoints(line) for line in lines]
    segments = [s for s in segments if s is not None]
    if len(segments) < 3:
        return []
    coords = [((s[0], s[1]), (s[2], s[3])) for s in segments]
    used = {0}
    polygon = [coords[0][0], coords[0][1]]
    while len(used) < len(coords):
        tail = polygon[-1]
        match = None
        for i, (a, b) in enumerate(coords):
            if i in used:
                continue
            if math.dist(tail, a) < 1e-3:
                match = (i, b)
                break
            if math.dist(tail, b) < 1e-3:
                match = (i, a)
                break
        if match is None:
            return []
        used.add(match[0])
        polygon.append(match[1])
    if math.dist(polygon[-1], polygon[0]) > 1e-3:
        return []
    return polygon[:-1]


def edge_key(a, b):
    return tuple(sorted(((round(a[0], 2), round(a[1], 2)),
                         (round(b[0], 2), round(b[1], 2)))))


@dataclass(frozen=True)
class Passage:
    neighbor: int
    point: tuple  # midpoint of shared line, for orientation only
    segment: tuple  # endpoints of the shared boundary


class RoomGraph:
    def __init__(self, sectors):
        self.polygons = {}
        shared = defaultdict(list)
        for i, sector in enumerate(sectors):
            lines = list(getattr(sector, 'lines', ()) or ())
            polygon = polygon_from_lines(lines)
            if polygon:
                self.polygons[i] = polygon
            for line in lines:
                coords = endpoints(line)
                if coords:
                    a, b = coords[:2], coords[2:]
                    shared[edge_key(a, b)].append((i, a, b))
        self.candidates = defaultdict(list)
        for entries in shared.values():
            ids = {item[0] for item in entries}
            if len(ids) != 2:
                continue
            a, b = sorted(ids)
            _, p, q = entries[0]
            midpoint = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            self.candidates[a].append(Passage(b, midpoint, (p, q)))
            self.candidates[b].append(Passage(a, midpoint, (p, q)))
        self.known = defaultdict(set)
        self.visited = set()
        self.failures = defaultdict(int)

    def locate(self, x, y, previous=None):
        if previous is not None and point_in_poly(x, y, self.polygons.get(previous, [])):
            return previous
        for i, poly in self.polygons.items():
            if point_in_poly(x, y, poly):
                return i
        return None

    def observe(self, room):
        if room is not None:
            self.visited.add(room)

    def record_transition(self, a, b):
        if a is not None and b is not None and a != b:
            self.known[a].add(b)
            self.known[b].add(a)
            self.failures[(a, b)] = 0
            self.failures[(b, a)] = 0
            self.visited.update((a, b))

    def record_failure(self, a, b):
        self.failures[(a, b)] += 1

    def available(self, room):
        # Never permanently delete an edge due to a steering failure.
        return list(self.candidates.get(room, ()))

    def crossing_target(self, room, passage, offset=28.0):
        """Aim beyond a boundary, inside the destination polygon if possible.

        This is a geometric waypoint, not proof that the line is traversable.
        """
        a, b = passage.segment
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return passage.point
        nx, ny = -dy / length, dx / length
        mx, my = passage.point
        for distance in (offset, 16., 8., 4.):
            for sign in (1, -1):
                x, y = mx + sign * distance * nx, my + sign * distance * ny
                if point_in_poly(x, y, self.polygons.get(passage.neighbor, [])):
                    return x, y
        return passage.point

    def path_to_frontier(self, start, visits=None):
        """BFS uses only previously physically confirmed connections."""
        if start is None:
            return []
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            room, path = queue.popleft()
            if room != start and any(p.neighbor not in self.visited for p in self.available(room)):
                return path
            for nxt in self.known[room]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return []
