"""A* and Breadth-First Search (BFS) Pathfinding on a 2D Grid with Obstacles."""
from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from app.schemas.mission import Coordinate


def astar_shortest_path(
    start: Coordinate,
    goal: Coordinate,
    width: int,
    height: int,
    obstacles: Set[Coordinate],
) -> Optional[List[Coordinate]]:
    """Compute shortest obstacle-free path from `start` to `goal` using A* search.

    Returns:
        List of coordinates `[start, ..., goal]` inclusive if reachable, or `None` if blocked.
    """
    if start == goal:
        return [start]
    if not (0 <= goal[0] < width and 0 <= goal[1] < height) or goal in obstacles:
        return None

    def heuristic(a: Coordinate, b: Coordinate) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap: List[Tuple[int, int, Coordinate]] = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))

    came_from: Dict[Coordinate, Coordinate] = {}
    g_score: Dict[Coordinate, int] = {start: 0}
    closed: Set[Coordinate] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            path: List[Coordinate] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in closed:
            continue
        closed.add(current)

        cx, cy = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cx + dx, cy + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in obstacles or nxt in closed:
                continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(nxt, 1_000_000):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, nxt))

    return None


def compute_distance_map(
    sources: Iterable[Coordinate],
    width: int,
    height: int,
    obstacles: Set[Coordinate],
) -> Dict[Coordinate, int]:
    """Multi-source BFS returning shortest step distance from any source to every reachable cell."""
    dist_map: Dict[Coordinate, int] = {}
    queue: deque[Coordinate] = deque()

    for src in sources:
        s = (int(src[0]), int(src[1]))
        if 0 <= s[0] < width and 0 <= s[1] < height and s not in obstacles:
            dist_map[s] = 0
            queue.append(s)

    while queue:
        cx, cy = queue.popleft()
        d = dist_map[(cx, cy)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cx + dx, cy + dy)
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and nxt not in obstacles and nxt not in dist_map:
                dist_map[nxt] = d + 1
                queue.append(nxt)

    return dist_map
