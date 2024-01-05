import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import heapq
import random


def read_pgm(path):
    with open(path, "rb") as pgmf:
        im = plt.imread(pgmf)
    return im


def heuristic(a, b):
    """Calculate the heuristic distance between two points (Euclidean distance)."""
    return distance.euclidean(a, b)


def astar(array, start, goal):
    """Perform A* algorithm on a grid."""
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_set = []

    heapq.heappush(open_set, (fscore[start], start))

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 0:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [
                i[1] for i in open_set
            ]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (fscore[neighbor], neighbor))

    return False


def nearest_node(tree, point):
    """Find the nearest node in the tree to the given point."""
    return min(tree, key=lambda x: distance.euclidean(x, point))


def is_clear_of_obstacles(point, clearance, obstacle_map):
    """Check if a point has the required clearance from obstacles."""
    y, x = point
    y_min, y_max = max(0, y - clearance), min(obstacle_map.shape[0], y + clearance + 1)
    x_min, x_max = max(0, x - clearance), min(obstacle_map.shape[1], x + clearance + 1)
    return np.all(obstacle_map[y_min:y_max, x_min:x_max] == 1)


def is_collision_free(new_point, nearest_point, obstacle_map, clearance):
    """Check if the path between two points is collision-free and maintains clearance."""
    y0, x0 = nearest_point
    y1, x1 = new_point
    if x0 == x1 or y0 == y1:
        return False
    y, x = np.ogrid[y0 : y1 : np.sign(y1 - y0) * 1, x0 : x1 : np.sign(x1 - x0) * 1]

    # Check if the line passes through any obstacles or too close to obstacles
    for i in range(y.shape[0]):
        for j in range(x.shape[1]):
            if not is_clear_of_obstacles((y[i, 0], x[0, j]), clearance, obstacle_map):
                return False
    return True


def rrt(obstacle_map, start, goal, clearance, max_iterations=10000):
    """RRT algorithm implementation."""
    tree = [start]
    for _ in range(max_iterations):
        # Sample a random point
        rand_point = (
            random.randint(0, obstacle_map.shape[0] - 1),
            random.randint(0, obstacle_map.shape[1] - 1),
        )

        # Find the nearest node in the tree
        nearest = nearest_node(tree, rand_point)

        # Check if the path between nearest and random point is collision-free
        if is_collision_free(rand_point, nearest, obstacle_map, clearance):
            tree.append(rand_point)

            # Check if we can connect to the goal from this new point
            if is_collision_free(goal, rand_point, obstacle_map, clearance):
                return tree, rand_point  # Path found

    return tree, None  # Path not found


if __name__ == "__main__":
    path = "/home/william/data/cybathlon/test.pgm"
    mat = read_pgm(path)
