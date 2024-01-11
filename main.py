import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

from scipy.spatial import distance
from scipy.ndimage import binary_dilation
from scipy.signal import convolve2d
from queue import Queue
from collections import deque

DIST_PER_GRID = 5


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


def is_collision_free_bresenham(new_point, nearest_point, obstacle_map, clearance):
    y0, x0 = nearest_point
    y1, x1 = new_point

    # Bresenham's line algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    while True:
        if not is_clear_of_obstacles((y, x), clearance, obstacle_map):
            return False
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= -dy:
            err -= dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
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


def cross_rectangle_conv(kernel_size, map):
    h_kernel = np.ones((1, kernel_size), dtype=np.float32)
    v_kernel = h_kernel.T

    hconv = convolve2d(map, h_kernel, mode='same', boundary='symm')
    hconv = hconv == kernel_size
    vconv = convolve2d(map, v_kernel, mode='same', boundary='symm')
    vconv = vconv == kernel_size

    return hconv * vconv


def wavefront_exploration(start, map):
    directions4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    directions8 = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    directions6 = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1)]
    height, width = map.shape
    visited = np.zeros_like(map, dtype=bool)
    path_map = np.full_like(map, -1, dtype=int)
    q = Queue()
    q.put(start)
    path_map[start] = 0

    # safe_walkable_area = get_safe_area(30, map)
    safe_walkable_area = cross_rectangle_conv(15, map)

    while not q.empty():
        x, y = q.get()
        for dx, dy in directions8:
            nx, ny = x + dx, y + dy

            if (
                0 <= nx < height
                and 0 <= ny < width
                and safe_walkable_area[nx, ny]
                and not visited[nx, ny]
            ):
                visited[nx, ny] = True
                path_map[nx, ny] = path_map[x, y] + 1
                q.put((nx, ny))

    return path_map


def is_safe2(point, map, visited, clearance=16):
    """Check if the point is within map bounds, clear of obstacles, and not visited."""
    x, y = point
    height, width = map.shape

    # Check map boundaries first to avoid out-of-bounds access
    if not (0 <= x < height and 0 <= y < width):
        return False

    # Check if the point is clear of obstacles and not visited
    is_all_clear = is_clear_of_obstacles(point, clearance, map)
    return is_all_clear and not visited[x, y]


def wfd(start, map):
    """Wavefront algorithm to create a path map from the start point using deque."""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    visited = np.zeros_like(map, dtype=bool)
    path_map = np.full_like(map, -1, dtype=int)
    q = deque()  # Using deque for the queue
    q.append(start)  # Use append() for deque
    visited[start] = True
    path_map[start] = 0

    while q:
        (
            x,
            y,
        ) = q.popleft()  # Use popleft() to get the element from the front of the deque
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_safe2((nx, ny), map, visited):
                visited[nx, ny] = True
                path_map[nx, ny] = path_map[x, y] + 1
                q.append((nx, ny))  # Use append() for deque

    return path_map


def is_safe(point, map, safe_dist=3):
    x, y = point
    height, width = map.shape
    return (
        0 <= x < height
        and 0 <= y < width
        and map[x, y]
        and all(
            map[
                max(0, x - dx) : min(height, x + dx + 1),
                max(0, y - dy) : min(width, y + dy + 1),
            ].all()
            for dx in range(safe_dist)
            for dy in range(safe_dist)
        )
    )


def rrt_star(start, map, iterations=1000, step_size=5, safe_dist=3, radius=10):
    height, width = map.shape
    tree = {start: (None, 0)}

    for _ in range(iterations):
        rand_point = (random.randint(0, height - 1), random.randint(0, width - 1))

        nearest_point = min(
            tree,
            key=lambda p: (p[0] - rand_point[0]) ** 2 + (p[1] - rand_point[1]) ** 2,
        )
        dx = rand_point[0] - nearest_point[0]
        dy = rand_point[1] - nearest_point[1]
        dist = max((dx**2 + dy**2) ** 0.5, 1)
        new_point = (
            int(nearest_point[0] + step_size * dx / dist),
            int(nearest_point[1] + step_size * dy / dist),
        )

        if not is_safe(new_point, map, safe_dist=safe_dist):
            continue

        min_dist = tree[nearest_point][1] + step_size
        min_parent = nearest_point
        for pt in tree:
            if (
                (pt[0] - new_point[0]) ** 2 + (pt[1] - new_point[1]) ** 2
            ) ** 0.5 <= radius:
                pt_dist = (
                    tree[pt][1]
                    + ((pt[0] - new_point[0]) ** 2 + (pt[1] - new_point[1]) ** 2) ** 0.5
                )
                if pt_dist < min_dist:
                    min_dist = pt_dist
                    min_parent = pt

        tree[new_point] = (min_parent, min_dist)

        if (
            new_point[0] == 0
            or new_point[0] == height - 1
            or new_point[1] == 0
            or new_point[1] == width - 1
        ):
            return tree, new_point

    return tree, None


def get_safe_area(clearance, grid_map, walkable_thr=254):
    clear_iter = int(clearance / DIST_PER_GRID)
    walkable_area = grid_map == walkable_thr
    obstacle_area = np.logical_not(walkable_area)

    safe_area = np.logical_not(binary_dilation(obstacle_area, iterations=clear_iter))
    return safe_area


def sconv(grip_map, walkable_thr=254):
    walkable_map = grip_map == walkable_thr
    special_kernel = np.ones((1, 16))
    conv_res = convolve2d(walkable_map, special_kernel, mode="full", boundary="symm")
    return conv_res == 16


if __name__ == "__main__":
    path = "test.pgm"
    mat = read_pgm(path)
    mat = np.flipud(mat)

    # mat = mat[:65, :]
    # conv_res = sconv(mat)
    # plt.figure()
    # plt.imshow(conv_res)
    # plt.show()

    # Define start and goal points
    start_point_updated = (5, 31)  # y, x as numpy arrays are row-major
    # goal_point_updated = (42, 34)  # y, x
    walkable_map = np.where(mat == 254, 1, 0)
    clearance_required = 3  # 16 pixels clearance
    # Apply RRT algorithm
    # tree, last_point = rrt(
    #     walkable_map, start_point_updated, goal_point_updated, clearance_required
    # )

    # # Visualize the result
    # plt.imshow(walkable_map, cmap="gray")
    # tree_y, tree_x = zip(*tree)
    # plt.plot(tree_x, tree_y, marker=".", color="red", markersize=2, linestyle="None")
    # plt.scatter(
    #     [start_point_updated[1], goal_point_updated[1]],
    #     [start_point_updated[0], goal_point_updated[0]],
    #     color="blue",
    # )

    # if last_point:
    #     plt.plot(
    #         [last_point[1], goal_point_updated[1]],
    #         [last_point[0], goal_point_updated[0]],
    #         color="green",
    #         linewidth=2,
    #     )

    # plt.title("RRT Path Planning")
    # plt.show()

    # # Indicate if the path to the goal was found
    # path_found = "Path to goal found." if last_point else "Path to goal not found."
    # print(path_found)



    wave_path = wavefront_exploration(start_point_updated, walkable_map)
    loc = np.where(wave_path == wave_path.max())
    print(loc)
    plt.figure()
    plt.imshow(walkable_map, cmap="gray")
    plt.plot(
        start_point_updated[1],
        start_point_updated[0],
        marker=".",
        color="blue",
        markersize=2,
    )
    plt.plot(*loc[1], *loc[0], marker=".", color="red", markersize=2)
    plt.show()

    import gc
    gc.collect()
