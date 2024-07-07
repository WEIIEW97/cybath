import heapq

def is_walkable(grid, x, y):
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0

def jump(grid, x, y, dx, dy):
    while is_walkable(grid, x + dx, y + dy):
        x += dx
        y += dy
        if (dx != 0 and (is_walkable(grid, x, y - 1) or is_walkable(grid, x, y + 1))) or \
           (dy != 0 and (is_walkable(grid, x - 1, y) or is_walkable(grid, x + 1, y))):
            return (x, y)
        if (dx != 0 and dy != 0 and (is_walkable(grid, x + dx, y) and is_walkable(grid, x, y + dy))):
            return (x, y)
    return None

def find_neighbors(grid, x, y):
    neighbors = []
    if is_walkable(grid, x - 1, y):
        neighbors.append((-1, 0))
    if is_walkable(grid, x + 1, y):
        neighbors.append((1, 0))
    if is_walkable(grid, x, y - 1):
        neighbors.append((0, -1))
    if is_walkable(grid, x, y + 1):
        neighbors.append((0, 1))
    return neighbors

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def jps(grid, start, end):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in find_neighbors(grid, current[0], current[1]):
            neighbor = jump(grid, current[0], current[1], dx, dy)
            if neighbor:
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

# Example usage:
grid = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]
start = (0, 0)
end = (4, 4)

path = jps(grid, start, end)
print("Path:", path)
