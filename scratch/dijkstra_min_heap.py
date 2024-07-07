import heapq

def dijkstra(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    # Min-heap priority queue
    priority_queue = [(0, start)]
    distances = {start: 0}
    previous_nodes = {start: None}
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_node == goal:
            return reconstruct_path_dijkstra(previous_nodes, goal)
        
        for dx, dy in directions:
            neighbor = (current_node[0] + dx, current_node[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                distance = current_distance + 1  # All edges have the same weight (1)
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))
    
    return None

def reconstruct_path_dijkstra(previous_nodes, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    return path[::-1]

# Example usage:
grid = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]
start = (0, 0)
goal = (4, 4)

path = dijkstra(grid, start, goal)
print("Path:", path)
