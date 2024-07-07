from collections import deque

def bidirectional_search(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    # Initialize the forward and backward queues
    forward_queue = deque([start])
    backward_queue = deque([goal])
    forward_visited = {start: None}
    backward_visited = {goal: None}
    
    while forward_queue and backward_queue:
        # Expand forward search
        if forward_queue:
            current = forward_queue.popleft()
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                    if neighbor in backward_visited:
                        return reconstruct_path(forward_visited, backward_visited, neighbor)
                    if neighbor not in forward_visited:
                        forward_queue.append(neighbor)
                        forward_visited[neighbor] = current
        
        # Expand backward search
        if backward_queue:
            current = backward_queue.popleft()
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                    if neighbor in forward_visited:
                        return reconstruct_path(forward_visited, backward_visited, neighbor)
                    if neighbor not in backward_visited:
                        backward_queue.append(neighbor)
                        backward_visited[neighbor] = current
    
    return None

def reconstruct_path(forward_visited, backward_visited, meeting_point):
    path = []
    # Reconstruct forward path
    current = meeting_point
    while current:
        path.append(current)
        current = forward_visited[current]
    path = path[::-1]  # Reverse the forward path
    
    # Reconstruct backward path
    current = backward_visited[meeting_point]
    while current:
        path.append(current)
        current = backward_visited[current]
    
    return path

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

path = bidirectional_search(grid, start, goal)
print("Path:", path)
