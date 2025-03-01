import heapq
import time
import random
import os
from PIL import Image, ImageDraw

def generate_gridworld(size=101, blocked_prob=0.3, start=None, goal=None):
    grid = [['unvisited' for _ in range(size)] for _ in range(size)]
    
    if start is None:
        start_x, start_y = random.randint(0, size - 1), random.randint(0, size - 1)
    else:
        start_x, start_y = start

    stack = [(start_x, start_y)]
    grid[start_y][start_x] = 'unblocked'

    def get_neighbors(x, y):
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < size - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < size - 1:
            neighbors.append((x, y + 1))
        return neighbors

    while stack:
        x, y = stack[-1]
        neighbors = get_neighbors(x, y)
        unvisited_neighbors = [(nx, ny) for nx, ny in neighbors if grid[ny][nx] == 'unvisited']

        if unvisited_neighbors:
            nx, ny = random.choice(unvisited_neighbors)
            if random.random() < (1 - blocked_prob):
                grid[ny][nx] = 'unblocked'
            else:
                grid[ny][nx] = 'blocked'
            stack.append((nx, ny))
        else:
            stack.pop()

    for y in range(size):
        for x in range(size):
            if grid[y][x] == 'unvisited':
                grid[y][x] = 'blocked'

    if goal is None:
        while True:
            goal_x, goal_y = random.randint(0, size - 1), random.randint(0, size - 1)
            if grid[goal_y][goal_x] == 'unblocked' and (goal_x, goal_y) != (start_x, start_y):
                break
    else:
        goal_x, goal_y = goal

    return grid, (start_x, start_y), (goal_x, goal_y)

def visualize_gridworld(grid, start, goal, filename):
    size = len(grid)
    cell_size = 5 
    image_size = size * cell_size
    image = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(image)

    for y in range(size):
        for x in range(size):
            if grid[y][x] == 'blocked':
                draw.rectangle((x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size), fill='black')
            else:
                draw.rectangle((x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size), fill='white')

    start_x, start_y = start
    goal_x, goal_y = goal
    draw.rectangle((start_x * cell_size, start_y * cell_size, (start_x + 1) * cell_size, (start_y + 1) * cell_size), fill='red')
    draw.rectangle((goal_x * cell_size, goal_y * cell_size, (goal_x + 1) * cell_size, (goal_y + 1) * cell_size), fill='blue')
    
    for y in range(size):
        for x in range(size):
            draw.line((x * cell_size, 0, x * cell_size, image_size), fill='gray', width=1)
            draw.line((0, y * cell_size, image_size, y * cell_size), fill='gray', width=1)
    
    image.save(filename)

class BinaryHeap:
    def __init__(self):
        self.heap = []

    def push(self, item):
        heapq.heappush(self.heap, item)

    def pop(self):
        return heapq.heappop(self.heap)

    def empty(self):
        return not bool(self.heap)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(x, y):
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and grid[ny][nx] == 'unblocked':
            neighbors.append((nx, ny))
    return neighbors

def repeated_forward_a_star_low_g(grid, start, goal):
    return repeated_forward_a_star(grid, start, goal, favor_high_g=False)

def repeated_forward_a_star_high_g(grid, start, goal):
    return repeated_forward_a_star(grid, start, goal, favor_high_g=True)

def repeated_forward_a_star(grid, start, goal, favor_high_g):
    is_Path = False  

    def is_start_blocked(start, grid):
        x, y = start
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and grid[ny][nx] == 'unblocked':
                return False  
        return True

    def search(start_node, goal_node, known_blocks):
        open_list = BinaryHeap()
        open_list.push((0, 0, start_node, [])) 
        closed_set = set()
        g_values = {start_node: 0}
        expanded_cells = 0

        while not open_list.empty():
            f, g, current_node, path = open_list.pop()

            if current_node == goal_node:
                return path + [current_node], expanded_cells

            if current_node in closed_set:
                continue
            closed_set.add(current_node)
            expanded_cells += 1

            for neighbor in get_neighbors(current_node[0], current_node[1]):
                if neighbor in known_blocks:
                    continue

                new_g = g_values.get(current_node, float('inf')) + 1
                if new_g > g_values.get(neighbor, float('inf')):
                    continue 

                g_values[neighbor] = new_g
                h = heuristic(neighbor, goal_node)
                new_f = new_g + h

                c = len(grid) * len(grid[0])
                if favor_high_g:
                    priority = (c * new_f - new_g, new_g)
                else:
                    priority = (new_f, new_g)

                open_list.push((priority, new_g, neighbor, path + [current_node]))

        return None, expanded_cells

    if is_start_blocked(start, grid):
        return None, 0, is_Path

    known_blocks = set()
    current = start
    full_path = [current]

    while current != goal:
        path, expanded = search(current, goal, known_blocks)

        if not path:
            return None, expanded, is_Path 

        for next_node in path[1:]:
            neighbors = get_neighbors(current[0], current[1])
            for neighbor in neighbors:
                if grid[neighbor[1]][neighbor[0]] == 'blocked':
                    known_blocks.add(neighbor)

            if next_node in known_blocks:
                break  

            current = next_node
            full_path.append(current)

            if current == goal:
                is_Path = True
                return full_path, expanded, is_Path
            
    return None, 0, is_Path

def repeated_backward_a_star(grid, start, goal):
    is_Path = False

    def is_start_blocked(goal, grid):
        x, y = goal
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and grid[ny][nx] == 'unblocked':
                return False  
        return True

    def search(start_node, goal_node, known_blocks):
        open_list = BinaryHeap()
        open_list.push((0, 0, start_node, [])) 
        closed_set = set()
        g_values = {start_node: 0}
        expanded_cells = 0

        while not open_list.empty():
            f, g, current_node, path = open_list.pop()

            if current_node == goal_node:
                return path + [current_node], expanded_cells

            if current_node in closed_set:
                continue
            closed_set.add(current_node)
            expanded_cells += 1

            for neighbor in get_neighbors(current_node[0], current_node[1]):
                if neighbor in known_blocks:
                    continue

                new_g = g_values.get(current_node, float('inf')) + 1
                if new_g > g_values.get(neighbor, float('inf')):
                    continue 

                g_values[neighbor] = new_g
                h = heuristic(neighbor, goal_node)
                new_f = new_g + h

                c = len(grid) * len(grid[0])
                priority = (c * new_f - new_g, new_g)

                open_list.push((priority, new_g, neighbor, path + [current_node]))

        return None, expanded_cells
    
    if is_start_blocked(goal, grid):
        return None, 0, is_Path
    
    known_blocks = set()
    current = goal
    full_path = [current]

    while current != start:
        path, expanded = search(current, start, known_blocks)

        if not path:
            return None, expanded, is_Path 

        for next_node in path[1:]:
            neighbors = get_neighbors(current[0], current[1])
            for neighbor in neighbors:
                if grid[neighbor[1]][neighbor[0]] == 'blocked':
                    known_blocks.add(neighbor)

            if next_node in known_blocks:
                break  

            current = next_node
            full_path.append(current)

            if current == start:
                is_Path = True
                return full_path, expanded, is_Path

    return None, 0, is_Path


def adaptive_a_star(grid, start, goal):
    isPath = False

    def path(prev, curr):
        path = []
        while curr in prev:
            path.append(curr)
            curr = prev[curr]
        path.append(curr)
        
        return path[::-1]

    def is_start_blocked(start, grid):
        x, y = start
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and grid[ny][nx] == 'unblocked':
                return False  
        return True

    open_list = BinaryHeap()
    open_list.push((heuristic(start, goal), 0, start))
    closed_set = set()
    previous_position = {}
    g_values = {start: 0}
    h_values = {}

    if is_start_blocked(start, grid):
        return None, 0, isPath

    while not open_list.empty():
        f, g, current = open_list.pop()

        if current in closed_set:
            continue

        if current == goal:
            distance = g_values[current]
            for state in closed_set:
                h_values[state] = max(h_values.get(state, 0), distance - g_values[state])

            isPath = True
            return path(previous_position, current), len(closed_set), isPath

        closed_set.add(current)

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if (neighbor in closed_set or 
                not (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])) or grid[neighbor[0]][neighbor[1]] == 0):
                continue

            new_g = g + 1
            if neighbor not in g_values or new_g < g_values[neighbor]:
                g_values[neighbor] = new_g
                h = h_values.get(neighbor, heuristic(neighbor, goal))
                open_list.push((new_g + h, -new_g, neighbor))
                previous_position[neighbor] = current

    return None, len(closed_set), isPath

def draw_path_on_grid(grid, path, start, goal, filename, method):
    size = len(grid)
    cell_size = 5
    image_size = size * cell_size
    image = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(image)

    for y in range(size):
        for x in range(size):
            if grid[y][x] == 'blocked':
                draw.rectangle((x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size), fill='black')
            else:
                draw.rectangle((x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size), fill='white')


    for (px, py) in path:
        if(method == 1):
            draw.rectangle((px * cell_size, py * cell_size, (px + 1) * cell_size, (py + 1) * cell_size), fill='orange')
        if(method == 2):
            draw.rectangle((px * cell_size, py * cell_size, (px + 1) * cell_size, (py + 1) * cell_size), fill='yellow')
        if(method == 3):
            draw.rectangle((px * cell_size, py * cell_size, (px + 1) * cell_size, (py + 1) * cell_size), fill='green')
        if(method == 4):
            draw.rectangle((px * cell_size, py * cell_size, (px + 1) * cell_size, (py + 1) * cell_size), fill='purple')


    start_x, start_y = start
    goal_x, goal_y = goal
    draw.rectangle((start_x * cell_size, start_y * cell_size, (start_x + 1) * cell_size, (start_y + 1) * cell_size), fill='red')
    draw.rectangle((goal_x * cell_size, goal_y * cell_size, (goal_x + 1) * cell_size, (goal_y + 1) * cell_size), fill='blue')

    for y in range(size):
        for x in range(size):
            draw.line((x * cell_size, 0, x * cell_size, image_size), fill='gray', width=1)
            draw.line((0, y * cell_size, image_size, y * cell_size), fill='gray', width=1)

    image.save(filename)

folder_name = "Mazes"
forward_low_g_folder = "Forward_Low_G"
forward_high_g_folder = "Forward_High_G"
backward_folder = "Backward"
adaptive_folder = "Adaptive"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

if not os.path.exists(forward_low_g_folder):
    os.makedirs(forward_low_g_folder)

if not os.path.exists(forward_high_g_folder):
    os.makedirs(forward_high_g_folder)

if not os.path.exists(backward_folder):
    os.makedirs(backward_folder)

if not os.path.exists(adaptive_folder):
    os.makedirs(adaptive_folder)

total1_times = []
total2_times = []
total3_times = []
total4_times = []

total_exp1 = []
total_exp2 = []
total_exp3 = []
total_exp4 = []

for i in range(50):
    grid, start, goal = generate_gridworld()
    filename = os.path.join(folder_name, f"maze_{i}.png")
    visualize_gridworld(grid, start, goal, filename)
    
    start_time1 = time.time()
    path1, expanded1, istrue1 = repeated_forward_a_star_low_g(grid, start, goal)
    end_time1 = time.time()
    total1 = end_time1 - start_time1
    
    start_time2 = time.time()
    path2, expanded2, istrue2 = repeated_forward_a_star_high_g(grid, start, goal) 
    end_time2 = time.time()
    total2 = end_time2 - start_time2

    start_time3 = time.time()
    path3, expanded3, istrue3 = repeated_backward_a_star(grid, start, goal) 
    end_time3 = time.time()
    total3 = end_time3 - start_time3

    start_time4 = time.time()
    path4, expanded4, istrue4 = adaptive_a_star(grid, start, goal)
    end_time4 = time.time()
    total4 = end_time4 - start_time4
    
    total1_times.append(total1)
    total2_times.append(total2)
    total3_times.append(total3)
    total4_times.append(total4)

    total_exp1.append(expanded1)
    total_exp2.append(expanded2)
    total_exp3.append(expanded3)
    total_exp4.append(expanded4)
    
    print(f"\nMaze {i}: Start:{start} End:{goal}")
    print(f"Is There Path?: {istrue1}")
    print("Repeated Forward Expanded low-g:", expanded1)
    print("Runtime:", total1)
    if(istrue1):
        low_g = os.path.join(forward_low_g_folder, f"maze_{i}_path.png")
        draw_path_on_grid(grid, path1, start, goal, low_g, 1)

    print("Repeated Forward Expanded high-g:", expanded2)
    print("Runtime:", total2)
    if(istrue2):
        high_g = os.path.join(forward_high_g_folder, f"maze_{i}_path.png")
        draw_path_on_grid(grid, path2, start, goal, high_g, 2)

    print(f"Is There Path?: {istrue3}")
    print("Repeated Backward Expanded high-g:", expanded3)
    print("Runtime:", total3)
    if(istrue3):
        back = os.path.join(backward_folder, f"maze_{i}_path.png")
        draw_path_on_grid(grid, path3, start, goal, back, 3)
    
    print("Adaptive high-g:", expanded4)
    print("Runtime:", total4)
    if(istrue4):
        adapt = os.path.join(adaptive_folder, f"maze_{i}_path.png")
        draw_path_on_grid(grid, path4, start, goal, adapt, 4)

average_total1 = sum(total1_times) / len(total1_times)
average_total2 = sum(total2_times) / len(total2_times)
average_total3 = sum(total3_times) / len(total3_times)
average_total4 = sum(total4_times) / len(total4_times)

average_exp1 = sum(total_exp1) / len(total_exp1)
average_exp2 = sum(total_exp2) / len(total_exp2)
average_exp3 = sum(total_exp3) / len(total_exp3)
average_exp4 = sum(total_exp4) / len(total_exp4)

print(f"\nAverage Runtime for Repeated Forward A* Low-g: {average_total1}")
print(f"Average Runtime for Repeated Forward A* High-g: {average_total2}")
print(f"Average Expanded Cells for Repeated Forward A* Low-g: {average_exp1}")
print(f"Average Expanded Cells for Repeated Forward A* High-g: {average_exp2}")

print(f"\nAverage Runtime for Repeated Forward A* High-g: {average_total2}")
print(f"Average Runtime for Repeated Backward A* High-g: {average_total3}")
print(f"Average Expanded Cells for Repeated Forward A* High-g: {average_exp2}")
print(f"Average Expanded Cells for Repeated Backward A* High-g: {average_exp3}")

print(f"\nAverage Runtime for Repeated Forward A* High-g: {average_total2}")
print(f"Average Runtime for Adaptive A* High-g: {average_total4}")
print(f"Average Expanded Cells for Repeated Forward A* High-g: {average_exp2}")
print(f"Average Expanded Cells for Adaptive A* High-g: {average_exp4}")

print(f"\nGenerated and saved 50 Mazes in the '{folder_name}' folder.")