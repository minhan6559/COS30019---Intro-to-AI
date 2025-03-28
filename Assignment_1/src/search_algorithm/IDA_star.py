import math
import reader

def heuristic(node, goal, graph, nodes_coordinates):
    # Heuristic: Euclidean distance between node and goal
    x1, y1 = nodes_coordinates[node]
    x2, y2 = nodes_coordinates[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def ida_star(graph, start, goals, nodes_coordinates, max_depth=10):
    def search(node, g, bound, path, goal):
        f = g + heuristic(node, goal, graph, nodes_coordinates)
        if f > bound:
            return f
        if node == goal:
            return path  # Found the goal, return the path
        min_bound = float('inf')
        
        for neighbor, cost in graph[node]:  # Changed this line to iterate over the list
            if neighbor not in path:  # Avoid cycles
                path.append(neighbor)
                temp = search(neighbor, g + cost, bound, path, goal)
                if isinstance(temp, list):  # Goal found, return path
                    return temp
                if temp < min_bound:
                    min_bound = temp
                path.pop()
        
        return min_bound
    
    # Iterate over all destination goals
    all_paths = {}
    for goal in goals:
        bound = heuristic(start, goal, graph, nodes_coordinates)
        path = [start]
        while True:
            result = search(start, 0, bound, path, goal)
            if isinstance(result, list):  # Goal found
                all_paths[goal] = result
                break
            if result == float('inf'):  # No path found
                all_paths[goal] = None
                break
            bound = result

    return all_paths

nodes, graph, origin, destinations = reader.parse_graph_file('PathFinder-test.txt')

paths = ida_star(graph, origin, destinations, nodes)

for destination, path in paths.items():
    print(f"Path to destination {destination}: {path}")
