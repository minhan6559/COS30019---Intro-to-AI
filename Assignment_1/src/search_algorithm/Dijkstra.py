import heapq
import reader

def dijkstra(graph, start, goals):
    # Priority queue for the open list (stores nodes with their cumulative cost)
    pq = [(0, start)]  #(cost, node)
    # Store the shortest distance to each node
    distances = {start: 0}
    # Store the parent node for path reconstruction
    previous_nodes = {start: None}
    
    all_paths = {}

    while pq:
        current_cost, current_node = heapq.heappop(pq)
        if current_node is None:
            continue
        
        # If we reached one of the goals, reconstruct the path
        if current_node in goals:
            # Reconstruct path
            path = []
            goal = current_node  # Ensure the goal is correctly set
            while current_node is not None:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            path.reverse()  # To get the correct order of nodes
            all_paths[goal] = (path, current_cost)  # Store the path and cost for the goal

            # If all goals are found, return the paths
            if len(all_paths) == len(goals):
                return all_paths
        
        # Explore the neighbors
        for neighbor, cost in graph.get(current_node, []):  # Use get to avoid KeyError
            new_cost = current_cost + cost
            # If this path is better, update the distance and the path
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (new_cost, neighbor))
    
    return all_paths


nodes, graph, origin, destinations = reader.parse_graph_file('PathFinder-test.txt')

paths = dijkstra(graph, origin, destinations)

for destination, (path, cost) in paths.items():
    print(f"Path to destination {destination}: {path}, Total cost: {cost}")
