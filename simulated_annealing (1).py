"""
Simulated Annealing for Network Path Optimization
"""

import random
import numpy as np
from typing import List, Tuple, Dict
import networkx as nx
import math


class SimulatedAnnealing:
    """Simulated Annealing for finding optimal paths in networks"""
    
    def __init__(self, graph: nx.Graph, source: int, destination: int,
                 initial_temp: float = 1000.0, cooling_rate: float = 0.95,
                 num_iterations: int = 50, iterations_per_temp: int = 5):
        """
        Initialize Simulated Annealing
        
        Args:
            graph: Network graph
            source: Source node
            destination: Destination node
            initial_temp: Initial temperature
            cooling_rate: Temperature cooling rate
            num_iterations: Number of temperature steps
            iterations_per_temp: Number of iterations per temperature
        """
        self.graph = graph
        self.source = source
        self.destination = destination
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.iterations_per_temp = iterations_per_temp
        
        self.weights = {'delay': 0.33, 'reliability': 0.33, 'resource': 0.34}
        self.network_generator = None
    
    def set_weights(self, weights: Dict[str, float]):
        """Set weights for multi-objective optimization"""
        self.weights = weights
    
    def set_network_generator(self, network_generator):
        """Set network generator for metric calculations"""
        self.network_generator = network_generator
    
    def _calculate_cost(self, path: List[int]) -> float:
        """Calculate cost of a path"""
        if not self.network_generator:
            return float('inf')
        
        return self.network_generator.calculate_total_cost(path, self.weights)
    
    def _get_initial_solution(self) -> List[int]:
        """Get initial solution (shortest path)"""
        try:
            return nx.shortest_path(self.graph, self.source, self.destination)
        except:
            return [self.source, self.destination]
    
    def _get_neighbor(self, current_path: List[int]) -> List[int]:
        """Generate a neighbor solution by modifying current path"""
        if len(current_path) < 2:
            return current_path[:]
        
        new_path = current_path[:]
        
        # Choose a random modification
        modification = random.randint(0, 3)
        
        if modification == 0 and len(new_path) > 2:
            # Remove a random intermediate node
            remove_idx = random.randint(1, len(new_path) - 2)
            new_path.pop(remove_idx)
        
        elif modification == 1:
            # Insert a random node
            if len(new_path) >= 2:
                insert_idx = random.randint(1, len(new_path))
                prev_node = new_path[insert_idx - 1]
                
                # Get neighbors of previous node
                neighbors = list(self.graph.neighbors(prev_node))
                neighbors = [n for n in neighbors if n not in new_path[:insert_idx]]
                
                if neighbors:
                    new_node = random.choice(neighbors)
                    new_path.insert(insert_idx, new_node)
        
        elif modification == 2 and len(new_path) > 3:
            # Replace a segment with shortest path
            idx1 = random.randint(1, len(new_path) - 3)
            idx2 = random.randint(idx1 + 2, len(new_path) - 1)
            
            try:
                segment = nx.shortest_path(self.graph, new_path[idx1 - 1], 
                                          new_path[idx2])
                new_path = new_path[:idx1] + segment[1:]
            except:
                pass
        
        else:
            # Swap two intermediate nodes
            if len(new_path) > 3:
                idx1 = random.randint(1, len(new_path) - 2)
                idx2 = random.randint(1, len(new_path) - 2)
                if idx1 != idx2:
                    new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
        
        # Repair the path to ensure connectivity
        return self._repair_path(new_path)
    
    def _repair_path(self, path: List[int]) -> List[int]:
        """Repair path to ensure it's valid and connected"""
        if not path or len(path) < 2:
            try:
                return nx.shortest_path(self.graph, self.source, self.destination)
            except:
                return [self.source, self.destination]
        
        # Ensure correct start and end
        if path[0] != self.source:
            path.insert(0, self.source)
        if path[-1] != self.destination:
            path.append(self.destination)
        
        # Remove duplicates
        seen = set()
        unique = []
        for node in path:
            if node not in seen:
                seen.add(node)
                unique.append(node)
        
        # Ensure connectivity
        repaired = [unique[0]]
        
        for i in range(1, len(unique)):
            current = repaired[-1]
            target = unique[i]
            
            if current == target:
                continue
            
            # Check direct connection
            if self.graph.has_edge(current, target):
                repaired.append(target)
            else:
                # Find shortest path
                try:
                    bridge = nx.shortest_path(self.graph, current, target)
                    repaired.extend(bridge[1:])
                except:
                    continue
        
        # Final check for destination
        if repaired[-1] != self.destination:
            try:
                bridge = nx.shortest_path(self.graph, repaired[-1], self.destination)
                repaired.extend(bridge[1:])
            except:
                try:
                    return nx.shortest_path(self.graph, self.source, self.destination)
                except:
                    return [self.source, self.destination]
        
        return repaired
    
    def _acceptance_probability(self, current_cost: float, new_cost: float, 
                                temperature: float) -> float:
        """Calculate acceptance probability for worse solution"""
        if new_cost < current_cost:
            return 1.0
        
        if temperature <= 0:
            return 0.0
        
        return math.exp(-(new_cost - current_cost) / temperature)
    
    def optimize(self) -> Tuple[List[int], float, Dict]:
        """
        Run Simulated Annealing optimization
        
        Returns:
            Tuple of (best_path, best_cost, statistics)
        """
        if not self.network_generator:
            return [], float('inf'), {}
        
        # Get initial solution
        current_path = self._get_initial_solution()
        current_cost = self._calculate_cost(current_path)
        
        best_path = current_path[:]
        best_cost = current_cost
        
        temperature = self.initial_temp
        
        history = {
            'best_cost': [],
            'current_cost': [],
            'temperature': [],
            'acceptance_rate': []
        }
        
        total_iterations = 0
        accepted_moves = 0
        
        for iteration in range(self.num_iterations):
            for _ in range(self.iterations_per_temp):
                # Generate neighbor
                new_path = self._get_neighbor(current_path)
                new_cost = self._calculate_cost(new_path)
                
                # Decide whether to accept new solution
                acceptance_prob = self._acceptance_probability(current_cost, 
                                                               new_cost, 
                                                               temperature)
                
                if random.random() < acceptance_prob:
                    current_path = new_path
                    current_cost = new_cost
                    accepted_moves += 1
                    
                    # Update best solution
                    if current_cost < best_cost:
                        best_path = current_path[:]
                        best_cost = current_cost
                
                total_iterations += 1
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Record statistics
            history['best_cost'].append(best_cost)
            history['current_cost'].append(current_cost)
            history['temperature'].append(temperature)
            
            if total_iterations > 0:
                acceptance_rate = accepted_moves / total_iterations
            else:
                acceptance_rate = 0
            history['acceptance_rate'].append(acceptance_rate)
        
        return best_path, best_cost, history
