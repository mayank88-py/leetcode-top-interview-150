"""
399. Evaluate Division

You are given an array of variable pairs equations and an array of real numbers values, where equations[i] = [Ai, Bi] and values[i] represent the equation Ai / Bi = values[i]. Each Ai or Bi is a string that represents a single variable.

You are also given some queries, where queries[j] = [Cj, Dj] represents the jth query where you must find the answer for Cj / Dj = ?.

Return the answers to all queries. If a single answer cannot be determined, return -1.0.

Note: The input is always valid. You may assume that evaluating the queries will not result in division by zero and that there is no contradiction.

Example 1:
Input: equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
Output: [6.00000,0.50000,-1.00000,1.00000,-1.00000]
Explanation: 
Given: a / b = 2.0, b / c = 3.0
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
return: [6.0, 0.5, -1.0, 1.0, -1.0]

Example 2:
Input: equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
Output: [3.75000,0.40000,5.00000,0.20000]

Example 3:
Input: equations = [["a","b"]], values = [0.5], queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
Output: [0.50000,2.00000,-1.00000,-1.00000]

Constraints:
- 1 <= equations.length <= 20
- equations[i].length == 2
- 1 <= Ai.length, Bi.length <= 5
- values.length == equations.length
- 0.0 < values[i] <= 20.0
- 1 <= queries.length <= 20
- queries[i].length == 2
- 1 <= Cj.length, Dj.length <= 5
- Ai, Bi, Cj, Dj consist of lower case English letters and digits.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque


def calc_equation_dfs(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    """
    DFS approach with weighted graph (optimal solution).
    
    Time Complexity: O(M * N) where M is queries and N is variables
    Space Complexity: O(N) - graph storage + recursion stack
    
    Algorithm:
    1. Build weighted directed graph from equations
    2. For each query, use DFS to find path and calculate product
    3. Return -1.0 if path doesn't exist
    """
    # Build weighted graph
    graph = defaultdict(list)
    
    for i, (a, b) in enumerate(equations):
        value = values[i]
        graph[a].append((b, value))
        graph[b].append((a, 1.0 / value))
    
    def dfs(start, end, visited):
        if start not in graph or end not in graph:
            return -1.0
        
        if start == end:
            return 1.0
        
        visited.add(start)
        
        for neighbor, weight in graph[start]:
            if neighbor not in visited:
                result = dfs(neighbor, end, visited)
                if result != -1.0:
                    return weight * result
        
        return -1.0
    
    results = []
    for dividend, divisor in queries:
        visited = set()
        result = dfs(dividend, divisor, visited)
        results.append(result)
    
    return results


def calc_equation_bfs(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    """
    BFS approach with weighted graph.
    
    Time Complexity: O(M * N) where M is queries and N is variables
    Space Complexity: O(N) - graph storage + queue
    
    Algorithm:
    1. Build weighted directed graph from equations
    2. For each query, use BFS to find path and calculate product
    3. Track cumulative product during BFS traversal
    """
    # Build weighted graph
    graph = defaultdict(list)
    
    for i, (a, b) in enumerate(equations):
        value = values[i]
        graph[a].append((b, value))
        graph[b].append((a, 1.0 / value))
    
    def bfs(start, end):
        if start not in graph or end not in graph:
            return -1.0
        
        if start == end:
            return 1.0
        
        queue = deque([(start, 1.0)])
        visited = {start}
        
        while queue:
            current, product = queue.popleft()
            
            for neighbor, weight in graph[current]:
                if neighbor == end:
                    return product * weight
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, product * weight))
        
        return -1.0
    
    results = []
    for dividend, divisor in queries:
        result = bfs(dividend, divisor)
        results.append(result)
    
    return results


def calc_equation_union_find(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    """
    Union-Find approach with weighted path compression.
    
    Time Complexity: O(N * α(N) + M) where N is variables, M is queries
    Space Complexity: O(N) - parent and weight arrays
    
    Algorithm:
    1. Use weighted Union-Find data structure
    2. Each variable has a parent and weight relative to parent
    3. Path compression maintains weight relationships
    """
    class WeightedUnionFind:
        def __init__(self):
            self.parent = {}
            self.weight = {}
        
        def add(self, x):
            if x not in self.parent:
                self.parent[x] = x
                self.weight[x] = 1.0
        
        def find(self, x):
            if x not in self.parent:
                return None
            
            if self.parent[x] != x:
                original_parent = self.parent[x]
                self.parent[x] = self.find(original_parent)
                self.weight[x] *= self.weight[original_parent]
            
            return self.parent[x]
        
        def union(self, x, y, value):
            self.add(x)
            self.add(y)
            
            root_x = self.find(x)
            root_y = self.find(y)
            
            if root_x != root_y:
                self.parent[root_x] = root_y
                self.weight[root_x] = self.weight[y] * value / self.weight[x]
        
        def is_connected(self, x, y):
            return (x in self.parent and y in self.parent and 
                    self.find(x) == self.find(y))
        
        def get_ratio(self, x, y):
            if not self.is_connected(x, y):
                return -1.0
            
            return self.weight[x] / self.weight[y]
    
    # Build Union-Find structure
    uf = WeightedUnionFind()
    
    for i, (a, b) in enumerate(equations):
        uf.union(a, b, values[i])
    
    # Process queries
    results = []
    for dividend, divisor in queries:
        result = uf.get_ratio(dividend, divisor)
        results.append(result)
    
    return results


def calc_equation_floyd_warshall(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    """
    Floyd-Warshall algorithm approach.
    
    Time Complexity: O(N^3) where N is number of variables
    Space Complexity: O(N^2) - distance matrix
    
    Algorithm:
    1. Create distance matrix for all variable pairs
    2. Use Floyd-Warshall to find all shortest paths
    3. Query results from precomputed matrix
    """
    # Collect all variables
    variables = set()
    for a, b in equations:
        variables.add(a)
        variables.add(b)
    
    # Map variables to indices
    var_to_idx = {var: i for i, var in enumerate(variables)}
    n = len(variables)
    
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Set diagonal to 1.0 (a/a = 1)
    for i in range(n):
        dist[i][i] = 1.0
    
    # Fill known distances
    for i, (a, b) in enumerate(equations):
        idx_a, idx_b = var_to_idx[a], var_to_idx[b]
        dist[idx_a][idx_b] = values[i]
        dist[idx_b][idx_a] = 1.0 / values[i]
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] * dist[k][j])
    
    # Process queries
    results = []
    for dividend, divisor in queries:
        if dividend not in var_to_idx or divisor not in var_to_idx:
            results.append(-1.0)
        else:
            idx_dividend = var_to_idx[dividend]
            idx_divisor = var_to_idx[divisor]
            result = dist[idx_dividend][idx_divisor]
            results.append(result if result != float('inf') else -1.0)
    
    return results


def calc_equation_iterative_dfs(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    """
    Iterative DFS approach using stack.
    
    Time Complexity: O(M * N) where M is queries and N is variables
    Space Complexity: O(N) - graph storage + stack
    
    Algorithm:
    1. Build weighted directed graph
    2. Use iterative DFS with explicit stack
    3. Track cumulative product during traversal
    """
    # Build weighted graph
    graph = defaultdict(list)
    
    for i, (a, b) in enumerate(equations):
        value = values[i]
        graph[a].append((b, value))
        graph[b].append((a, 1.0 / value))
    
    def iterative_dfs(start, end):
        if start not in graph or end not in graph:
            return -1.0
        
        if start == end:
            return 1.0
        
        stack = [(start, 1.0)]
        visited = {start}
        
        while stack:
            current, product = stack.pop()
            
            for neighbor, weight in graph[current]:
                if neighbor == end:
                    return product * weight
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, product * weight))
        
        return -1.0
    
    results = []
    for dividend, divisor in queries:
        result = iterative_dfs(dividend, divisor)
        results.append(result)
    
    return results


def calc_equation_bidirectional_bfs(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    """
    Bidirectional BFS approach.
    
    Time Complexity: O(M * N) where M is queries and N is variables
    Space Complexity: O(N) - graph storage + queues
    
    Algorithm:
    1. Search from both start and end simultaneously
    2. Meet in the middle to find path
    3. Combine products from both directions
    """
    # Build weighted graph
    graph = defaultdict(list)
    
    for i, (a, b) in enumerate(equations):
        value = values[i]
        graph[a].append((b, value))
        graph[b].append((a, 1.0 / value))
    
    def bidirectional_bfs(start, end):
        if start not in graph or end not in graph:
            return -1.0
        
        if start == end:
            return 1.0
        
        # Forward search from start
        forward_queue = deque([(start, 1.0)])
        forward_visited = {start: 1.0}
        
        # Backward search from end
        backward_queue = deque([(end, 1.0)])
        backward_visited = {end: 1.0}
        
        while forward_queue or backward_queue:
            # Forward step
            if forward_queue:
                current, product = forward_queue.popleft()
                
                if current in backward_visited:
                    return product * backward_visited[current]
                
                for neighbor, weight in graph[current]:
                    new_product = product * weight
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = new_product
                        forward_queue.append((neighbor, new_product))
            
            # Backward step
            if backward_queue:
                current, product = backward_queue.popleft()
                
                if current in forward_visited:
                    return forward_visited[current] * product
                
                for neighbor, weight in graph[current]:
                    new_product = product * weight
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = new_product
                        backward_queue.append((neighbor, new_product))
        
        return -1.0
    
    results = []
    for dividend, divisor in queries:
        result = bidirectional_bfs(dividend, divisor)
        results.append(result)
    
    return results


def calc_equation_memoization(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    """
    DFS with memoization approach.
    
    Time Complexity: O(M + N^2) where M is queries and N is variables
    Space Complexity: O(N^2) - memoization cache
    
    Algorithm:
    1. Use DFS with memoization to cache results
    2. Store computed ratios to avoid recomputation
    3. Query from cache when available
    """
    # Build weighted graph
    graph = defaultdict(list)
    
    for i, (a, b) in enumerate(equations):
        value = values[i]
        graph[a].append((b, value))
        graph[b].append((a, 1.0 / value))
    
    # Memoization cache
    memo = {}
    
    def dfs_memo(start, end, visited):
        if (start, end) in memo:
            return memo[(start, end)]
        
        if start not in graph or end not in graph:
            memo[(start, end)] = -1.0
            return -1.0
        
        if start == end:
            memo[(start, end)] = 1.0
            return 1.0
        
        visited.add(start)
        
        for neighbor, weight in graph[start]:
            if neighbor not in visited:
                result = dfs_memo(neighbor, end, visited)
                if result != -1.0:
                    final_result = weight * result
                    memo[(start, end)] = final_result
                    visited.remove(start)
                    return final_result
        
        visited.remove(start)
        memo[(start, end)] = -1.0
        return -1.0
    
    results = []
    for dividend, divisor in queries:
        visited = set()
        result = dfs_memo(dividend, divisor, visited)
        results.append(result)
    
    return results


def calc_equation_matrix_multiplication(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    """
    Matrix multiplication approach.
    
    Time Complexity: O(N^3 + M) where N is variables and M is queries
    Space Complexity: O(N^2) - adjacency matrix
    
    Algorithm:
    1. Represent relationships as adjacency matrix
    2. Use matrix multiplication to find transitive relationships
    3. Query results from final matrix
    """
    # Collect all variables
    variables = set()
    for a, b in equations:
        variables.add(a)
        variables.add(b)
    
    var_to_idx = {var: i for i, var in enumerate(variables)}
    n = len(variables)
    
    # Initialize adjacency matrix
    matrix = [[0.0] * n for _ in range(n)]
    
    # Set diagonal to 1.0
    for i in range(n):
        matrix[i][i] = 1.0
    
    # Fill known relationships
    for i, (a, b) in enumerate(equations):
        idx_a, idx_b = var_to_idx[a], var_to_idx[b]
        matrix[idx_a][idx_b] = values[i]
        matrix[idx_b][idx_a] = 1.0 / values[i]
    
    # Matrix multiplication to find transitive relationships
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if matrix[i][k] != 0.0 and matrix[k][j] != 0.0:
                    if matrix[i][j] == 0.0 and i != j:
                        matrix[i][j] = matrix[i][k] * matrix[k][j]
    
    # Process queries
    results = []
    for dividend, divisor in queries:
        if dividend not in var_to_idx or divisor not in var_to_idx:
            results.append(-1.0)
        else:
            idx_dividend = var_to_idx[dividend]
            idx_divisor = var_to_idx[divisor]
            result = matrix[idx_dividend][idx_divisor]
            results.append(result if result != 0.0 else -1.0)
    
    return results


# Test cases
def test_calc_equation():
    """Test all evaluate division approaches."""
    
    def are_close(a, b, tolerance=1e-5):
        """Check if two floating point numbers are close."""
        return abs(a - b) < tolerance
    
    def compare_results(result, expected):
        """Compare two result lists with floating point tolerance."""
        if len(result) != len(expected):
            return False
        
        for i in range(len(result)):
            if not are_close(result[i], expected[i]):
                return False
        
        return True
    
    # Test cases
    test_cases = [
        (
            [["a","b"],["b","c"]], 
            [2.0, 3.0], 
            [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]], 
            [6.0, 0.5, -1.0, 1.0, -1.0],
            "Basic division chain"
        ),
        (
            [["a","b"],["b","c"],["bc","cd"]], 
            [1.5, 2.5, 5.0], 
            [["a","c"],["c","b"],["bc","cd"],["cd","bc"]], 
            [3.75, 0.4, 5.0, 0.2],
            "Multiple chains"
        ),
        (
            [["a","b"]], 
            [0.5], 
            [["a","b"],["b","a"],["a","c"],["x","y"]], 
            [0.5, 2.0, -1.0, -1.0],
            "Simple ratio"
        ),
        (
            [["x1","x2"],["x2","x3"],["x3","x4"],["x4","x5"]], 
            [3.0, 4.0, 5.0, 6.0], 
            [["x1","x5"],["x5","x2"],["x2","x4"],["x2","x2"],["x2","x9"],["x9","x9"]], 
            [360.0, 1.0/24.0, 20.0, 1.0, -1.0, -1.0],
            "Long chain"
        ),
        (
            [], 
            [], 
            [["a","b"]], 
            [-1.0],
            "No equations"
        ),
    ]
    
    # Test all approaches
    approaches = [
        ("DFS", calc_equation_dfs),
        ("BFS", calc_equation_bfs),
        ("Union-Find", calc_equation_union_find),
        ("Floyd-Warshall", calc_equation_floyd_warshall),
        ("Iterative DFS", calc_equation_iterative_dfs),
        ("Bidirectional BFS", calc_equation_bidirectional_bfs),
        ("Memoization", calc_equation_memoization),
        ("Matrix Multiplication", calc_equation_matrix_multiplication),
    ]
    
    print("Testing evaluate division approaches:")
    print("=" * 50)
    
    for i, (equations, values, queries, expected, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        print(f"Equations: {equations}")
        print(f"Values: {values}")
        print(f"Queries: {queries}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            try:
                result = func(equations, values, queries)
                is_correct = compare_results(result, expected)
                status = "✓" if is_correct else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    # Create larger test case for performance testing
    def create_large_test_case():
        """Create a larger test case."""
        equations = []
        values = []
        
        # Create a chain of 20 variables
        for i in range(19):
            equations.append([f"var{i}", f"var{i+1}"])
            values.append(2.0)  # Each ratio is 2.0
        
        # Create some additional cross connections
        equations.append(["var0", "var5"])
        values.append(32.0)  # 2^5 = 32
        
        queries = [
            ["var0", "var19"],  # Should be 2^19
            ["var19", "var0"],  # Should be 1/2^19
            ["var5", "var15"],  # Should be 2^10
            ["var0", "varX"],   # Should be -1.0
        ]
        
        return equations, values, queries
    
    large_equations, large_values, large_queries = create_large_test_case()
    
    import time
    
    print(f"Testing with larger dataset ({len(large_equations)} equations, {len(large_queries)} queries):")
    for name, func in approaches:
        try:
            start_time = time.time()
            result = func(large_equations, large_values, large_queries)
            end_time = time.time()
            print(f"{name}: {result} (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_calc_equation() 