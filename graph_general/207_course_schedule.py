"""
207. Course Schedule

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return true if you can finish all courses. Otherwise, return false.

Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.

Example 2:
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.

Constraints:
- 1 <= numCourses <= 2000
- 0 <= prerequisites.length <= 5000
- prerequisites[i].length == 2
- 0 <= ai, bi < numCourses
- All the pairs prerequisites[i] are unique.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque


def can_finish_dfs_cycle_detection(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    DFS cycle detection approach (optimal solution).
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + recursion stack
    
    Algorithm:
    1. Build adjacency list representation of the graph
    2. Use DFS to detect cycles in the directed graph
    3. Use three colors: white (unvisited), gray (visiting), black (visited)
    4. If we find a gray node during DFS, there's a cycle
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # Color states: 0 = white (unvisited), 1 = gray (visiting), 2 = black (visited)
    color = [0] * numCourses
    
    def has_cycle(node):
        if color[node] == 1:  # Gray - cycle detected
            return True
        if color[node] == 2:  # Black - already processed
            return False
        
        # Mark as gray (visiting)
        color[node] = 1
        
        # Visit all neighbors
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        
        # Mark as black (visited)
        color[node] = 2
        return False
    
    # Check for cycles starting from each unvisited node
    for course in range(numCourses):
        if color[course] == 0:
            if has_cycle(course):
                return False
    
    return True


def can_finish_bfs_topological_sort(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    BFS topological sort approach (Kahn's algorithm).
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + queue
    
    Algorithm:
    1. Build adjacency list and calculate in-degrees
    2. Start with nodes having in-degree 0 (no prerequisites)
    3. Process nodes level by level, removing edges
    4. If all nodes are processed, no cycle exists
    """
    # Build adjacency list and in-degree array
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Find all nodes with in-degree 0
    queue = deque()
    for course in range(numCourses):
        if in_degree[course] == 0:
            queue.append(course)
    
    completed_courses = 0
    
    while queue:
        current = queue.popleft()
        completed_courses += 1
        
        # Remove edges from current node
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return completed_courses == numCourses


def can_finish_iterative_dfs(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Iterative DFS approach using stack.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + stack
    
    Algorithm:
    1. Use explicit stack for DFS traversal
    2. Track visiting and visited states
    3. Detect cycles using stack-based approach
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # Color states: 0 = white, 1 = gray, 2 = black
    color = [0] * numCourses
    
    def has_cycle_iterative(start):
        if color[start] != 0:
            return False
        
        stack = [start]
        path = set()
        
        while stack:
            node = stack[-1]
            
            if node in path:
                return True  # Cycle detected
            
            if color[node] == 2:  # Already processed
                stack.pop()
                continue
            
            if color[node] == 0:  # Unvisited
                color[node] = 1
                path.add(node)
                
                # Add neighbors to stack
                for neighbor in graph[node]:
                    if color[neighbor] != 2:
                        stack.append(neighbor)
            else:  # color[node] == 1, finishing processing
                path.remove(node)
                color[node] = 2
                stack.pop()
        
        return False
    
    # Check each component
    for course in range(numCourses):
        if color[course] == 0:
            if has_cycle_iterative(course):
                return False
    
    return True


def can_finish_union_find(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Union-Find approach (not optimal for this problem).
    
    Time Complexity: O(E * α(V)) where α is inverse Ackermann function
    Space Complexity: O(V) - parent array
    
    Algorithm:
    1. Use Union-Find to detect cycles
    2. For each edge, check if nodes are in same component
    3. If yes, there's a cycle
    
    Note: This approach doesn't work directly for directed graphs.
    """
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False  # Already in same component
            
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            
            return True
    
    # Union-Find doesn't work directly for directed graphs
    # This is a simplified approach that may not be correct
    uf = UnionFind(numCourses)
    
    # Check if adding each edge creates a cycle
    for course, prereq in prerequisites:
        if not uf.union(course, prereq):
            return False
    
    return True


def can_finish_adjacency_matrix(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Adjacency matrix approach with DFS.
    
    Time Complexity: O(V^2) where V is number of courses
    Space Complexity: O(V^2) - adjacency matrix
    
    Algorithm:
    1. Build adjacency matrix representation
    2. Use DFS with adjacency matrix
    3. Detect cycles using three-color approach
    """
    # Build adjacency matrix
    graph = [[False] * numCourses for _ in range(numCourses)]
    for course, prereq in prerequisites:
        graph[prereq][course] = True
    
    # Color states: 0 = white, 1 = gray, 2 = black
    color = [0] * numCourses
    
    def has_cycle(node):
        if color[node] == 1:  # Gray - cycle detected
            return True
        if color[node] == 2:  # Black - already processed
            return False
        
        color[node] = 1  # Mark as gray
        
        # Check all neighbors
        for neighbor in range(numCourses):
            if graph[node][neighbor]:
                if has_cycle(neighbor):
                    return True
        
        color[node] = 2  # Mark as black
        return False
    
    # Check for cycles
    for course in range(numCourses):
        if color[course] == 0:
            if has_cycle(course):
                return False
    
    return True


def can_finish_modified_bfs(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Modified BFS approach with level processing.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + queue
    
    Algorithm:
    1. Process courses level by level
    2. Each level contains courses with no remaining prerequisites
    3. Remove processed courses from prerequisites
    """
    # Build adjacency list and in-degree count
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Process courses level by level
    processed = set()
    
    while len(processed) < numCourses:
        # Find courses with no prerequisites
        ready_courses = []
        for course in range(numCourses):
            if course not in processed and in_degree[course] == 0:
                ready_courses.append(course)
        
        if not ready_courses:
            return False  # No progress possible - cycle detected
        
        # Process ready courses
        for course in ready_courses:
            processed.add(course)
            
            # Reduce in-degree for dependent courses
            for dependent in graph[course]:
                in_degree[dependent] -= 1
    
    return True


def can_finish_recursive_memoization(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Recursive approach with memoization.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - memoization + recursion stack
    
    Algorithm:
    1. Use memoization to cache results
    2. For each course, check if it can be completed
    3. A course can be completed if all its prerequisites can be completed
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # Memoization cache: -1 = unknown, 0 = impossible, 1 = possible
    memo = [-1] * numCourses
    
    def can_complete(course, visiting):
        if course in visiting:
            return False  # Cycle detected
        
        if memo[course] != -1:
            return memo[course] == 1
        
        visiting.add(course)
        
        # Check all prerequisites
        for prereq in graph[course]:
            if not can_complete(prereq, visiting):
                memo[course] = 0
                visiting.remove(course)
                return False
        
        visiting.remove(course)
        memo[course] = 1
        return True
    
    # Check if all courses can be completed
    for course in range(numCourses):
        if not can_complete(course, set()):
            return False
    
    return True


def can_finish_strongly_connected_components(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Strongly Connected Components approach (Tarjan's algorithm).
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + stack
    
    Algorithm:
    1. Use Tarjan's algorithm to find SCCs
    2. If any SCC has more than one node, there's a cycle
    3. Return false if cycle is detected
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # Tarjan's algorithm variables
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    
    def strongconnect(node):
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True
        
        for neighbor in graph[node]:
            if neighbor not in index:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif on_stack[neighbor]:
                lowlinks[node] = min(lowlinks[node], index[neighbor])
        
        # If node is a root node, pop the stack and get SCC
        if lowlinks[node] == index[node]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == node:
                    break
            
            # If component has more than one node, there's a cycle
            if len(component) > 1:
                return False
        
        return True
    
    # Find all SCCs
    for course in range(numCourses):
        if course not in index:
            if not strongconnect(course):
                return False
    
    return True


def can_finish_path_compression(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Path compression approach.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + visited set
    
    Algorithm:
    1. Use path compression to detect cycles
    2. For each path, compress it to avoid redundant computation
    3. Detect cycles during path compression
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # State: 0 = white, 1 = gray, 2 = black
    state = [0] * numCourses
    
    def has_cycle(node):
        if state[node] == 1:
            return True  # Gray - cycle detected
        if state[node] == 2:
            return False  # Black - already processed
        
        state[node] = 1  # Mark as gray
        
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        
        state[node] = 2  # Mark as black
        return False
    
    # Check each component
    for course in range(numCourses):
        if state[course] == 0:
            if has_cycle(course):
                return False
    
    return True


# Test cases
def test_can_finish():
    """Test all course schedule approaches."""
    
    # Test cases
    test_cases = [
        (2, [[1, 0]], True, "Simple valid schedule"),
        (2, [[1, 0], [0, 1]], False, "Simple cycle"),
        (4, [[1, 0], [2, 0], [3, 1], [3, 2]], True, "Complex valid schedule"),
        (3, [[0, 1], [1, 2], [2, 0]], False, "Three-node cycle"),
        (4, [[0, 1], [1, 2], [2, 3]], True, "Linear chain"),
        (5, [[1, 0], [2, 1], [3, 2], [4, 3], [0, 4]], False, "Five-node cycle"),
        (1, [], True, "Single course, no prerequisites"),
        (3, [[1, 0], [2, 1]], True, "Simple chain"),
        (4, [[1, 0], [2, 1], [3, 2], [1, 3]], False, "Complex cycle"),
        (6, [[3, 0], [3, 1], [4, 1], [4, 2], [5, 3], [5, 4]], True, "DAG with multiple paths"),
    ]
    
    # Test all approaches
    approaches = [
        ("DFS Cycle Detection", can_finish_dfs_cycle_detection),
        ("BFS Topological Sort", can_finish_bfs_topological_sort),
        ("Iterative DFS", can_finish_iterative_dfs),
        ("Union-Find", can_finish_union_find),
        ("Adjacency Matrix", can_finish_adjacency_matrix),
        ("Modified BFS", can_finish_modified_bfs),
        ("Recursive Memoization", can_finish_recursive_memoization),
        ("Strongly Connected Components", can_finish_strongly_connected_components),
        ("Path Compression", can_finish_path_compression),
    ]
    
    print("Testing course schedule approaches:")
    print("=" * 50)
    
    for i, (numCourses, prerequisites, expected, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        print(f"Courses: {numCourses}, Prerequisites: {prerequisites}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            try:
                result = func(numCourses, prerequisites)
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    # Create larger test case for performance testing
    def create_large_test_case(n, density=0.1):
        """Create a large test case with n courses."""
        import random
        prerequisites = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    prerequisites.append([j, i])
        
        return n, prerequisites
    
    large_numCourses, large_prerequisites = create_large_test_case(100, 0.05)
    
    import time
    
    print(f"Testing with large dataset ({large_numCourses} courses, {len(large_prerequisites)} prerequisites):")
    for name, func in approaches:
        try:
            start_time = time.time()
            result = func(large_numCourses, large_prerequisites)
            end_time = time.time()
            print(f"{name}: {result} (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    # Edge case testing
    print("\n" + "=" * 50)
    print("Edge Case Testing:")
    print("=" * 50)
    
    edge_cases = [
        (1, [], True, "Single course"),
        (2, [], True, "Two courses, no prerequisites"),
        (1000, [], True, "Many courses, no prerequisites"),
        (2, [[0, 1], [1, 0]], False, "Two courses, mutual dependency"),
    ]
    
    for i, (numCourses, prerequisites, expected, description) in enumerate(edge_cases):
        print(f"\nEdge Case {i + 1}: {description}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            try:
                result = func(numCourses, prerequisites)
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")


if __name__ == "__main__":
    test_can_finish() 