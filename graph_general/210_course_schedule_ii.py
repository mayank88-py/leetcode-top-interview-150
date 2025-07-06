"""
210. Course Schedule II

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].

Example 2:
Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0.
So one correct course order is [0,2,1,3]. Another correct order is [0,1,2,3].

Example 3:
Input: numCourses = 1, prerequisites = []
Output: [0]

Constraints:
- 1 <= numCourses <= 2000
- 0 <= prerequisites.length <= numCourses * (numCourses - 1)
- prerequisites[i].length == 2
- 0 <= ai, bi < numCourses
- ai != bi
- All the pairs [ai, bi] are distinct.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque


def find_order_bfs_topological_sort(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    BFS topological sort approach (Kahn's algorithm) - optimal solution.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + queue
    
    Algorithm:
    1. Build adjacency list and calculate in-degrees
    2. Start with nodes having in-degree 0 (no prerequisites)
    3. Process nodes level by level, removing edges
    4. Return topological order or empty array if cycle exists
    """
    # Build adjacency list and in-degree array
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Find all courses with no prerequisites
    queue = deque()
    for course in range(numCourses):
        if in_degree[course] == 0:
            queue.append(course)
    
    result = []
    
    while queue:
        current = queue.popleft()
        result.append(current)
        
        # Remove edges from current course
        for dependent in graph[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    return result if len(result) == numCourses else []


def find_order_dfs_postorder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    DFS post-order traversal approach.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + recursion stack
    
    Algorithm:
    1. Build adjacency list representation
    2. Use DFS with post-order traversal
    3. Add courses to result in post-order (reverse topological order)
    4. Reverse the result to get correct topological order
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # Color states: 0 = white, 1 = gray, 2 = black
    color = [0] * numCourses
    result = []
    has_cycle = [False]
    
    def dfs(node):
        if has_cycle[0]:
            return
        
        if color[node] == 1:  # Gray - cycle detected
            has_cycle[0] = True
            return
        
        if color[node] == 2:  # Black - already processed
            return
        
        color[node] = 1  # Mark as gray
        
        # Visit all dependent courses
        for dependent in graph[node]:
            dfs(dependent)
        
        color[node] = 2  # Mark as black
        result.append(node)  # Add in post-order
    
    # Process all courses
    for course in range(numCourses):
        if color[course] == 0:
            dfs(course)
            if has_cycle[0]:
                return []
    
    return result[::-1]  # Reverse to get topological order


def find_order_iterative_dfs(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Iterative DFS approach using stack.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + stack
    
    Algorithm:
    1. Use explicit stack for DFS traversal
    2. Track visiting and visited states
    3. Collect courses in post-order during traversal
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # Color states: 0 = white, 1 = gray, 2 = black
    color = [0] * numCourses
    result = []
    
    def iterative_dfs(start):
        if color[start] != 0:
            return True
        
        stack = [(start, False)]
        
        while stack:
            node, processed = stack.pop()
            
            if processed:
                # Post-processing: add to result
                result.append(node)
                color[node] = 2
            else:
                if color[node] == 1:
                    return False  # Cycle detected
                if color[node] == 2:
                    continue  # Already processed
                
                color[node] = 1  # Mark as visiting
                stack.append((node, True))  # Mark for post-processing
                
                # Add neighbors to stack
                for neighbor in graph[node]:
                    if color[neighbor] != 2:
                        stack.append((neighbor, False))
        
        return True
    
    # Process all courses
    for course in range(numCourses):
        if color[course] == 0:
            if not iterative_dfs(course):
                return []
    
    return result[::-1]  # Reverse to get topological order


def find_order_modified_kahn(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Modified Kahn's algorithm with priority.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + queue
    
    Algorithm:
    1. Use standard Kahn's algorithm
    2. Process courses in priority order (smallest first)
    3. Return topological order
    """
    import heapq
    
    # Build adjacency list and in-degree array
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Use min-heap for priority ordering
    heap = []
    for course in range(numCourses):
        if in_degree[course] == 0:
            heapq.heappush(heap, course)
    
    result = []
    
    while heap:
        current = heapq.heappop(heap)
        result.append(current)
        
        # Remove edges from current course
        for dependent in graph[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                heapq.heappush(heap, dependent)
    
    return result if len(result) == numCourses else []


def find_order_level_order(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Level-order processing approach.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list + queue
    
    Algorithm:
    1. Process courses level by level
    2. Each level contains courses with no remaining prerequisites
    3. Maintain level structure in the result
    """
    # Build adjacency list and in-degree array
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    result = []
    processed = set()
    
    while len(processed) < numCourses:
        # Find courses ready to be taken
        current_level = []
        for course in range(numCourses):
            if course not in processed and in_degree[course] == 0:
                current_level.append(course)
        
        if not current_level:
            return []  # Cycle detected
        
        # Process current level
        for course in current_level:
            processed.add(course)
            result.append(course)
            
            # Update in-degrees
            for dependent in graph[course]:
                in_degree[dependent] -= 1
    
    return result


def find_order_recursive_with_memoization(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Recursive approach with memoization.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - memoization + recursion stack
    
    Algorithm:
    1. Use memoization to cache course completion order
    2. For each course, recursively determine prerequisites
    3. Build topological order through recursive calls
    """
    # Build adjacency list (reverse direction for easier processing)
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # Memoization: -1 = processing, 0 = not processed, 1 = completed
    memo = [0] * numCourses
    result = []
    
    def process_course(course):
        if memo[course] == -1:
            return False  # Cycle detected
        if memo[course] == 1:
            return True  # Already processed
        
        memo[course] = -1  # Mark as processing
        
        # Process all prerequisites first
        for prereq in graph[course]:
            if not process_course(prereq):
                return False
        
        memo[course] = 1  # Mark as completed
        result.append(course)
        return True
    
    # Process all courses
    for course in range(numCourses):
        if memo[course] == 0:
            if not process_course(course):
                return []
    
    return result


def find_order_adjacency_matrix(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Adjacency matrix approach with topological sort.
    
    Time Complexity: O(V^2) where V is number of courses
    Space Complexity: O(V^2) - adjacency matrix
    
    Algorithm:
    1. Build adjacency matrix representation
    2. Use topological sort with adjacency matrix
    3. Process courses by removing edges
    """
    # Build adjacency matrix and in-degree array
    graph = [[False] * numCourses for _ in range(numCourses)]
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq][course] = True
        in_degree[course] += 1
    
    result = []
    processed = [False] * numCourses
    
    for _ in range(numCourses):
        # Find a course with in-degree 0
        found = False
        for course in range(numCourses):
            if not processed[course] and in_degree[course] == 0:
                result.append(course)
                processed[course] = True
                found = True
                
                # Update in-degrees
                for dependent in range(numCourses):
                    if graph[course][dependent]:
                        in_degree[dependent] -= 1
                break
        
        if not found:
            return []  # Cycle detected
    
    return result


def find_order_backwards_construction(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Backwards construction approach.
    
    Time Complexity: O(V + E) where V is courses and E is prerequisites
    Space Complexity: O(V + E) - adjacency list
    
    Algorithm:
    1. Start from courses with no dependents
    2. Work backwards to build the schedule
    3. Reverse the final result
    """
    # Build reverse adjacency list (dependents -> prerequisites)
    graph = defaultdict(list)
    out_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[course].append(prereq)
        out_degree[prereq] += 1
    
    # Find courses with no dependents
    queue = deque()
    for course in range(numCourses):
        if out_degree[course] == 0:
            queue.append(course)
    
    result = []
    
    while queue:
        current = queue.popleft()
        result.append(current)
        
        # Remove edges to current course
        for prereq in graph[current]:
            out_degree[prereq] -= 1
            if out_degree[prereq] == 0:
                queue.append(prereq)
    
    return result[::-1] if len(result) == numCourses else []


def find_order_union_find_approach(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Union-Find based approach (not optimal for directed graphs).
    
    Time Complexity: O(V + E + V log V) for sorting
    Space Complexity: O(V) - parent array
    
    Algorithm:
    1. Use Union-Find to detect cycles
    2. If no cycles, use topological sort
    3. This is more of a demonstration approach
    """
    # First check for cycles using simple DFS
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    color = [0] * numCourses
    
    def has_cycle(node):
        if color[node] == 1:
            return True
        if color[node] == 2:
            return False
        
        color[node] = 1
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        color[node] = 2
        return False
    
    # Check for cycles
    for course in range(numCourses):
        if color[course] == 0:
            if has_cycle(course):
                return []
    
    # If no cycles, use BFS topological sort
    return find_order_bfs_topological_sort(numCourses, prerequisites)


# Test cases
def test_find_order():
    """Test all course schedule II approaches."""
    
    # Test cases
    test_cases = [
        (2, [[1, 0]], [0, 1], "Simple linear schedule"),
        (4, [[1, 0], [2, 0], [3, 1], [3, 2]], [0, 1, 2, 3], "Complex valid schedule"),
        (1, [], [0], "Single course"),
        (3, [[1, 0], [2, 1]], [0, 1, 2], "Linear chain"),
        (2, [[1, 0], [0, 1]], [], "Simple cycle"),
        (3, [[0, 1], [1, 2], [2, 0]], [], "Three-node cycle"),
        (4, [[0, 1], [1, 2], [2, 3]], [1, 2, 3, 0], "Linear dependency"),
        (6, [[3, 0], [3, 1], [4, 1], [4, 2], [5, 3], [5, 4]], [0, 1, 2, 3, 4, 5], "DAG with multiple paths"),
    ]
    
    def is_valid_order(numCourses, prerequisites, order):
        """Check if the order is a valid topological sort."""
        if not order or len(order) != numCourses:
            return len(order) == 0  # Should be empty if impossible
        
        # Create position map
        position = {course: i for i, course in enumerate(order)}
        
        # Check all prerequisites
        for course, prereq in prerequisites:
            if position[prereq] >= position[course]:
                return False
        
        return True
    
    # Test all approaches
    approaches = [
        ("BFS Topological Sort", find_order_bfs_topological_sort),
        ("DFS Post-order", find_order_dfs_postorder),
        ("Iterative DFS", find_order_iterative_dfs),
        ("Modified Kahn", find_order_modified_kahn),
        ("Level Order", find_order_level_order),
        ("Recursive Memoization", find_order_recursive_with_memoization),
        ("Adjacency Matrix", find_order_adjacency_matrix),
        ("Backwards Construction", find_order_backwards_construction),
        ("Union-Find Approach", find_order_union_find_approach),
    ]
    
    print("Testing course schedule II approaches:")
    print("=" * 50)
    
    for i, (numCourses, prerequisites, expected_type, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        print(f"Courses: {numCourses}, Prerequisites: {prerequisites}")
        print(f"Expected: Valid order of length {len(expected_type) if expected_type else 0}")
        
        for name, func in approaches:
            try:
                result = func(numCourses, prerequisites)
                is_valid = is_valid_order(numCourses, prerequisites, result)
                status = "✓" if is_valid else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    # Create larger test case for performance testing
    def create_large_test_case(n, density=0.05):
        """Create a large test case with n courses."""
        import random
        prerequisites = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    prerequisites.append([j, i])
        
        return n, prerequisites
    
    large_numCourses, large_prerequisites = create_large_test_case(100, 0.03)
    
    import time
    
    print(f"Testing with large dataset ({large_numCourses} courses, {len(large_prerequisites)} prerequisites):")
    for name, func in approaches:
        try:
            start_time = time.time()
            result = func(large_numCourses, large_prerequisites)
            end_time = time.time()
            is_valid = is_valid_order(large_numCourses, large_prerequisites, result)
            status = "✓" if is_valid else "✗"
            print(f"{status} {name}: Length {len(result)} (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")


if __name__ == "__main__":
    test_find_order() 