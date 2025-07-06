"""
133. Clone Graph

Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

Example 1:
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).

Example 2:
Input: adjList = [[]]
Output: [[]]
Explanation: Note that the input contains one empty list. The graph consists of only one node with val = 1 and it does not have any neighbors.

Example 3:
Input: adjList = []
Output: []
Explanation: This an empty graph, it does not have any nodes.

Constraints:
- The number of nodes in the graph is in the range [0, 100].
- 1 <= Node.val <= 100
- Node.val is unique for each node.
- There are no repeated edges and no self-loops in the graph.
- The Graph is connected and all nodes can be visited starting from the given node.
"""

from typing import Optional, List, Dict, Set
from collections import deque


class Node:
    """Definition for a Node in the graph."""
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
    
    def __repr__(self):
        """String representation for debugging."""
        return f"Node({self.val})"


def clone_graph_dfs(node: Optional[Node]) -> Optional[Node]:
    """
    DFS approach with recursion (optimal solution).
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) - visited map + recursion stack
    
    Algorithm:
    1. Use DFS to traverse the graph
    2. Maintain a visited map to avoid cycles
    3. Create new nodes and establish connections
    4. Return the cloned node
    """
    if not node:
        return None
    
    visited = {}
    
    def dfs(current):
        if current in visited:
            return visited[current]
        
        # Create clone of current node
        clone = Node(current.val)
        visited[current] = clone
        
        # Clone all neighbors
        for neighbor in current.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)


def clone_graph_bfs(node: Optional[Node]) -> Optional[Node]:
    """
    BFS approach using queue.
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) - queue + visited map
    
    Algorithm:
    1. Use BFS to traverse the graph level by level
    2. Maintain a visited map to track cloned nodes
    3. Create nodes and establish connections during traversal
    """
    if not node:
        return None
    
    visited = {}
    queue = deque([node])
    
    # Create clone of the starting node
    visited[node] = Node(node.val)
    
    while queue:
        current = queue.popleft()
        
        # Process all neighbors
        for neighbor in current.neighbors:
            if neighbor not in visited:
                # Create clone for unvisited neighbor
                visited[neighbor] = Node(neighbor.val)
                queue.append(neighbor)
            
            # Add neighbor to current node's clone
            visited[current].neighbors.append(visited[neighbor])
    
    return visited[node]


def clone_graph_iterative_dfs(node: Optional[Node]) -> Optional[Node]:
    """
    Iterative DFS approach using stack.
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) - stack + visited map
    
    Algorithm:
    1. Use explicit stack for DFS traversal
    2. Maintain visited map for cloned nodes
    3. Process nodes and their neighbors iteratively
    """
    if not node:
        return None
    
    visited = {}
    stack = [node]
    
    # Create clone of the starting node
    visited[node] = Node(node.val)
    
    while stack:
        current = stack.pop()
        
        # Process all neighbors
        for neighbor in current.neighbors:
            if neighbor not in visited:
                # Create clone for unvisited neighbor
                visited[neighbor] = Node(neighbor.val)
                stack.append(neighbor)
            
            # Add neighbor to current node's clone
            visited[current].neighbors.append(visited[neighbor])
    
    return visited[node]


def clone_graph_two_pass(node: Optional[Node]) -> Optional[Node]:
    """
    Two-pass approach: first create nodes, then establish connections.
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) - visited map
    
    Algorithm:
    1. First pass: traverse and create all nodes
    2. Second pass: establish all connections
    3. Return the cloned starting node
    """
    if not node:
        return None
    
    # First pass: create all nodes
    visited = {}
    
    def create_nodes(current):
        if current in visited:
            return
        
        visited[current] = Node(current.val)
        
        for neighbor in current.neighbors:
            create_nodes(neighbor)
    
    create_nodes(node)
    
    # Second pass: establish connections
    def establish_connections(current):
        if not current:
            return
        
        clone = visited[current]
        
        for neighbor in current.neighbors:
            if visited[neighbor] not in clone.neighbors:
                clone.neighbors.append(visited[neighbor])
    
    visited_connections = set()
    
    def dfs_connections(current):
        if current in visited_connections:
            return
        
        visited_connections.add(current)
        establish_connections(current)
        
        for neighbor in current.neighbors:
            dfs_connections(neighbor)
    
    dfs_connections(node)
    
    return visited[node]


def clone_graph_adjacency_list(node: Optional[Node]) -> Optional[Node]:
    """
    Adjacency list approach.
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V + E) - adjacency list + visited map
    
    Algorithm:
    1. First, build adjacency list representation
    2. Create all nodes based on adjacency list
    3. Establish connections using adjacency list
    """
    if not node:
        return None
    
    # Build adjacency list
    adj_list = {}
    visited_build = set()
    
    def build_adj_list(current):
        if current in visited_build:
            return
        
        visited_build.add(current)
        adj_list[current.val] = [neighbor.val for neighbor in current.neighbors]
        
        for neighbor in current.neighbors:
            build_adj_list(neighbor)
    
    build_adj_list(node)
    
    # Create all nodes
    nodes = {}
    for val in adj_list:
        nodes[val] = Node(val)
    
    # Establish connections
    for val, neighbors in adj_list.items():
        for neighbor_val in neighbors:
            nodes[val].neighbors.append(nodes[neighbor_val])
    
    return nodes[node.val]


def clone_graph_value_based(node: Optional[Node]) -> Optional[Node]:
    """
    Value-based approach using node values as keys.
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) - value-based map
    
    Algorithm:
    1. Use node values as keys in the visited map
    2. Create nodes based on values
    3. Traverse and establish connections
    """
    if not node:
        return None
    
    value_to_node = {}
    
    def dfs(current):
        if current.val in value_to_node:
            return value_to_node[current.val]
        
        # Create clone based on value
        clone = Node(current.val)
        value_to_node[current.val] = clone
        
        # Clone all neighbors
        for neighbor in current.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)


def clone_graph_post_order(node: Optional[Node]) -> Optional[Node]:
    """
    Post-order traversal approach.
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) - visited map + recursion stack
    
    Algorithm:
    1. Use post-order traversal
    2. Process neighbors first, then current node
    3. Establish connections after processing children
    """
    if not node:
        return None
    
    visited = {}
    
    def post_order(current):
        if current in visited:
            return visited[current]
        
        # Create clone first
        clone = Node(current.val)
        visited[current] = clone
        
        # Process neighbors in post-order
        cloned_neighbors = []
        for neighbor in current.neighbors:
            cloned_neighbors.append(post_order(neighbor))
        
        # Establish connections after processing all neighbors
        clone.neighbors = cloned_neighbors
        
        return clone
    
    return post_order(node)


def clone_graph_level_order(node: Optional[Node]) -> Optional[Node]:
    """
    Level-order traversal approach.
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) - queue + visited map
    
    Algorithm:
    1. Use level-order traversal (BFS)
    2. Process nodes level by level
    3. Maintain queue for level-order processing
    """
    if not node:
        return None
    
    visited = {}
    queue = deque([node])
    visited[node] = Node(node.val)
    
    while queue:
        level_size = len(queue)
        
        for _ in range(level_size):
            current = queue.popleft()
            
            for neighbor in current.neighbors:
                if neighbor not in visited:
                    visited[neighbor] = Node(neighbor.val)
                    queue.append(neighbor)
                
                visited[current].neighbors.append(visited[neighbor])
    
    return visited[node]


def clone_graph_memoization(node: Optional[Node]) -> Optional[Node]:
    """
    Memoization approach with caching.
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) - memoization cache
    
    Algorithm:
    1. Use memoization to cache cloned nodes
    2. Avoid recomputation for already cloned nodes
    3. Recursive approach with caching
    """
    if not node:
        return None
    
    memo = {}
    
    def clone_with_memo(current):
        if not current:
            return None
        
        if current in memo:
            return memo[current]
        
        # Create clone
        clone = Node(current.val)
        memo[current] = clone
        
        # Clone neighbors with memoization
        for neighbor in current.neighbors:
            clone.neighbors.append(clone_with_memo(neighbor))
        
        return clone
    
    return clone_with_memo(node)


def clone_graph_recursive_template(node: Optional[Node]) -> Optional[Node]:
    """
    Recursive template approach.
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) - recursion stack + visited map
    
    Algorithm:
    1. Use recursive template pattern
    2. Define base case and recursive case clearly
    3. Maintain visited state
    """
    def clone_recursive(current, visited):
        # Base case
        if not current:
            return None
        
        if current in visited:
            return visited[current]
        
        # Create clone
        clone = Node(current.val)
        visited[current] = clone
        
        # Recursive case
        for neighbor in current.neighbors:
            clone.neighbors.append(clone_recursive(neighbor, visited))
        
        return clone
    
    return clone_recursive(node, {})


# Test cases
def test_clone_graph():
    """Test all graph cloning approaches."""
    
    def create_test_graph_1():
        """Create graph: [[2,4],[1,3],[2,4],[1,3]]"""
        nodes = [Node(i) for i in range(1, 5)]
        
        nodes[0].neighbors = [nodes[1], nodes[3]]  # Node 1 -> [2, 4]
        nodes[1].neighbors = [nodes[0], nodes[2]]  # Node 2 -> [1, 3]
        nodes[2].neighbors = [nodes[1], nodes[3]]  # Node 3 -> [2, 4]
        nodes[3].neighbors = [nodes[0], nodes[2]]  # Node 4 -> [1, 3]
        
        return nodes[0]
    
    def create_test_graph_2():
        """Create graph: [[]]"""
        node = Node(1)
        return node
    
    def create_test_graph_3():
        """Create graph: [[2],[1]]"""
        node1 = Node(1)
        node2 = Node(2)
        
        node1.neighbors = [node2]
        node2.neighbors = [node1]
        
        return node1
    
    def create_test_graph_4():
        """Create graph: [[2,3],[1,3],[1,2]]"""
        nodes = [Node(i) for i in range(1, 4)]
        
        nodes[0].neighbors = [nodes[1], nodes[2]]  # Node 1 -> [2, 3]
        nodes[1].neighbors = [nodes[0], nodes[2]]  # Node 2 -> [1, 3]
        nodes[2].neighbors = [nodes[0], nodes[1]]  # Node 3 -> [1, 2]
        
        return nodes[0]
    
    def graph_to_adjacency_list(node):
        """Convert graph to adjacency list for comparison."""
        if not node:
            return []
        
        visited = {}
        adj_list = {}
        
        def dfs(current):
            if current in visited:
                return
            
            visited[current] = True
            adj_list[current.val] = sorted([neighbor.val for neighbor in current.neighbors])
            
            for neighbor in current.neighbors:
                dfs(neighbor)
        
        dfs(node)
        
        # Convert to list format
        if not adj_list:
            return []
        
        max_val = max(adj_list.keys())
        result = [[] for _ in range(max_val)]
        
        for val in range(1, max_val + 1):
            if val in adj_list:
                result[val - 1] = adj_list[val]
        
        return result
    
    def verify_clone(original, cloned):
        """Verify that the clone is correct."""
        if original is None and cloned is None:
            return True
        
        if original is None or cloned is None:
            return False
        
        # Check that they're different objects
        if original is cloned:
            return False
        
        # Check that adjacency lists are identical
        original_adj = graph_to_adjacency_list(original)
        cloned_adj = graph_to_adjacency_list(cloned)
        
        return original_adj == cloned_adj
    
    # Test cases
    test_cases = [
        (create_test_graph_1, "4-node cycle graph"),
        (create_test_graph_2, "Single node graph"),
        (create_test_graph_3, "Two-node graph"),
        (create_test_graph_4, "3-node complete graph"),
        (lambda: None, "Empty graph"),
    ]
    
    # Test all approaches
    approaches = [
        ("DFS Recursive", clone_graph_dfs),
        ("BFS", clone_graph_bfs),
        ("Iterative DFS", clone_graph_iterative_dfs),
        ("Two-pass", clone_graph_two_pass),
        ("Adjacency List", clone_graph_adjacency_list),
        ("Value-based", clone_graph_value_based),
        ("Post-order", clone_graph_post_order),
        ("Level-order", clone_graph_level_order),
        ("Memoization", clone_graph_memoization),
        ("Recursive Template", clone_graph_recursive_template),
    ]
    
    print("Testing graph cloning approaches:")
    print("=" * 50)
    
    for i, (create_graph, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        
        original = create_graph()
        expected_adj = graph_to_adjacency_list(original)
        print(f"Expected adjacency list: {expected_adj}")
        
        for name, func in approaches:
            try:
                # Create fresh original graph for each test
                original_fresh = create_graph()
                cloned = func(original_fresh)
                
                if verify_clone(original_fresh, cloned):
                    print(f"✓ {name}: Correct clone")
                else:
                    cloned_adj = graph_to_adjacency_list(cloned)
                    print(f"✗ {name}: Incorrect clone - got {cloned_adj}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    # Create larger test graph for performance testing
    def create_large_graph(n):
        """Create a large connected graph with n nodes."""
        nodes = [Node(i) for i in range(1, n + 1)]
        
        # Create a connected graph (each node connected to next few nodes)
        for i in range(n):
            for j in range(1, min(4, n - i)):
                if i + j < n:
                    nodes[i].neighbors.append(nodes[i + j])
                    nodes[i + j].neighbors.append(nodes[i])
        
        return nodes[0]
    
    large_graph = create_large_graph(50)
    
    import time
    
    print(f"Testing with large graph (50 nodes):")
    for name, func in approaches:
        try:
            start_time = time.time()
            cloned = func(large_graph)
            end_time = time.time()
            
            # Quick verification
            is_correct = verify_clone(large_graph, cloned)
            status = "✓" if is_correct else "✗"
            print(f"{status} {name}: (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")


if __name__ == "__main__":
    test_clone_graph() 