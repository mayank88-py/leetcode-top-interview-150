"""
138. Copy List with Random Pointer

A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

Construct a deep copy of the list. The deep copy should consist of exactly n new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointers of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:
- val: an integer representing Node.val
- random_index: the index of the node (0-indexed) that the random pointer points to, or null if it does not point to any node.

Your code will only be given the head of the original linked list.

Example 1:
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]

Example 2:
Input: head = [[1,1],[2,1]]
Output: [[1,1],[2,1]]

Example 3:
Input: head = [[3,null],[3,0],[3,null]]
Output: [[3,null],[3,0],[3,null]]

Constraints:
- 0 <= n <= 1000
- -10^4 <= Node.val <= 10^4
- Node.random is null or is pointing to some node in the linked list
"""

from typing import Optional, List, Dict


class Node:
    """Definition for a Node with random pointer."""
    def __init__(self, val: int = 0, next: 'Node' = None, random: 'Node' = None):
        self.val = val
        self.next = next
        self.random = random
    
    def __repr__(self):
        """String representation for debugging."""
        result = []
        current = self
        visited = set()
        
        while current and current not in visited:
            random_val = current.random.val if current.random else None
            result.append(f"({current.val}, {random_val})")
            visited.add(current)
            current = current.next
        
        return " -> ".join(result)


def copy_random_list_hashmap(head: Optional[Node]) -> Optional[Node]:
    """
    HashMap approach to track original to copy mapping.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - hashmap to store node mappings
    
    Algorithm:
    1. First pass: create all nodes and store original->copy mapping
    2. Second pass: set next and random pointers using the mapping
    """
    if not head:
        return None
    
    # Map original nodes to copied nodes
    node_map = {}
    
    # First pass: create all nodes
    current = head
    while current:
        node_map[current] = Node(current.val)
        current = current.next
    
    # Second pass: set pointers
    current = head
    while current:
        copy_node = node_map[current]
        
        if current.next:
            copy_node.next = node_map[current.next]
        
        if current.random:
            copy_node.random = node_map[current.random]
        
        current = current.next
    
    return node_map[head]


def copy_random_list_interweave(head: Optional[Node]) -> Optional[Node]:
    """
    Interweave approach: insert copied nodes between original nodes.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - no extra space except for the result
    
    Algorithm:
    1. Create copied nodes and insert them after original nodes
    2. Set random pointers for copied nodes
    3. Separate the interweaved list into original and copied lists
    """
    if not head:
        return None
    
    # Step 1: Create copied nodes and interweave them
    current = head
    while current:
        copy_node = Node(current.val)
        copy_node.next = current.next
        current.next = copy_node
        current = copy_node.next
    
    # Step 2: Set random pointers for copied nodes
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next
    
    # Step 3: Separate the lists
    dummy = Node(0)
    copy_current = dummy
    current = head
    
    while current:
        copy_node = current.next
        current.next = copy_node.next
        copy_current.next = copy_node
        copy_current = copy_node
        current = current.next
    
    return dummy.next


def copy_random_list_recursive(head: Optional[Node]) -> Optional[Node]:
    """
    Recursive approach with memoization.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - recursion stack + memoization
    
    Algorithm:
    1. Use recursion with memoization to avoid creating duplicate nodes
    2. Recursively copy next and random pointers
    """
    def helper(node, memo):
        if not node:
            return None
        
        if node in memo:
            return memo[node]
        
        # Create new node
        copy_node = Node(node.val)
        memo[node] = copy_node
        
        # Recursively set pointers
        copy_node.next = helper(node.next, memo)
        copy_node.random = helper(node.random, memo)
        
        return copy_node
    
    return helper(head, {})


def copy_random_list_array(head: Optional[Node]) -> Optional[Node]:
    """
    Array-based approach for index tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - arrays to store nodes
    
    Algorithm:
    1. Store all original nodes in an array
    2. Create corresponding copied nodes
    3. Set pointers using array indices
    """
    if not head:
        return None
    
    # Store all original nodes
    original_nodes = []
    current = head
    while current:
        original_nodes.append(current)
        current = current.next
    
    # Create copied nodes
    copied_nodes = [Node(node.val) for node in original_nodes]
    
    # Map original nodes to their indices
    node_to_index = {node: i for i, node in enumerate(original_nodes)}
    
    # Set pointers
    for i, original_node in enumerate(original_nodes):
        copied_node = copied_nodes[i]
        
        # Set next pointer
        if original_node.next:
            copied_node.next = copied_nodes[i + 1]
        
        # Set random pointer
        if original_node.random:
            random_index = node_to_index[original_node.random]
            copied_node.random = copied_nodes[random_index]
    
    return copied_nodes[0]


def copy_random_list_dfs(head: Optional[Node]) -> Optional[Node]:
    """
    DFS approach with visited tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - visited set + recursion stack
    
    Algorithm:
    1. Use DFS to traverse the graph
    2. Track visited nodes to avoid cycles
    3. Create nodes on-demand during traversal
    """
    if not head:
        return None
    
    visited = {}
    
    def dfs(node):
        if not node:
            return None
        
        if node in visited:
            return visited[node]
        
        # Create new node
        copy_node = Node(node.val)
        visited[node] = copy_node
        
        # DFS on next and random
        copy_node.next = dfs(node.next)
        copy_node.random = dfs(node.random)
        
        return copy_node
    
    return dfs(head)


def copy_random_list_bfs(head: Optional[Node]) -> Optional[Node]:
    """
    BFS approach using queue.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - queue + visited mapping
    
    Algorithm:
    1. Use BFS to traverse the graph
    2. Create nodes level by level
    3. Set pointers during traversal
    """
    if not head:
        return None
    
    from collections import deque
    
    visited = {}
    queue = deque([head])
    visited[head] = Node(head.val)
    
    while queue:
        current = queue.popleft()
        copy_current = visited[current]
        
        # Process next pointer
        if current.next:
            if current.next not in visited:
                visited[current.next] = Node(current.next.val)
                queue.append(current.next)
            copy_current.next = visited[current.next]
        
        # Process random pointer
        if current.random:
            if current.random not in visited:
                visited[current.random] = Node(current.random.val)
                queue.append(current.random)
            copy_current.random = visited[current.random]
    
    return visited[head]


def copy_random_list_serialization(head: Optional[Node]) -> Optional[Node]:
    """
    Serialization approach: serialize then deserialize.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - serialization data
    
    Algorithm:
    1. Serialize the list with node values and pointer indices
    2. Deserialize to create the copied list
    """
    if not head:
        return None
    
    # Serialize: collect node values and random indices
    nodes = []
    node_to_index = {}
    current = head
    index = 0
    
    # First pass: collect nodes and create index mapping
    while current:
        nodes.append(current)
        node_to_index[current] = index
        current = current.next
        index += 1
    
    # Create serialization data
    serialized = []
    for node in nodes:
        random_index = node_to_index[node.random] if node.random else None
        serialized.append((node.val, random_index))
    
    # Deserialize: create copied list
    if not serialized:
        return None
    
    copied_nodes = [Node(val) for val, _ in serialized]
    
    # Set pointers
    for i, (val, random_index) in enumerate(serialized):
        if i < len(copied_nodes) - 1:
            copied_nodes[i].next = copied_nodes[i + 1]
        
        if random_index is not None:
            copied_nodes[i].random = copied_nodes[random_index]
    
    return copied_nodes[0]


def copy_random_list_node_splitting(head: Optional[Node]) -> Optional[Node]:
    """
    Node splitting approach: split original nodes to track relationships.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Split each node into two: original and copy
    2. Use the split structure to set random pointers
    3. Merge back and separate the lists
    """
    if not head:
        return None
    
    # Step 1: Split each node
    current = head
    while current:
        copy_node = Node(current.val)
        copy_node.next = current.next
        current.next = copy_node
        current = copy_node.next
    
    # Step 2: Set random pointers
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next
    
    # Step 3: Separate the lists
    dummy = Node(0)
    copy_current = dummy
    current = head
    
    while current:
        copy_node = current.next
        current.next = copy_node.next
        copy_current.next = copy_node
        copy_current = copy_node
        current = current.next
    
    return dummy.next


def create_random_list(data: List[List]) -> Optional[Node]:
    """Helper function to create a list with random pointers from test data."""
    if not data:
        return None
    
    # Create nodes first
    nodes = [Node(item[0]) for item in data]
    
    # Set next pointers
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    
    # Set random pointers
    for i, item in enumerate(data):
        if len(item) > 1 and item[1] is not None:
            nodes[i].random = nodes[item[1]]
    
    return nodes[0]


def random_list_to_data(head: Optional[Node]) -> List[List]:
    """Helper function to convert random list to test data format."""
    if not head:
        return []
    
    # First pass: collect nodes and create index mapping
    nodes = []
    node_to_index = {}
    current = head
    index = 0
    
    while current:
        nodes.append(current)
        node_to_index[current] = index
        current = current.next
        index += 1
    
    # Second pass: create data
    result = []
    for node in nodes:
        val = node.val
        random_index = node_to_index[node.random] if node.random else None
        result.append([val, random_index])
    
    return result


def test_copy_random_list():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        [[7,None],[13,0],[11,4],[10,2],[1,0]],
        [[1,1],[2,1]],
        [[3,None],[3,0],[3,None]],
        
        # Edge cases
        [],
        [[1,None]],
        [[1,0]],
        
        # No random pointers
        [[1,None],[2,None],[3,None]],
        
        # All point to same node
        [[1,1],[2,1],[3,1]],
        
        # Chain of random pointers
        [[1,1],[2,2],[3,0]],
        
        # Complex random structure
        [[1,2],[2,0],[3,None],[4,1],[5,3]],
        
        # Self-pointing
        [[1,0],[2,1],[3,2]],
        
        # Multiple self-pointing
        [[1,0],[2,1],[3,2],[4,3]],
        
        # Mixed structure
        [[1,3],[2,None],[3,1],[4,0],[5,2]],
        
        # Large values
        [[100,1],[200,0],[300,None]],
        
        # Negative values
        [[-1,1],[-2,0],[-3,None]],
        
        # Long chain
        [[i,None] for i in range(10)],
        
        # Random circular references
        [[1,4],[2,0],[3,2],[4,1],[5,3]],
    ]
    
    # Test all implementations
    implementations = [
        ("HashMap", copy_random_list_hashmap),
        ("Interweave", copy_random_list_interweave),
        ("Recursive", copy_random_list_recursive),
        ("Array", copy_random_list_array),
        ("DFS", copy_random_list_dfs),
        ("BFS", copy_random_list_bfs),
        ("Serialization", copy_random_list_serialization),
        ("Node Splitting", copy_random_list_node_splitting),
    ]
    
    print("Testing Copy List with Random Pointer implementations:")
    print("=" * 60)
    
    for i, test_data in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_data}")
        
        for name, func in implementations:
            try:
                # Create fresh list for each implementation
                head = create_random_list(test_data)
                result_head = func(head)
                result_data = random_list_to_data(result_head)
                
                status = "✓" if result_data == test_data else "✗"
                print(f"{status} {name}: {result_data}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Analysis:")
    print("=" * 60)
    
    import time
    
    # Generate large test cases
    test_scenarios = [
        ("Small list", [[i, (i+1) % 5 if i < 4 else None] for i in range(5)]),
        ("Medium list", [[i, (i+5) % 20 if i < 15 else None] for i in range(20)]),
        ("Large list", [[i, (i+10) % 50 if i < 40 else None] for i in range(50)]),
        ("No random pointers", [[i, None] for i in range(100)]),
        ("All self-pointing", [[i, i] for i in range(50)]),
    ]
    
    for scenario_name, test_data in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                head = create_random_list(test_data)
                
                start_time = time.time()
                result_head = func(head)
                end_time = time.time()
                
                result_length = len(random_list_to_data(result_head))
                print(f"  {name}: {result_length} nodes in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_copy_random_list() 