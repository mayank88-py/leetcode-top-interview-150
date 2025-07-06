"""
LeetCode 257: Binary Tree Paths

Given the root of a binary tree, return all root-to-leaf paths in any order.

A leaf is a node with no children.

Example 1:
Input: root = [1,2,3,null,5]
Output: ["1->2->5","1->3"]

Example 2:
Input: root = [1]
Output: ["1"]

Constraints:
- The number of nodes in the tree is in the range [1, 100].
- -100 <= Node.val <= 100
"""

from typing import Optional, List
from collections import deque


class TreeNode:
    """Binary tree node definition."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def binary_tree_paths_recursive(root: Optional[TreeNode]) -> List[str]:
    """
    Recursive DFS approach.
    
    Time Complexity: O(n * h) where h is height
    Space Complexity: O(n * h) for storing paths
    
    Algorithm:
    1. Use DFS to traverse tree
    2. Build path string as we go
    3. Add to result when reaching leaf
    """
    if not root:
        return []
    
    result = []
    
    def dfs(node, path):
        if not node:
            return
        
        # Add current node to path
        if path:
            path += "->" + str(node.val)
        else:
            path = str(node.val)
        
        # If leaf node, add path to result
        if not node.left and not node.right:
            result.append(path)
            return
        
        # Continue to children
        dfs(node.left, path)
        dfs(node.right, path)
    
    dfs(root, "")
    return result


def binary_tree_paths_backtracking(root: Optional[TreeNode]) -> List[str]:
    """
    Backtracking approach with path list.
    
    Time Complexity: O(n * h)
    Space Complexity: O(h) for recursion stack
    
    Algorithm:
    1. Maintain current path as list
    2. Use backtracking to add/remove nodes
    3. Convert to string when reaching leaf
    """
    if not root:
        return []
    
    result = []
    
    def backtrack(node, path):
        if not node:
            return
        
        # Add current node to path
        path.append(str(node.val))
        
        # If leaf, add path to result
        if not node.left and not node.right:
            result.append("->".join(path))
        else:
            # Continue to children
            backtrack(node.left, path)
            backtrack(node.right, path)
        
        # Backtrack - remove current node
        path.pop()
    
    backtrack(root, [])
    return result


def binary_tree_paths_iterative_dfs(root: Optional[TreeNode]) -> List[str]:
    """
    Iterative DFS using stack.
    
    Time Complexity: O(n * h)
    Space Complexity: O(n * h)
    
    Algorithm:
    1. Use stack to store (node, path) pairs
    2. Process nodes in DFS order
    3. Add to result when reaching leaf
    """
    if not root:
        return []
    
    result = []
    stack = [(root, str(root.val))]
    
    while stack:
        node, path = stack.pop()
        
        # If leaf node, add path to result
        if not node.left and not node.right:
            result.append(path)
        
        # Add children to stack
        if node.right:
            stack.append((node.right, path + "->" + str(node.right.val)))
        if node.left:
            stack.append((node.left, path + "->" + str(node.left.val)))
    
    return result


def binary_tree_paths_iterative_bfs(root: Optional[TreeNode]) -> List[str]:
    """
    Iterative BFS using queue.
    
    Time Complexity: O(n * h)
    Space Complexity: O(n * h)
    
    Algorithm:
    1. Use queue for level-order traversal
    2. Store (node, path) pairs in queue
    3. Process level by level
    """
    if not root:
        return []
    
    result = []
    queue = deque([(root, str(root.val))])
    
    while queue:
        node, path = queue.popleft()
        
        # If leaf node, add path to result
        if not node.left and not node.right:
            result.append(path)
        
        # Add children to queue
        if node.left:
            queue.append((node.left, path + "->" + str(node.left.val)))
        if node.right:
            queue.append((node.right, path + "->" + str(node.right.val)))
    
    return result


def binary_tree_paths_preorder(root: Optional[TreeNode]) -> List[str]:
    """
    Preorder traversal approach.
    
    Time Complexity: O(n * h)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use preorder traversal pattern
    2. Build path during traversal
    3. Collect paths at leaf nodes
    """
    if not root:
        return []
    
    result = []
    
    def preorder(node, path):
        if not node:
            return
        
        # Process current node
        current_path = path + [str(node.val)]
        
        # If leaf, add to result
        if not node.left and not node.right:
            result.append("->".join(current_path))
        
        # Traverse children
        preorder(node.left, current_path)
        preorder(node.right, current_path)
    
    preorder(root, [])
    return result


def binary_tree_paths_path_builder(root: Optional[TreeNode]) -> List[str]:
    """
    Path builder approach with string manipulation.
    
    Time Complexity: O(n * h)
    Space Complexity: O(h)
    
    Algorithm:
    1. Build path string incrementally
    2. Use string slicing for backtracking
    3. Optimize string operations
    """
    if not root:
        return []
    
    result = []
    
    def build_paths(node, path_str):
        if not node:
            return
        
        # Build current path
        if path_str:
            current_path = path_str + "->" + str(node.val)
        else:
            current_path = str(node.val)
        
        # If leaf, add to result
        if not node.left and not node.right:
            result.append(current_path)
            return
        
        # Continue building
        build_paths(node.left, current_path)
        build_paths(node.right, current_path)
    
    build_paths(root, "")
    return result


def binary_tree_paths_generator(root: Optional[TreeNode]) -> List[str]:
    """
    Generator-based approach.
    
    Time Complexity: O(n * h)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use generator to yield paths
    2. Memory efficient for large trees
    3. Lazy evaluation of paths
    """
    if not root:
        return []
    
    def generate_paths(node, path):
        if not node:
            return
        
        current_path = path + [str(node.val)]
        
        if not node.left and not node.right:
            yield "->".join(current_path)
        else:
            yield from generate_paths(node.left, current_path)
            yield from generate_paths(node.right, current_path)
    
    return list(generate_paths(root, []))


def binary_tree_paths_morris(root: Optional[TreeNode]) -> List[str]:
    """
    Morris traversal inspired approach.
    
    Time Complexity: O(n * h)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use Morris traversal concepts
    2. Build paths without recursion stack
    3. Maintain parent pointers implicitly
    """
    if not root:
        return []
    
    result = []
    
    def find_paths_morris(node):
        if not node:
            return
        
        # Simple DFS implementation (Morris concept applied differently)
        stack = [(node, [str(node.val)])]
        
        while stack:
            current, path = stack.pop()
            
            if not current.left and not current.right:
                result.append("->".join(path))
            
            if current.right:
                stack.append((current.right, path + [str(current.right.val)]))
            if current.left:
                stack.append((current.left, path + [str(current.left.val)]))
    
    find_paths_morris(root)
    return result


def binary_tree_paths_divide_conquer(root: Optional[TreeNode]) -> List[str]:
    """
    Divide and conquer approach.
    
    Time Complexity: O(n * h)
    Space Complexity: O(n * h)
    
    Algorithm:
    1. Divide problem into subproblems
    2. Get paths from left and right subtrees
    3. Combine results with current node
    """
    if not root:
        return []
    
    # Base case: leaf node
    if not root.left and not root.right:
        return [str(root.val)]
    
    paths = []
    
    # Get paths from left subtree
    if root.left:
        left_paths = binary_tree_paths_divide_conquer(root.left)
        for path in left_paths:
            paths.append(str(root.val) + "->" + path)
    
    # Get paths from right subtree
    if root.right:
        right_paths = binary_tree_paths_divide_conquer(root.right)
        for path in right_paths:
            paths.append(str(root.val) + "->" + path)
    
    return paths


def binary_tree_paths_functional(root: Optional[TreeNode]) -> List[str]:
    """
    Functional programming approach.
    
    Time Complexity: O(n * h)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use functional programming concepts
    2. Immutable path passing
    3. List comprehensions and map/filter
    """
    if not root:
        return []
    
    def get_paths(node, prefix=""):
        if not node:
            return []
        
        current_path = prefix + str(node.val)
        
        if not node.left and not node.right:
            return [current_path]
        
        left_paths = get_paths(node.left, current_path + "->")
        right_paths = get_paths(node.right, current_path + "->")
        
        return left_paths + right_paths
    
    return get_paths(root)


def binary_tree_paths_tail_recursive(root: Optional[TreeNode]) -> List[str]:
    """
    Tail recursive approach.
    
    Time Complexity: O(n * h)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use tail recursion optimization
    2. Accumulate results in parameter
    3. Reduce stack usage
    """
    if not root:
        return []
    
    def tail_recursive_helper(node, path, result):
        if not node:
            return result
        
        current_path = path + [str(node.val)]
        
        if not node.left and not node.right:
            result.append("->".join(current_path))
            return result
        
        if node.left:
            tail_recursive_helper(node.left, current_path, result)
        if node.right:
            tail_recursive_helper(node.right, current_path, result)
        
        return result
    
    return tail_recursive_helper(root, [], [])


def build_test_tree(nodes: List[Optional[int]]) -> Optional[TreeNode]:
    """Build tree from list representation."""
    if not nodes or nodes[0] is None:
        return None
    
    root = TreeNode(nodes[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(nodes):
        node = queue.popleft()
        
        if i < len(nodes) and nodes[i] is not None:
            node.left = TreeNode(nodes[i])
            queue.append(node.left)
        i += 1
        
        if i < len(nodes) and nodes[i] is not None:
            node.right = TreeNode(nodes[i])
            queue.append(node.right)
        i += 1
    
    return root


def test_binary_tree_paths():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1], ["1"]),
        ([1, 2], ["1->2"]),
        ([1, None, 2], ["1->2"]),
        
        # Given examples
        ([1, 2, 3, None, 5], ["1->2->5", "1->3"]),
        
        # Balanced trees
        ([1, 2, 3], ["1->2", "1->3"]),
        ([1, 2, 3, 4, 5, 6, 7], ["1->2->4", "1->2->5", "1->3->6", "1->3->7"]),
        
        # Left skewed
        ([1, 2, None, 3, None, None, None, 4], ["1->2->3->4"]),
        
        # Right skewed
        ([1, None, 2, None, 3, None, 4], ["1->2->3->4"]),
        
        # Complex trees
        ([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1], 
         ["5->4->11->7", "5->4->11->2", "5->8->13", "5->8->4->1"]),
        
        # Negative values
        ([-1, -2, -3], ["-1->-2", "-1->-3"]),
        ([1, -2, 3, 4, 5], ["1->-2->4", "1->-2->5", "1->3"]),
        
        # Single child patterns
        ([1, 2, None, 3, None, None, None, None, None, None, None, None, None, None, 4], 
         ["1->2->3->4"]),
        
        # Mixed patterns
        ([1, 2, 3, 4, None, None, 5], ["1->2->4", "1->3->5"]),
        ([1, 2, 3, None, 4, 5], ["1->2->4", "1->3->5"]),
        
        # Larger values
        ([100, 50, 150, 25, 75], ["100->50->25", "100->50->75", "100->150"]),
        
        # Deep tree
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
         ["1->2->4->8", "1->2->4->9", "1->2->5->10", "1->2->5->11", 
          "1->3->6->12", "1->3->6->13", "1->3->7->14", "1->3->7->15"]),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", binary_tree_paths_recursive),
        ("Backtracking", binary_tree_paths_backtracking),
        ("Iterative DFS", binary_tree_paths_iterative_dfs),
        ("Iterative BFS", binary_tree_paths_iterative_bfs),
        ("Preorder", binary_tree_paths_preorder),
        ("Path Builder", binary_tree_paths_path_builder),
        ("Generator", binary_tree_paths_generator),
        ("Morris", binary_tree_paths_morris),
        ("Divide Conquer", binary_tree_paths_divide_conquer),
        ("Functional", binary_tree_paths_functional),
        ("Tail Recursive", binary_tree_paths_tail_recursive),
    ]
    
    print("Testing Binary Tree Paths implementations:")
    print("=" * 75)
    
    for i, (nodes, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Tree:     {nodes}")
        print(f"Expected: {sorted(expected)}")
        
        tree = build_test_tree(nodes)
        
        for name, func in implementations:
            try:
                result = func(tree)
                # Sort both lists for comparison (order might differ)
                status = "✓" if sorted(result) == sorted(expected) else "✗"
                print(f"{status} {name}: {sorted(result)}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 75)
    print("Performance Analysis:")
    print("=" * 75)
    
    import time
    
    # Generate large test cases
    def generate_test_tree(height):
        """Generate a complete binary tree of given height."""
        nodes = []
        for i in range(2**height - 1):
            nodes.append(i + 1)
        return nodes
    
    test_scenarios = [
        ("Small tree", 4),   # Height 4: 15 nodes
        ("Medium tree", 6),  # Height 6: 63 nodes  
        ("Large tree", 8),   # Height 8: 255 nodes
        ("Very large", 10),  # Height 10: 1023 nodes
    ]
    
    for scenario_name, height in test_scenarios:
        print(f"\n{scenario_name} (height {height}, {2**height - 1} nodes):")
        nodes = generate_test_tree(height)
        tree = build_test_tree(nodes)
        
        for name, func in implementations:
            try:
                start_time = time.time()
                result = func(tree)
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
                num_paths = len(result)
                print(f"  {name}: {elapsed:.2f} ms ({num_paths} paths)")
            except Exception as e:
                print(f"  {name}: Error - {e}")
    
    print(f"\n{'='*75}")
    print("Algorithm Complexity Analysis:")
    print("="*75)
    print("1. Recursive:        O(n*h) time, O(h) space")
    print("2. Backtracking:     O(n*h) time, O(h) space")
    print("3. Iterative DFS:    O(n*h) time, O(n*h) space")
    print("4. Iterative BFS:    O(n*h) time, O(n*h) space")
    print("5. Preorder:         O(n*h) time, O(h) space")
    print("6. Path Builder:     O(n*h) time, O(h) space")
    print("7. Generator:        O(n*h) time, O(h) space")
    print("8. Morris:           O(n*h) time, O(h) space")
    print("9. Divide Conquer:   O(n*h) time, O(n*h) space")
    print("10. Functional:      O(n*h) time, O(h) space")
    print("11. Tail Recursive:  O(n*h) time, O(h) space")
    
    print(f"\n{'='*75}")
    print("Key Insights:")
    print("="*75)
    print("• All approaches have same time complexity O(n*h)")
    print("• Space complexity varies: recursive O(h), iterative O(n*h)")
    print("• Backtracking saves space by reusing path array")
    print("• Generator approach is memory efficient for large results")
    print("• String operations can be expensive; list operations often better")
    print("• In complete binary tree: h = log(n), so complexity is O(n*log(n))")


if __name__ == "__main__":
    test_binary_tree_paths() 