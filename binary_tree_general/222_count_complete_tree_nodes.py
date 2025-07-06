"""
LeetCode 222: Count Complete Tree Nodes

Given the root of a complete binary tree, return the number of the nodes in the tree.

According to Wikipedia, every level, except possibly the last, is completely filled 
in a complete binary tree, and all nodes in the last level are as far left as possible. 
It can have between 1 and 2^h nodes inclusive at the last level h.

Design an algorithm that runs in less than O(n) time complexity.

Example 1:
Input: root = [1,2,3,4,5,6]
Output: 6

Example 2:
Input: root = []
Output: 0

Example 3:
Input: root = [1]
Output: 1

Constraints:
- The number of nodes in the tree is in the range [0, 5 * 10^4].
- 0 <= Node.val <= 5 * 10^4
- The tree is guaranteed to be complete.
"""

from typing import Optional, List
from collections import deque
import math


class TreeNode:
    """Binary tree node definition."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def count_nodes_naive(root: Optional[TreeNode]) -> int:
    """
    Naive approach - count all nodes.
    
    Time Complexity: O(n)
    Space Complexity: O(h) where h is tree height
    
    Algorithm:
    1. Recursively count all nodes
    2. Return 1 + left_count + right_count
    """
    if not root:
        return 0
    
    return 1 + count_nodes_naive(root.left) + count_nodes_naive(root.right)


def count_nodes_iterative(root: Optional[TreeNode]) -> int:
    """
    Iterative approach using stack.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use stack to traverse all nodes
    2. Count nodes during traversal
    """
    if not root:
        return 0
    
    stack = [root]
    count = 0
    
    while stack:
        node = stack.pop()
        count += 1
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return count


def count_nodes_level_order(root: Optional[TreeNode]) -> int:
    """
    Level order traversal approach.
    
    Time Complexity: O(n)
    Space Complexity: O(w) where w is max width
    
    Algorithm:
    1. Use BFS to traverse level by level
    2. Count all nodes
    """
    if not root:
        return 0
    
    queue = deque([root])
    count = 0
    
    while queue:
        node = queue.popleft()
        count += 1
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return count


def count_nodes_binary_search(root: Optional[TreeNode]) -> int:
    """
    Binary search approach optimized for complete binary tree.
    
    Time Complexity: O(log²n)
    Space Complexity: O(log n)
    
    Algorithm:
    1. Use complete tree property
    2. Binary search on the last level
    3. Check if nodes exist at specific positions
    """
    if not root:
        return 0
    
    def get_height(node):
        """Get height by going left."""
        height = 0
        while node:
            height += 1
            node = node.left
        return height
    
    def exists(idx, height, node):
        """Check if node exists at given index in last level."""
        left, right = 0, 2**(height-1) - 1
        
        for _ in range(height - 1):
            mid = (left + right) // 2
            if idx <= mid:
                node = node.left
                right = mid
            else:
                node = node.right
                left = mid + 1
        
        return node is not None
    
    height = get_height(root)
    if height == 1:
        return 1
    
    # Nodes in all levels except last
    upper_count = 2**(height-1) - 1
    
    # Binary search for last level
    left, right = 0, 2**(height-1) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if exists(mid, height, root):
            left = mid + 1
        else:
            right = mid - 1
    
    return upper_count + left


def count_nodes_recursive_optimized(root: Optional[TreeNode]) -> int:
    """
    Recursive approach optimized for complete tree.
    
    Time Complexity: O(log²n)
    Space Complexity: O(log n)
    
    Algorithm:
    1. Check if left and right heights are equal
    2. If equal, it's a perfect tree: 2^h - 1
    3. If not equal, recursively count subtrees
    """
    if not root:
        return 0
    
    def get_height(node, go_left=True):
        """Get height going in specified direction."""
        height = 0
        while node:
            height += 1
            node = node.left if go_left else node.right
        return height
    
    left_height = get_height(root.left, True)
    right_height = get_height(root.right, False)
    
    if left_height == right_height:
        # Perfect binary tree
        return (1 << (left_height + 1)) - 1
    else:
        # Recursively count
        return 1 + count_nodes_recursive_optimized(root.left) + \
               count_nodes_recursive_optimized(root.right)


def count_nodes_path_finding(root: Optional[TreeNode]) -> int:
    """
    Path finding approach using complete tree properties.
    
    Time Complexity: O(log²n)
    Space Complexity: O(log n)
    
    Algorithm:
    1. Find the rightmost node in the last level
    2. Use binary representation to navigate
    3. Count nodes based on complete tree structure
    """
    if not root:
        return 0
    
    def get_depth(node):
        depth = 0
        while node.left:
            depth += 1
            node = node.left
        return depth
    
    depth = get_depth(root)
    if depth == 0:
        return 1
    
    # Binary search for rightmost node in last level
    left, right = 1, 2**depth
    
    while left <= right:
        mid = (left + right) // 2
        
        # Navigate to position mid in last level
        node = root
        path = bin(mid)[3:]  # Remove '0b1' prefix
        
        for bit in path:
            if bit == '0':
                node = node.left
            else:
                node = node.right
        
        if node:
            left = mid + 1
        else:
            right = mid - 1
    
    # Total nodes = nodes in complete levels + nodes in last level
    return 2**depth - 1 + right


def count_nodes_bit_manipulation(root: Optional[TreeNode]) -> int:
    """
    Bit manipulation approach.
    
    Time Complexity: O(log²n)
    Space Complexity: O(log n)
    
    Algorithm:
    1. Use bit operations for efficient calculations
    2. Leverage complete tree properties
    3. Navigate using binary representation
    """
    if not root:
        return 0
    
    def get_height(node):
        height = 0
        while node:
            height += 1
            node = node.left
        return height
    
    height = get_height(root)
    
    def node_exists(idx):
        """Check if node exists at given index."""
        node = root
        # Start from second bit (skip root bit)
        for i in range(height - 2, -1, -1):
            if (idx >> i) & 1:
                node = node.right
            else:
                node = node.left
        return node is not None
    
    # Binary search on last level
    left, right = 0, (1 << (height - 1)) - 1
    
    while left <= right:
        mid = (left + right) >> 1
        if node_exists(mid):
            left = mid + 1
        else:
            right = mid - 1
    
    return ((1 << (height - 1)) - 1) + left


def count_nodes_mathematical(root: Optional[TreeNode]) -> int:
    """
    Mathematical approach using complete tree properties.
    
    Time Complexity: O(log²n)
    Space Complexity: O(1)
    
    Algorithm:
    1. Use mathematical formulas for complete trees
    2. Calculate nodes without explicit traversal
    3. Optimize using bit operations
    """
    if not root:
        return 0
    
    node = root
    height = 0
    
    # Calculate height
    while node:
        height += 1
        node = node.left
    
    if height == 1:
        return 1
    
    def path_exists(path, length):
        """Check if path exists from root."""
        node = root
        for i in range(length - 1, -1, -1):
            if (path >> i) & 1:
                node = node.right
            else:
                node = node.left
            if not node:
                return False
        return True
    
    # Binary search for rightmost node in last level
    max_nodes_last_level = 1 << (height - 1)
    left, right = 0, max_nodes_last_level - 1
    
    while left <= right:
        mid = (left + right) >> 1
        if path_exists(mid, height - 1):
            left = mid + 1
        else:
            right = mid - 1
    
    # Nodes in complete levels + nodes in last level
    return (1 << (height - 1)) - 1 + left


def count_nodes_divide_conquer(root: Optional[TreeNode]) -> int:
    """
    Divide and conquer approach.
    
    Time Complexity: O(log²n)
    Space Complexity: O(log n)
    
    Algorithm:
    1. Divide tree into left and right subtrees
    2. Use complete tree properties to optimize
    3. Combine results efficiently
    """
    if not root:
        return 0
    
    def height(node):
        h = 0
        while node:
            h += 1
            node = node.left
        return h
    
    left_h = height(root.left)
    right_h = height(root.right)
    
    if left_h == right_h:
        # Left subtree is perfect, right subtree is complete
        return (1 << left_h) + count_nodes_divide_conquer(root.right)
    else:
        # Right subtree is perfect, left subtree is complete
        return count_nodes_divide_conquer(root.left) + (1 << right_h)


def count_nodes_tail_recursive(root: Optional[TreeNode]) -> int:
    """
    Tail recursive approach.
    
    Time Complexity: O(log²n)
    Space Complexity: O(log n)
    
    Algorithm:
    1. Use tail recursion for optimization
    2. Accumulate count during recursion
    3. Leverage complete tree structure
    """
    def count_helper(node, acc=0):
        if not node:
            return acc
        
        def height(n):
            h = 0
            while n:
                h += 1
                n = n.left
            return h
        
        left_h = height(node.left)
        right_h = height(node.right)
        
        if left_h == right_h:
            # Left subtree is perfect
            new_acc = acc + (1 << left_h)
            return count_helper(node.right, new_acc)
        else:
            # Right subtree is perfect
            new_acc = acc + (1 << right_h)
            return count_helper(node.left, new_acc)
    
    if not root:
        return 0
    
    return count_helper(root)


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


def test_count_nodes():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([], 0),
        ([1], 1),
        ([1, 2], 2),
        ([1, 2, 3], 3),
        
        # Small complete trees
        ([1, 2, 3, 4], 4),
        ([1, 2, 3, 4, 5], 5),
        ([1, 2, 3, 4, 5, 6], 6),
        ([1, 2, 3, 4, 5, 6, 7], 7),
        
        # Medium complete trees
        ([1, 2, 3, 4, 5, 6, 7, 8], 8),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 9),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 11),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 12),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 13),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 14),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 15),
        
        # Perfect binary trees
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 16),
        
        # Larger complete trees
        (list(range(1, 32)), 31),  # 31 nodes
        (list(range(1, 64)), 63),  # 63 nodes (perfect tree)
        (list(range(1, 100)), 99), # 99 nodes
    ]
    
    # Test all implementations
    implementations = [
        ("Naive", count_nodes_naive),
        ("Iterative", count_nodes_iterative),
        ("Level Order", count_nodes_level_order),
        ("Binary Search", count_nodes_binary_search),
        ("Recursive Optimized", count_nodes_recursive_optimized),
        ("Path Finding", count_nodes_path_finding),
        ("Bit Manipulation", count_nodes_bit_manipulation),
        ("Mathematical", count_nodes_mathematical),
        ("Divide Conquer", count_nodes_divide_conquer),
        ("Tail Recursive", count_nodes_tail_recursive),
    ]
    
    print("Testing Count Complete Tree Nodes implementations:")
    print("=" * 75)
    
    for i, (nodes, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Tree:     {nodes[:20]}{'...' if len(nodes) > 20 else ''}")
        print(f"Expected: {expected}")
        
        tree = build_test_tree(nodes)
        
        for name, func in implementations:
            try:
                result = func(tree)
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 75)
    print("Performance Analysis:")
    print("=" * 75)
    
    import time
    
    # Generate large complete trees
    def generate_complete_tree(size):
        """Generate complete tree with given number of nodes."""
        return list(range(1, size + 1))
    
    test_scenarios = [
        ("Small tree", 63),      # 2^6 - 1
        ("Medium tree", 255),    # 2^8 - 1
        ("Large tree", 1023),    # 2^10 - 1
        ("Very large", 4095),    # 2^12 - 1
    ]
    
    for scenario_name, size in test_scenarios:
        print(f"\n{scenario_name} ({size} nodes):")
        nodes = generate_complete_tree(size)
        tree = build_test_tree(nodes)
        
        for name, func in implementations:
            try:
                start_time = time.time()
                result = func(tree)
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
                print(f"  {name}: {elapsed:.2f} ms (result: {result})")
            except Exception as e:
                print(f"  {name}: Error - {e}")
    
    print(f"\n{'='*75}")
    print("Algorithm Complexity Analysis:")
    print("="*75)
    print("1. Naive:               O(n) time, O(h) space")
    print("2. Iterative:           O(n) time, O(h) space")
    print("3. Level Order:         O(n) time, O(w) space")
    print("4. Binary Search:       O(log²n) time, O(log n) space")
    print("5. Recursive Optimized: O(log²n) time, O(log n) space")
    print("6. Path Finding:        O(log²n) time, O(log n) space")
    print("7. Bit Manipulation:    O(log²n) time, O(log n) space")
    print("8. Mathematical:        O(log²n) time, O(1) space")
    print("9. Divide Conquer:      O(log²n) time, O(log n) space")
    print("10. Tail Recursive:     O(log²n) time, O(log n) space")
    
    print(f"\n{'='*75}")
    print("Key Insights:")
    print("="*75)
    print("• Complete binary tree property enables O(log²n) solutions")
    print("• Binary search on last level is the key optimization")
    print("• Height can be found in O(log n) by going left")
    print("• Perfect subtrees can be counted using 2^h - 1 formula")
    print("• Bit manipulation provides elegant path navigation")
    print("• Mathematical approach achieves O(1) space complexity")


if __name__ == "__main__":
    test_count_nodes() 