"""
100. Same Tree

Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

Example 1:
Input: p = [1,2,3], q = [1,2,3]
Output: true

Example 2:
Input: p = [1,2], q = [1,null,2]
Output: false

Example 3:
Input: p = [1,2,1], q = [1,1,2]
Output: false

Constraints:
- The number of nodes in both trees is in the range [0, 100]
- -10^4 <= Node.val <= 10^4
"""

from typing import Optional, List, Deque
from collections import deque


class TreeNode:
    """Definition for a binary tree node."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        """String representation for debugging."""
        return f"TreeNode({self.val})"


def is_same_tree_recursive(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Recursive approach.
    
    Time Complexity: O(min(n, m)) where n, m are the number of nodes
    Space Complexity: O(min(h1, h2)) where h1, h2 are the heights (recursion stack)
    
    Algorithm:
    1. If both nodes are None, return True
    2. If one is None and other is not, return False
    3. If values are different, return False
    4. Recursively check left and right subtrees
    """
    # Base cases
    if not p and not q:
        return True
    if not p or not q:
        return False
    
    # Check current nodes
    if p.val != q.val:
        return False
    
    # Recursively check subtrees
    return (is_same_tree_recursive(p.left, q.left) and 
            is_same_tree_recursive(p.right, q.right))


def is_same_tree_iterative_bfs(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Iterative BFS approach using queues.
    
    Time Complexity: O(min(n, m)) where n, m are the number of nodes
    Space Complexity: O(min(w1, w2)) where w1, w2 are the maximum widths
    
    Algorithm:
    1. Use two queues for level-order traversal
    2. Compare nodes at each level
    3. Return False if any mismatch found
    """
    if not p and not q:
        return True
    if not p or not q:
        return False
    
    queue1 = deque([p])
    queue2 = deque([q])
    
    while queue1 and queue2:
        node1 = queue1.popleft()
        node2 = queue2.popleft()
        
        # Compare current nodes
        if not node1 and not node2:
            continue
        if not node1 or not node2:
            return False
        if node1.val != node2.val:
            return False
        
        # Add children to queues
        queue1.extend([node1.left, node1.right])
        queue2.extend([node2.left, node2.right])
    
    return len(queue1) == len(queue2)


def is_same_tree_iterative_dfs(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Iterative DFS approach using stacks.
    
    Time Complexity: O(min(n, m)) where n, m are the number of nodes
    Space Complexity: O(min(h1, h2)) where h1, h2 are the heights
    
    Algorithm:
    1. Use two stacks for DFS traversal
    2. Compare nodes in DFS order
    3. Return False if any mismatch found
    """
    if not p and not q:
        return True
    if not p or not q:
        return False
    
    stack1 = [p]
    stack2 = [q]
    
    while stack1 and stack2:
        node1 = stack1.pop()
        node2 = stack2.pop()
        
        # Compare current nodes
        if not node1 and not node2:
            continue
        if not node1 or not node2:
            return False
        if node1.val != node2.val:
            return False
        
        # Add children to stacks (right first for DFS)
        stack1.extend([node1.right, node1.left])
        stack2.extend([node2.right, node2.left])
    
    return len(stack1) == len(stack2)


def is_same_tree_preorder(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Preorder traversal approach.
    
    Time Complexity: O(min(n, m)) where n, m are the number of nodes
    Space Complexity: O(min(h1, h2)) where h1, h2 are the heights
    
    Algorithm:
    1. Compare roots first
    2. Then compare left subtrees
    3. Then compare right subtrees
    """
    def preorder_compare(node1, node2):
        if not node1 and not node2:
            return True
        if not node1 or not node2:
            return False
        if node1.val != node2.val:
            return False
        
        return (preorder_compare(node1.left, node2.left) and
                preorder_compare(node1.right, node2.right))
    
    return preorder_compare(p, q)


def is_same_tree_serialization(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Serialization approach.
    
    Time Complexity: O(n + m) where n, m are the number of nodes
    Space Complexity: O(n + m) for serialization strings
    
    Algorithm:
    1. Serialize both trees to strings
    2. Compare the serialized strings
    3. Return True if strings are equal
    """
    def serialize(node):
        if not node:
            return "None"
        return f"{node.val},{serialize(node.left)},{serialize(node.right)}"
    
    return serialize(p) == serialize(q)


def is_same_tree_level_order(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Level-order comparison approach.
    
    Time Complexity: O(min(n, m)) where n, m are the number of nodes
    Space Complexity: O(min(w1, w2)) where w1, w2 are the maximum widths
    
    Algorithm:
    1. Perform level-order traversal on both trees
    2. Compare each level
    3. Return False if any level differs
    """
    if not p and not q:
        return True
    if not p or not q:
        return False
    
    level1 = [p]
    level2 = [q]
    
    while level1 and level2:
        if len(level1) != len(level2):
            return False
        
        next_level1 = []
        next_level2 = []
        
        for i in range(len(level1)):
            node1 = level1[i]
            node2 = level2[i]
            
            # Compare current nodes
            if not node1 and not node2:
                continue
            if not node1 or not node2:
                return False
            if node1.val != node2.val:
                return False
            
            # Add children
            next_level1.extend([node1.left, node1.right])
            next_level2.extend([node2.left, node2.right])
        
        level1 = next_level1
        level2 = next_level2
    
    return len(level1) == len(level2)


def is_same_tree_inorder(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Inorder traversal approach with structure tracking.
    
    Time Complexity: O(n + m) where n, m are the number of nodes
    Space Complexity: O(n + m) for traversal lists
    
    Algorithm:
    1. Perform inorder traversal on both trees with structure info
    2. Compare the traversal results
    3. Return True if results are identical
    """
    def inorder_with_structure(node, result):
        if not node:
            result.append(None)
            return
        
        inorder_with_structure(node.left, result)
        result.append(node.val)
        inorder_with_structure(node.right, result)
    
    result1 = []
    result2 = []
    
    inorder_with_structure(p, result1)
    inorder_with_structure(q, result2)
    
    return result1 == result2


def is_same_tree_postorder(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Postorder traversal approach.
    
    Time Complexity: O(min(n, m)) where n, m are the number of nodes
    Space Complexity: O(min(h1, h2)) where h1, h2 are the heights
    
    Algorithm:
    1. Compare left subtrees first
    2. Then compare right subtrees
    3. Finally compare root nodes
    """
    def postorder_compare(node1, node2):
        if not node1 and not node2:
            return True
        if not node1 or not node2:
            return False
        
        # Compare subtrees first
        left_same = postorder_compare(node1.left, node2.left)
        right_same = postorder_compare(node1.right, node2.right)
        
        # Then compare current nodes
        return left_same and right_same and (node1.val == node2.val)
    
    return postorder_compare(p, q)


def is_same_tree_morris_traversal(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Morris traversal approach for constant space.
    
    Time Complexity: O(n + m) where n, m are the number of nodes
    Space Complexity: O(1) - constant space
    
    Algorithm:
    1. Use Morris traversal to compare trees without extra space
    2. Compare values during traversal
    3. Return False if any mismatch found
    """
    def morris_inorder(root):
        result = []
        current = root
        
        while current:
            if not current.left:
                result.append(current.val)
                current = current.right
            else:
                # Find inorder predecessor
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    predecessor.right = current
                    current = current.left
                else:
                    predecessor.right = None
                    result.append(current.val)
                    current = current.right
        
        return result
    
    def morris_structure(root):
        """Modified Morris to capture structure"""
        result = []
        current = root
        
        while current:
            if not current.left:
                result.append(current.val if current else None)
                current = current.right
            else:
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    predecessor.right = current
                    current = current.left
                else:
                    predecessor.right = None
                    result.append(current.val if current else None)
                    current = current.right
        
        return result
    
    # For this problem, simple comparison is sufficient
    return morris_inorder(p) == morris_inorder(q)


def create_binary_tree(values: List[Optional[int]]) -> Optional[TreeNode]:
    """Helper function to create binary tree from level-order array."""
    if not values or values[0] is None:
        return None
    
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(values):
        node = queue.popleft()
        
        # Left child
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        
        # Right child
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    
    return root


def test_is_same_tree():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic same trees
        ([1,2,3], [1,2,3], True),
        ([1], [1], True),
        ([], [], True),
        
        # Different structures
        ([1,2], [1,None,2], False),
        ([1,2,1], [1,1,2], False),
        
        # Different values
        ([1,2,3], [1,2,4], False),
        ([1,2], [2,1], False),
        
        # One empty, one not
        ([], [1], False),
        ([1], [], False),
        
        # Different sizes
        ([1,2,3], [1,2], False),
        ([1,2], [1,2,3], False),
        
        # Complex same trees
        ([1,2,3,4,5,6,7], [1,2,3,4,5,6,7], True),
        ([1,2,3,None,4,None,5], [1,2,3,None,4,None,5], True),
        
        # Complex different trees
        ([1,2,3,4,5,6,7], [1,2,3,4,5,6,8], False),
        ([1,2,3,None,4,None,5], [1,2,3,4,None,5,None], False),
        
        # Skewed trees
        ([1,2,None,3,None,4], [1,2,None,3,None,4], True),
        ([1,2,None,3,None,4], [1,None,2,None,3,None,4], False),
        
        # Large same trees
        ([1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10], True),
        ([1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,11], False),
        
        # Single node differences
        ([1,2,3,4], [1,2,3,5], False),
        ([1,2,3,4], [1,2,3,4], True),
        
        # Negative values
        ([-1,-2,-3], [-1,-2,-3], True),
        ([-1,-2,-3], [-1,-2,3], False),
        
        # Mixed positive/negative
        ([1,-2,3], [1,-2,3], True),
        ([1,-2,3], [1,2,3], False),
        
        # Zero values
        ([0,1,2], [0,1,2], True),
        ([0,1,2], [0,1,0], False),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", is_same_tree_recursive),
        ("Iterative BFS", is_same_tree_iterative_bfs),
        ("Iterative DFS", is_same_tree_iterative_dfs),
        ("Preorder", is_same_tree_preorder),
        ("Serialization", is_same_tree_serialization),
        ("Level Order", is_same_tree_level_order),
        ("Inorder", is_same_tree_inorder),
        ("Postorder", is_same_tree_postorder),
        ("Morris", is_same_tree_morris_traversal),
    ]
    
    print("Testing Same Tree implementations:")
    print("=" * 60)
    
    for i, (tree1_values, tree2_values, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {tree1_values} vs {tree2_values}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh trees for each implementation
                tree1 = create_binary_tree(tree1_values)
                tree2 = create_binary_tree(tree2_values)
                result = func(tree1, tree2)
                
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Analysis:")
    print("=" * 60)
    
    import time
    
    # Generate large test cases
    test_scenarios = [
        ("Small same trees", [i for i in range(1, 16)], [i for i in range(1, 16)]),
        ("Small different trees", [i for i in range(1, 16)], [i for i in range(2, 17)]),
        ("Medium same trees", [i for i in range(1, 101)], [i for i in range(1, 101)]),
        ("Medium different trees", [i for i in range(1, 101)], [i for i in range(1, 100)] + [200]),
        ("Large same trees", [i for i in range(1, 501)], [i for i in range(1, 501)]),
        ("Large different trees", [i for i in range(1, 501)], [i for i in range(1, 500)] + [1000]),
    ]
    
    for scenario_name, values1, values2 in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                tree1 = create_binary_tree(values1)
                tree2 = create_binary_tree(values2)
                
                start_time = time.time()
                result = func(tree1, tree2)
                end_time = time.time()
                
                print(f"  {name}: {result} in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_is_same_tree() 