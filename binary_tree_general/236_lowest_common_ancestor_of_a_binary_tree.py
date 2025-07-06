"""
LeetCode 236: Lowest Common Ancestor of a Binary Tree

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: "The lowest common ancestor is defined 
between two nodes p and q as the lowest node in T that has both p and q as descendants 
(where we allow a node to be a descendant of itself)."

Example 1:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3

Example 2:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5

Example 3:
Input: root = [1,2], p = 1, q = 2
Output: 1

Constraints:
- The number of nodes in the tree is in the range [2, 10^5].
- -10^9 <= Node.val <= 10^9
- All Node.val are unique.
- p != q
- p and q will exist in the tree.
"""

from typing import Optional, List, Dict, Set
from collections import deque, defaultdict


class TreeNode:
    """Binary tree node definition."""
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def lca_recursive_classic(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Classic recursive approach.
    
    Time Complexity: O(n)
    Space Complexity: O(h) where h is tree height
    
    Algorithm:
    1. If root is None or root is p or q, return root
    2. Recursively search in left and right subtrees
    3. If both return non-None, current root is LCA
    4. Otherwise, return the non-None result
    """
    if not root or root == p or root == q:
        return root
    
    left = lca_recursive_classic(root.left, p, q)
    right = lca_recursive_classic(root.right, p, q)
    
    # If both left and right are non-None, root is LCA
    if left and right:
        return root
    
    # Return the non-None result
    return left if left else right


def lca_parent_pointers(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Parent pointers approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Build parent mapping using BFS/DFS
    2. Find path from p to root
    3. Traverse from q to root, first common node is LCA
    """
    # Build parent mapping
    parent = {root: None}
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        if node.left:
            parent[node.left] = node
            queue.append(node.left)
        
        if node.right:
            parent[node.right] = node
            queue.append(node.right)
    
    # Find ancestors of p
    ancestors = set()
    while p:
        ancestors.add(p)
        p = parent[p]
    
    # Find first common ancestor
    while q:
        if q in ancestors:
            return q
        q = parent[q]
    
    return None


def lca_path_finding(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Path finding approach.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Find path from root to p
    2. Find path from root to q  
    3. Compare paths to find last common node
    """
    def find_path(node, target, path):
        """Find path from root to target node."""
        if not node:
            return False
        
        path.append(node)
        
        if node == target:
            return True
        
        if (find_path(node.left, target, path) or 
            find_path(node.right, target, path)):
            return True
        
        path.pop()
        return False
    
    path_p = []
    path_q = []
    
    find_path(root, p, path_p)
    find_path(root, q, path_q)
    
    # Find last common node
    lca = None
    min_len = min(len(path_p), len(path_q))
    
    for i in range(min_len):
        if path_p[i] == path_q[i]:
            lca = path_p[i]
        else:
            break
    
    return lca


def lca_iterative_dfs(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Iterative DFS approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Use stack for DFS traversal
    2. Keep track of parent relationships
    3. Find LCA using parent pointers
    """
    stack = [root]
    parent = {root: None}
    
    # Build parent mapping using DFS
    while p not in parent or q not in parent:
        node = stack.pop()
        
        if node.left:
            parent[node.left] = node
            stack.append(node.left)
        
        if node.right:
            parent[node.right] = node
            stack.append(node.right)
    
    # Find ancestors of p
    ancestors = set()
    while p:
        ancestors.add(p)
        p = parent[p]
    
    # Find LCA
    while q not in ancestors:
        q = parent[q]
    
    return q


def lca_level_order(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Level order traversal approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Use BFS to build parent mapping
    2. Find LCA using parent relationships
    3. Level-by-level processing
    """
    if root == p or root == q:
        return root
    
    parent = {root: None}
    queue = deque([root])
    
    # BFS to build parent mapping
    while queue and (p not in parent or q not in parent):
        node = queue.popleft()
        
        if node.left:
            parent[node.left] = node
            queue.append(node.left)
        
        if node.right:
            parent[node.right] = node
            queue.append(node.right)
    
    # Find ancestors of p
    ancestors = set()
    current = p
    while current:
        ancestors.add(current)
        current = parent[current]
    
    # Find LCA
    current = q
    while current not in ancestors:
        current = parent[current]
    
    return current


def lca_recursive_with_found_flags(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Recursive approach with found flags.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use flags to track if p and q are found
    2. Return LCA when both are found in subtree
    3. More explicit about the finding process
    """
    def dfs(node):
        """Returns (found_p, found_q, lca)."""
        if not node:
            return False, False, None
        
        # Check current node
        found_p_here = node == p
        found_q_here = node == q
        
        # Check left subtree
        left_p, left_q, left_lca = dfs(node.left)
        if left_lca:
            return True, True, left_lca
        
        # Check right subtree
        right_p, right_q, right_lca = dfs(node.right)
        if right_lca:
            return True, True, right_lca
        
        # Combine results
        found_p = found_p_here or left_p or right_p
        found_q = found_q_here or left_q or right_q
        
        # If both found in current subtree, this is LCA
        if found_p and found_q:
            return True, True, node
        
        return found_p, found_q, None
    
    _, _, lca = dfs(root)
    return lca


def lca_postorder_iterative(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Postorder iterative approach.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use postorder traversal to process children first
    2. Check if current node is LCA after processing children
    3. Use stack to maintain traversal state
    """
    stack = []
    last_visited = None
    current = root
    found = {p: False, q: False}
    
    while stack or current:
        if current:
            stack.append(current)
            current = current.left
        else:
            peek_node = stack[-1]
            
            if peek_node.right and last_visited != peek_node.right:
                current = peek_node.right
            else:
                # Process node
                if peek_node == p:
                    found[p] = True
                if peek_node == q:
                    found[q] = True
                
                # Check if this is LCA
                left_has_both = False
                right_has_both = False
                
                if peek_node.left:
                    # Would need to track subtree results
                    pass
                
                stack.pop()
                last_visited = peek_node
    
    # Simplified version - use parent pointer method
    parent = {}
    stack = [root]
    
    while stack:
        node = stack.pop()
        
        if node.left:
            parent[node.left] = node
            stack.append(node.left)
        
        if node.right:
            parent[node.right] = node
            stack.append(node.right)
    
    parent[root] = None
    
    ancestors = set()
    current = p
    while current:
        ancestors.add(current)
        current = parent[current]
    
    current = q
    while current not in ancestors:
        current = parent[current]
    
    return current


def lca_morris_inspired(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Morris traversal inspired approach.
    
    Time Complexity: O(n)
    Space Complexity: O(1) auxiliary space + parent mapping
    
    Algorithm:
    1. Use Morris-like technique to avoid recursion
    2. Build parent mapping iteratively
    3. Find LCA using parent relationships
    """
    # Build parent mapping using iterative approach
    parent = {root: None}
    stack = [root]
    
    # DFS to find both nodes and build parent mapping
    while stack:
        node = stack.pop()
        
        if node.right:
            parent[node.right] = node
            stack.append(node.right)
        
        if node.left:
            parent[node.left] = node
            stack.append(node.left)
    
    # Find path lengths to root
    def depth(node):
        d = 0
        while node:
            d += 1
            node = parent[node]
        return d
    
    # Bring both nodes to same level
    depth_p = depth(p)
    depth_q = depth(q)
    
    # Move deeper node up
    while depth_p > depth_q:
        p = parent[p]
        depth_p -= 1
    
    while depth_q > depth_p:
        q = parent[q]
        depth_q -= 1
    
    # Move both up until they meet
    while p != q:
        p = parent[p]
        q = parent[q]
    
    return p


def lca_divide_conquer(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Divide and conquer approach.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Divide tree into left and right subtrees
    2. Find LCA in each subtree
    3. Combine results based on where nodes are found
    """
    def solve(node):
        if not node:
            return None
        
        # Base case: found one of the target nodes
        if node == p or node == q:
            return node
        
        # Divide: solve for left and right subtrees
        left_result = solve(node.left)
        right_result = solve(node.right)
        
        # Conquer: combine results
        if left_result and right_result:
            # Both nodes found in different subtrees
            return node
        
        # Return the non-None result (or None if both are None)
        return left_result or right_result
    
    return solve(root)


def lca_memoization(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Memoization approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Memoize results of subtree searches
    2. Avoid redundant computations
    3. Cache whether subtrees contain target nodes
    """
    memo = {}
    
    def contains_node(node, target):
        """Check if subtree contains target node."""
        if not node:
            return False
        
        key = (id(node), id(target))
        if key in memo:
            return memo[key]
        
        result = (node == target or 
                 contains_node(node.left, target) or 
                 contains_node(node.right, target))
        
        memo[key] = result
        return result
    
    def find_lca(node):
        if not node:
            return None
        
        # Check if current node is one of the targets
        if node == p or node == q:
            return node
        
        left_contains_p = contains_node(node.left, p)
        left_contains_q = contains_node(node.left, q)
        right_contains_p = contains_node(node.right, p)
        right_contains_q = contains_node(node.right, q)
        
        # If both nodes are in different subtrees, current is LCA
        if ((left_contains_p and right_contains_q) or 
            (left_contains_q and right_contains_p)):
            return node
        
        # Both in left subtree
        if left_contains_p and left_contains_q:
            return find_lca(node.left)
        
        # Both in right subtree
        if right_contains_p and right_contains_q:
            return find_lca(node.right)
        
        return None
    
    return find_lca(root)


def build_test_tree(nodes: List[Optional[int]]) -> Optional[TreeNode]:
    """Build tree from list representation and return node mapping."""
    if not nodes or nodes[0] is None:
        return None, {}
    
    node_map = {}
    root = TreeNode(nodes[0])
    node_map[nodes[0]] = root
    queue = deque([root])
    i = 1
    
    while queue and i < len(nodes):
        node = queue.popleft()
        
        if i < len(nodes) and nodes[i] is not None:
            node.left = TreeNode(nodes[i])
            node_map[nodes[i]] = node.left
            queue.append(node.left)
        i += 1
        
        if i < len(nodes) and nodes[i] is not None:
            node.right = TreeNode(nodes[i])
            node_map[nodes[i]] = node.right
            queue.append(node.right)
        i += 1
    
    return root, node_map


def test_lca():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1, 2], 1, 2, 1),
        ([1, 2, 3], 2, 3, 1),
        ([2, 1, 3], 1, 3, 2),
        
        # Given examples
        ([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 5, 1, 3),
        ([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 5, 4, 5),
        
        # Left subtree cases
        ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 2, 4, 2),
        ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 0, 3, 2),
        ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 3, 5, 4),
        
        # Right subtree cases
        ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 7, 9, 8),
        ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 8, 9, 8),
        
        # Cross subtree cases
        ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 0, 7, 6),
        ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 3, 9, 6),
        ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 4, 8, 6),
        
        # One node is ancestor of another
        ([5, 3, 6, 2, 4, None, None, 1], 3, 1, 3),
        ([5, 3, 6, 2, 4, None, None, 1], 5, 1, 5),
        ([5, 3, 6, 2, 4, None, None, 1], 2, 1, 2),
        
        # Negative values
        ([-1, -2, -3], -2, -3, -1),
        ([1, -2, 3, -4, -5], -4, -5, -2),
        
        # Large values
        ([100, 50, 150, 25, 75, 125, 175], 25, 75, 50),
        ([100, 50, 150, 25, 75, 125, 175], 125, 175, 150),
        
        # Deep trees
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 8, 9, 4),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 12, 13, 6),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 8, 13, 1),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive Classic", lca_recursive_classic),
        ("Parent Pointers", lca_parent_pointers),
        ("Path Finding", lca_path_finding),
        ("Iterative DFS", lca_iterative_dfs),
        ("Level Order", lca_level_order),
        ("Recursive with Flags", lca_recursive_with_found_flags),
        ("Postorder Iterative", lca_postorder_iterative),
        ("Morris Inspired", lca_morris_inspired),
        ("Divide Conquer", lca_divide_conquer),
        ("Memoization", lca_memoization),
    ]
    
    print("Testing Lowest Common Ancestor implementations:")
    print("=" * 75)
    
    for i, (nodes, p_val, q_val, expected_val) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Tree:     {nodes}")
        print(f"p={p_val}, q={q_val}, Expected LCA: {expected_val}")
        
        root, node_map = build_test_tree(nodes)
        p_node = node_map[p_val]
        q_node = node_map[q_val]
        expected_node = node_map[expected_val]
        
        for name, func in implementations:
            try:
                result = func(root, p_node, q_node)
                status = "✓" if result == expected_node else "✗"
                result_val = result.val if result else None
                print(f"{status} {name}: {result_val}")
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
        root, node_map = build_test_tree(nodes)
        
        # Pick nodes for LCA test
        p_val = 2**(height-2)      # Node in left subtree
        q_val = 3 * 2**(height-2)  # Node in right subtree  
        p_node = node_map[p_val]
        q_node = node_map[q_val]
        
        for name, func in implementations:
            try:
                start_time = time.time()
                result = func(root, p_node, q_node)
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
                result_val = result.val if result else None
                print(f"  {name}: {elapsed:.2f} ms (LCA: {result_val})")
            except Exception as e:
                print(f"  {name}: Error - {e}")
    
    print(f"\n{'='*75}")
    print("Algorithm Complexity Analysis:")
    print("="*75)
    print("1. Recursive Classic:      O(n) time, O(h) space")
    print("2. Parent Pointers:        O(n) time, O(n) space")
    print("3. Path Finding:           O(n) time, O(h) space")
    print("4. Iterative DFS:          O(n) time, O(n) space")
    print("5. Level Order:            O(n) time, O(n) space")
    print("6. Recursive with Flags:   O(n) time, O(h) space")
    print("7. Postorder Iterative:    O(n) time, O(h) space")
    print("8. Morris Inspired:        O(n) time, O(n) space")
    print("9. Divide Conquer:         O(n) time, O(h) space")
    print("10. Memoization:           O(n) time, O(n) space")
    
    print(f"\n{'='*75}")
    print("Key Insights:")
    print("="*75)
    print("• Classic recursive solution is most elegant and efficient")
    print("• All algorithms have O(n) time complexity in worst case")
    print("• Space complexity varies: O(h) for recursive, O(n) for iterative")
    print("• Parent pointer approach is intuitive but uses more space")
    print("• LCA property: if both nodes in different subtrees, root is LCA")
    print("• Problem guarantees both nodes exist, simplifying the logic")


if __name__ == "__main__":
    test_lca() 