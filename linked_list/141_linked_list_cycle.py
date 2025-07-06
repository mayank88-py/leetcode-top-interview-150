"""
141. Linked List Cycle

Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

Example 1:
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

Example 2:
Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.

Example 3:
Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.

Constraints:
- The number of the nodes in the list is in the range [0, 10^4]
- -10^5 <= Node.val <= 10^5
- pos is -1 or a valid index in the linked-list

Follow up: Can you solve it using O(1) (i.e. constant) memory?
"""

from typing import Optional, List, Set


class ListNode:
    """Definition for singly-linked list."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        """String representation for debugging (with cycle detection)."""
        result = []
        current = self
        seen = set()
        
        while current and current not in seen:
            result.append(str(current.val))
            seen.add(current)
            current = current.next
        
        if current:
            result.append(f"...cycle to {current.val}")
        
        return " -> ".join(result)


def has_cycle_floyd(head: Optional[ListNode]) -> bool:
    """
    Floyd's Cycle Detection Algorithm (Two Pointers - Tortoise and Hare).
    
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use two pointers: slow (moves 1 step) and fast (moves 2 steps)
    2. If there's a cycle, fast will eventually meet slow
    3. If there's no cycle, fast will reach the end
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    
    return True


def has_cycle_floyd_same_start(head: Optional[ListNode]) -> bool:
    """
    Floyd's algorithm with both pointers starting from head.
    
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Both pointers start from head
    2. Move slow by 1, fast by 2
    3. If they meet, there's a cycle
    """
    if not head:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False


def has_cycle_hash_set(head: Optional[ListNode]) -> bool:
    """
    Hash set approach to track visited nodes.
    
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(n) - hash set to store visited nodes
    
    Algorithm:
    1. Use a set to track visited nodes
    2. If we visit a node we've seen before, there's a cycle
    3. If we reach the end, there's no cycle
    """
    if not head:
        return False
    
    visited = set()
    current = head
    
    while current:
        if current in visited:
            return True
        visited.add(current)
        current = current.next
    
    return False


def has_cycle_marking(head: Optional[ListNode]) -> bool:
    """
    Node marking approach (modifies the original list).
    
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Mark visited nodes by setting their value to a special marker
    2. If we encounter a marked node, there's a cycle
    3. Restore original values after detection
    
    Note: This approach modifies the original list temporarily
    """
    if not head:
        return False
    
    MARKER = float('inf')  # Special marker value
    original_values = []
    current = head
    
    while current:
        if current.val == MARKER:
            # Restore original values
            restore_current = head
            for orig_val in original_values:
                restore_current.val = orig_val
                restore_current = restore_current.next
            return True
        
        original_values.append(current.val)
        current.val = MARKER
        current = current.next
    
    # Restore original values
    restore_current = head
    for orig_val in original_values:
        restore_current.val = orig_val
        restore_current = restore_current.next
    
    return False


def has_cycle_recursive(head: Optional[ListNode]) -> bool:
    """
    Recursive approach with memoization.
    
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(n) - recursion stack + memoization
    
    Algorithm:
    1. Use recursion with a visited set
    2. If we visit a node we've seen before, return True
    3. If we reach the end, return False
    """
    def helper(node, visited):
        if not node:
            return False
        if node in visited:
            return True
        
        visited.add(node)
        return helper(node.next, visited)
    
    return helper(head, set())


def has_cycle_step_counting(head: Optional[ListNode]) -> bool:
    """
    Step counting approach with limit.
    
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Count steps as we traverse
    2. If we exceed the maximum possible steps (n), there's a cycle
    3. This works because in a cycle, we'd never reach the end
    """
    if not head:
        return False
    
    max_steps = 10000  # Based on constraint: at most 10^4 nodes
    current = head
    steps = 0
    
    while current and steps < max_steps:
        current = current.next
        steps += 1
    
    return current is not None


def has_cycle_brent(head: Optional[ListNode]) -> bool:
    """
    Brent's Cycle Detection Algorithm (alternative to Floyd's).
    
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use two pointers with different movement patterns
    2. Periodically reset the slow pointer to fast pointer's position
    3. More efficient than Floyd's in practice for some cases
    """
    if not head:
        return False
    
    slow = head
    fast = head
    power = 1
    length = 1
    
    while fast and fast.next:
        fast = fast.next
        
        if slow == fast:
            return True
        
        if power == length:
            slow = fast
            power *= 2
            length = 0
        
        length += 1
    
    return False


def has_cycle_three_pointers(head: Optional[ListNode]) -> bool:
    """
    Three pointers approach for cycle detection.
    
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use three pointers moving at different speeds
    2. If any two pointers meet, there's a cycle
    """
    if not head or not head.next:
        return False
    
    slow = head
    medium = head.next
    fast = head.next.next if head.next else None
    
    while fast and fast.next and fast.next.next:
        if slow == medium or slow == fast or medium == fast:
            return True
        
        slow = slow.next
        medium = medium.next.next
        fast = fast.next.next.next
    
    return False


def create_linked_list_with_cycle(values: List[int], pos: int) -> Optional[ListNode]:
    """Helper function to create a linked list with a cycle at given position."""
    if not values:
        return None
    
    nodes = [ListNode(val) for val in values]
    
    # Connect nodes
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    
    # Create cycle if pos is valid
    if 0 <= pos < len(nodes):
        nodes[-1].next = nodes[pos]
    
    return nodes[0]


def create_linked_list(values: List[int]) -> Optional[ListNode]:
    """Helper function to create a linked list without cycle."""
    return create_linked_list_with_cycle(values, -1)


def test_has_cycle():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases with cycles
        ([3,2,0,-4], 1, True),
        ([1,2], 0, True),
        ([1], -1, False),
        
        # No cycles
        ([1,2,3,4,5], -1, False),
        ([], -1, False),
        ([42], -1, False),
        
        # Edge cases
        ([1,2], 1, True),  # Self-loop at end
        ([1], 0, True),    # Self-loop at single node
        
        # Longer cycles
        ([1,2,3,4,5,6], 2, True),
        ([1,2,3,4,5,6], 0, True),
        ([1,2,3,4,5,6], 5, True),
        
        # Large lists
        (list(range(100)), 50, True),
        (list(range(100)), -1, False),
        
        # Repeated values
        ([1,1,1,1], 1, True),
        ([1,1,1,1], -1, False),
        
        # Negative values
        ([-1,-2,-3,-4], 2, True),
        ([-1,-2,-3,-4], -1, False),
    ]
    
    # Test all implementations
    implementations = [
        ("Floyd's (Different Start)", has_cycle_floyd),
        ("Floyd's (Same Start)", has_cycle_floyd_same_start),
        ("Hash Set", has_cycle_hash_set),
        ("Marking", has_cycle_marking),
        ("Recursive", has_cycle_recursive),
        ("Step Counting", has_cycle_step_counting),
        ("Brent's Algorithm", has_cycle_brent),
        ("Three Pointers", has_cycle_three_pointers),
    ]
    
    print("Testing Linked List Cycle implementations:")
    print("=" * 60)
    
    for i, (values, pos, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {values}, pos={pos}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh linked list for each implementation
                head = create_linked_list_with_cycle(values, pos)
                result = func(head)
                
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
        ("Large list with cycle", list(range(1000)), 500),
        ("Large list without cycle", list(range(1000)), -1),
        ("Small cycle", list(range(10)), 5),
        ("Large cycle", list(range(1000)), 100),
    ]
    
    for scenario_name, values, pos in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                head = create_linked_list_with_cycle(values, pos)
                
                start_time = time.time()
                result = func(head)
                end_time = time.time()
                
                print(f"  {name}: {result} in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_has_cycle() 