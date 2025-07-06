"""
61. Rotate List

Given the head of a linked list, rotate the list to the right by k places.

Example 1:
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]

Example 2:
Input: head = [0,1,2], k = 4
Output: [2,0,1]

Example 3:
Input: head = [1], k = 1
Output: [1]

Constraints:
- The number of nodes in the list is in the range [0, 500]
- -100 <= Node.val <= 100
- 0 <= k <= 2 * 10^9
"""

from typing import Optional, List


class ListNode:
    """Definition for singly-linked list."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        """String representation for debugging."""
        result = []
        current = self
        while current:
            result.append(str(current.val))
            current = current.next
        return " -> ".join(result)


def rotate_right_two_pass(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Two-pass approach: find length first, then rotate.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. First pass: find the length of the list
    2. Calculate effective rotation: k % length
    3. Second pass: find the new tail and head
    4. Perform the rotation
    """
    if not head or not head.next or k == 0:
        return head
    
    # First pass: find length
    length = 1
    tail = head
    while tail.next:
        tail = tail.next
        length += 1
    
    # Calculate effective rotation
    k = k % length
    if k == 0:
        return head
    
    # Find new tail (length - k - 1 steps from head)
    new_tail = head
    for _ in range(length - k - 1):
        new_tail = new_tail.next
    
    # New head is next to new tail
    new_head = new_tail.next
    
    # Break the connection and connect tail to original head
    new_tail.next = None
    tail.next = head
    
    return new_head


def rotate_right_circular(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Circular list approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Make the list circular by connecting tail to head
    2. Find the new tail position
    3. Break the circle at the appropriate position
    """
    if not head or not head.next or k == 0:
        return head
    
    # Find length and make circular
    length = 1
    current = head
    while current.next:
        current = current.next
        length += 1
    
    # Make circular
    current.next = head
    
    # Calculate effective rotation
    k = k % length
    
    # Find new tail (length - k steps from head)
    new_tail = head
    for _ in range(length - k - 1):
        new_tail = new_tail.next
    
    new_head = new_tail.next
    new_tail.next = None
    
    return new_head


def rotate_right_array(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Array-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - array to store nodes
    
    Algorithm:
    1. Store all nodes in an array
    2. Calculate new positions after rotation
    3. Reconnect nodes in new order
    """
    if not head or not head.next or k == 0:
        return head
    
    # Store all nodes
    nodes = []
    current = head
    while current:
        nodes.append(current)
        current = current.next
    
    length = len(nodes)
    k = k % length
    
    if k == 0:
        return head
    
    # Rotate array
    rotated_nodes = nodes[-k:] + nodes[:-k]
    
    # Reconnect nodes
    for i in range(length - 1):
        rotated_nodes[i].next = rotated_nodes[i + 1]
    
    rotated_nodes[-1].next = None
    
    return rotated_nodes[0]


def rotate_right_values(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Values-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - array to store values
    
    Algorithm:
    1. Extract all values into an array
    2. Rotate the values array
    3. Create new linked list with rotated values
    """
    if not head or not head.next or k == 0:
        return head
    
    # Extract values
    values = []
    current = head
    while current:
        values.append(current.val)
        current = current.next
    
    length = len(values)
    k = k % length
    
    if k == 0:
        return head
    
    # Rotate values
    rotated_values = values[-k:] + values[:-k]
    
    # Create new list
    new_head = ListNode(rotated_values[0])
    current = new_head
    
    for val in rotated_values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return new_head


def rotate_right_recursive(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - recursion stack
    
    Algorithm:
    1. Use recursion to find length and tail
    2. Calculate rotation and perform it recursively
    """
    def get_length_and_tail(node):
        if not node.next:
            return 1, node
        length, tail = get_length_and_tail(node.next)
        return length + 1, tail
    
    def rotate_helper(node, steps_to_tail):
        if steps_to_tail == 0:
            new_head = node.next
            node.next = None
            return new_head, node
        
        new_head, old_tail = rotate_helper(node.next, steps_to_tail - 1)
        return new_head, old_tail
    
    if not head or not head.next or k == 0:
        return head
    
    length, tail = get_length_and_tail(head)
    k = k % length
    
    if k == 0:
        return head
    
    steps_to_new_tail = length - k - 1
    new_head, new_tail = rotate_helper(head, steps_to_new_tail)
    tail.next = head
    
    return new_head


def rotate_right_stack(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Stack-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - stack to store nodes
    
    Algorithm:
    1. Push all nodes to stack
    2. Pop k nodes to get the rotated part
    3. Reconstruct the list
    """
    if not head or not head.next or k == 0:
        return head
    
    # Push all nodes to stack
    stack = []
    current = head
    while current:
        stack.append(current)
        current = current.next
    
    length = len(stack)
    k = k % length
    
    if k == 0:
        return head
    
    # Get nodes in new order
    nodes = []
    
    # Add last k nodes first
    for _ in range(k):
        nodes.append(stack.pop())
    
    # Add remaining nodes
    while stack:
        nodes.append(stack.pop())
    
    # Reverse to get correct order
    nodes.reverse()
    
    # Reconnect
    for i in range(length - 1):
        nodes[i].next = nodes[i + 1]
    
    nodes[-1].next = None
    
    return nodes[0]


def rotate_right_deque(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Deque-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - deque to store nodes
    
    Algorithm:
    1. Use deque for efficient rotation
    2. Rotate deque by k positions
    3. Reconstruct list from rotated deque
    """
    if not head or not head.next or k == 0:
        return head
    
    from collections import deque
    
    # Add all nodes to deque
    dq = deque()
    current = head
    while current:
        dq.append(current)
        current = current.next
    
    length = len(dq)
    k = k % length
    
    if k == 0:
        return head
    
    # Rotate deque
    dq.rotate(k)
    
    # Reconnect nodes
    nodes = list(dq)
    for i in range(length - 1):
        nodes[i].next = nodes[i + 1]
    
    nodes[-1].next = None
    
    return nodes[0]


def rotate_right_find_kth(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Find kth from end approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use two pointers to find kth node from end
    2. That becomes the new head
    3. Reconnect appropriately
    """
    if not head or not head.next or k == 0:
        return head
    
    # Find length first
    length = 0
    current = head
    while current:
        length += 1
        current = current.next
    
    k = k % length
    if k == 0:
        return head
    
    # Find kth node from end using two pointers
    fast = head
    slow = head
    
    # Move fast k steps ahead
    for _ in range(k):
        fast = fast.next
    
    # Move both until fast reaches end
    while fast.next:
        fast = fast.next
        slow = slow.next
    
    # slow.next is the new head
    new_head = slow.next
    slow.next = None
    fast.next = head
    
    return new_head


def rotate_right_split_merge(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Split and merge approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Split the list at the rotation point
    2. Merge the two parts in reverse order
    """
    if not head or not head.next or k == 0:
        return head
    
    # Find length and tail
    length = 1
    tail = head
    while tail.next:
        tail = tail.next
        length += 1
    
    k = k % length
    if k == 0:
        return head
    
    # Find split point
    split_point = head
    for _ in range(length - k - 1):
        split_point = split_point.next
    
    # Split
    second_part = split_point.next
    split_point.next = None
    
    # Merge: second_part + first_part
    tail.next = head
    
    return second_part


def rotate_right_multiple_rotations(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Multiple single rotations approach (inefficient but educational).
    
    Time Complexity: O(n * k) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Perform k single rotations
    2. Each rotation moves last node to front
    """
    if not head or not head.next or k == 0:
        return head
    
    # Find length first to optimize
    length = 0
    current = head
    while current:
        length += 1
        current = current.next
    
    k = k % length
    
    for _ in range(k):
        # Find second last node
        if not head.next:
            break
            
        second_last = head
        while second_last.next.next:
            second_last = second_last.next
        
        # Move last node to front
        last = second_last.next
        second_last.next = None
        last.next = head
        head = last
    
    return head


def create_linked_list(values: List[int]) -> Optional[ListNode]:
    """Helper function to create a linked list from a list of values."""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head


def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    """Helper function to convert linked list to list for testing."""
    result = []
    current = head
    
    while current:
        result.append(current.val)
        current = current.next
    
    return result


def test_rotate_right():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,2,3,4,5], 2, [4,5,1,2,3]),
        ([0,1,2], 4, [2,0,1]),
        ([1], 1, [1]),
        ([], 1, []),
        
        # No rotation needed
        ([1,2,3], 0, [1,2,3]),
        ([1,2,3], 3, [1,2,3]),
        ([1,2,3], 6, [1,2,3]),
        
        # Single rotation
        ([1,2,3,4], 1, [4,1,2,3]),
        ([1,2], 1, [2,1]),
        
        # Full rotation
        ([1,2,3], 3, [1,2,3]),
        ([1,2,3,4], 4, [1,2,3,4]),
        
        # Large k values
        ([1,2], 99, [2,1]),
        ([1,2,3], 100, [2,3,1]),
        
        # Two elements
        ([1,2], 2, [1,2]),
        ([1,2], 3, [2,1]),
        
        # Negative values
        ([-1,-2,-3], 1, [-3,-1,-2]),
        ([1,-2,3], 2, [-2,3,1]),
        
        # Same values
        ([1,1,1], 1, [1,1,1]),
        ([2,2,2,2], 2, [2,2,2,2]),
        
        # Large lists
        ([1,2,3,4,5,6,7,8,9,10], 3, [8,9,10,1,2,3,4,5,6,7]),
        ([1,2,3,4,5,6], 6, [1,2,3,4,5,6]),
        
        # Edge rotations
        ([1,2,3,4,5], 5, [1,2,3,4,5]),
        ([1,2,3,4,5], 4, [2,3,4,5,1]),
        ([1,2,3,4,5], 1, [5,1,2,3,4]),
        
        # Zero values
        ([0,1,2,3], 2, [2,3,0,1]),
        ([0,0,1], 1, [1,0,0]),
        
        # Large values
        ([100,200,300], 1, [300,100,200]),
        ([1000,2000], 3, [2000,1000]),
        
        # Mixed patterns
        ([1,3,2,4], 2, [2,4,1,3]),
        ([5,4,3,2,1], 3, [3,2,1,5,4]),
    ]
    
    # Test all implementations
    implementations = [
        ("Two Pass", rotate_right_two_pass),
        ("Circular", rotate_right_circular),
        ("Array", rotate_right_array),
        ("Values", rotate_right_values),
        ("Recursive", rotate_right_recursive),
        ("Stack", rotate_right_stack),
        ("Deque", rotate_right_deque),
        ("Find Kth", rotate_right_find_kth),
        ("Split Merge", rotate_right_split_merge),
        ("Multiple Rotations", rotate_right_multiple_rotations),
    ]
    
    print("Testing Rotate List implementations:")
    print("=" * 60)
    
    for i, (values, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {values}, k={k}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh linked list for each implementation
                head = create_linked_list(values)
                result_head = func(head, k)
                result = linked_list_to_list(result_head)
                
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
        ("Small list", list(range(1, 11)), 3),
        ("Medium list", list(range(1, 51)), 20),
        ("Large list", list(range(1, 101)), 50),
        ("Large k", list(range(1, 21)), 1000),
        ("No rotation", list(range(1, 51)), 50),
    ]
    
    for scenario_name, values, k in test_scenarios:
        print(f"\n{scenario_name} (k={k}):")
        
        for name, func in implementations:
            try:
                head = create_linked_list(values)
                
                start_time = time.time()
                result_head = func(head, k)
                end_time = time.time()
                
                result_length = len(linked_list_to_list(result_head))
                print(f"  {name}: {result_length} nodes in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_rotate_right() 