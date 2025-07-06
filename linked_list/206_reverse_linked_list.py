"""
206. Reverse Linked List

Given the head of a singly linked list, reverse the list, and return the reversed list.

Example 1:
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Example 2:
Input: head = [1,2]
Output: [2,1]

Example 3:
Input: head = []
Output: []

Constraints:
- The number of nodes in the list is the range [0, 5000]
- -5000 <= Node.val <= 5000

Follow up: A linked list can be reversed either iteratively or recursively. Could you implement both?
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


def reverse_list_iterative(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Iterative approach using three pointers.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use three pointers: prev, current, next
    2. Iterate through the list
    3. For each node, reverse the link
    4. Move pointers forward
    """
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev


def reverse_list_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - recursion stack
    
    Algorithm:
    1. Base case: if head is None or head.next is None, return head
    2. Recursively reverse the rest of the list
    3. Reverse the current connection
    4. Return the new head
    """
    if not head or not head.next:
        return head
    
    # Recursively reverse the rest
    new_head = reverse_list_recursive(head.next)
    
    # Reverse the current connection
    head.next.next = head
    head.next = None
    
    return new_head


def reverse_list_stack(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Stack-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - stack space
    
    Algorithm:
    1. Push all nodes onto a stack
    2. Pop nodes and rebuild the list in reverse order
    """
    if not head:
        return None
    
    stack = []
    current = head
    
    # Push all nodes onto stack
    while current:
        stack.append(current)
        current = current.next
    
    # Pop nodes and rebuild
    new_head = stack.pop()
    current = new_head
    
    while stack:
        current.next = stack.pop()
        current = current.next
    
    current.next = None
    return new_head


def reverse_list_two_pointers(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Two pointers approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use two pointers: prev and current
    2. Iterate and reverse links
    """
    prev = None
    current = head
    
    while current:
        # Save next node
        temp = current.next
        # Reverse the link
        current.next = prev
        # Move pointers
        prev = current
        current = temp
    
    return prev


def reverse_list_array(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Array-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - array to store nodes
    
    Algorithm:
    1. Store all nodes in an array
    2. Reverse the array
    3. Reconnect nodes
    """
    if not head:
        return None
    
    nodes = []
    current = head
    
    # Store all nodes
    while current:
        nodes.append(current)
        current = current.next
    
    # Reverse connections
    for i in range(len(nodes) - 1, 0, -1):
        nodes[i].next = nodes[i - 1]
    
    nodes[0].next = None
    
    return nodes[-1]


def reverse_list_recursive_tail(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Tail-recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - recursion stack
    
    Algorithm:
    1. Use helper function with accumulator
    2. Build reversed list using tail recursion
    """
    def reverse_helper(current, prev):
        if not current:
            return prev
        
        next_node = current.next
        current.next = prev
        return reverse_helper(next_node, current)
    
    return reverse_helper(head, None)


def reverse_list_values(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Values-based approach (creates new list).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - new list
    
    Algorithm:
    1. Extract all values
    2. Reverse the values
    3. Create new list with reversed values
    """
    if not head:
        return None
    
    values = []
    current = head
    
    # Extract values
    while current:
        values.append(current.val)
        current = current.next
    
    # Reverse values
    values.reverse()
    
    # Create new list
    dummy = ListNode(0)
    current = dummy
    
    for val in values:
        current.next = ListNode(val)
        current = current.next
    
    return dummy.next


def reverse_list_swap_pairs(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Approach using node swapping technique.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use swapping technique similar to reversing
    2. Maintain connections properly
    """
    if not head or not head.next:
        return head
    
    # Handle the first two nodes
    new_head = head.next
    head.next = new_head.next
    new_head.next = head
    
    # Continue with the rest
    if head.next:
        head.next = reverse_list_swap_pairs(head.next)
    
    return new_head


def reverse_list_iterative_clean(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Clean iterative approach with clear variable names.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use descriptive variable names
    2. Clearly show the reversal process
    """
    reversed_head = None
    current_node = head
    
    while current_node:
        next_node = current_node.next
        current_node.next = reversed_head
        reversed_head = current_node
        current_node = next_node
    
    return reversed_head


def reverse_list_divide_conquer(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Divide and conquer approach.
    
    Time Complexity: O(n log n) where n is the number of nodes
    Space Complexity: O(log n) - recursion stack
    
    Algorithm:
    1. Divide list into two halves
    2. Recursively reverse each half
    3. Combine the results
    """
    if not head or not head.next:
        return head
    
    # Find the middle
    slow = head
    fast = head
    prev = None
    
    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next
    
    # Split the list
    prev.next = None
    
    # Recursively reverse both halves
    first_half = reverse_list_divide_conquer(head)
    second_half = reverse_list_divide_conquer(slow)
    
    # Combine: attach second_half to the end of first_half
    current = second_half
    while current.next:
        current = current.next
    current.next = first_half
    
    return second_half


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


def test_reverse_list():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,2,3,4,5], [5,4,3,2,1]),
        ([1,2], [2,1]),
        ([], []),
        ([1], [1]),
        
        # Longer lists
        ([1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]),
        ([10,20,30,40,50], [50,40,30,20,10]),
        
        # Negative numbers
        ([-1,-2,-3], [-3,-2,-1]),
        ([1,-2,3,-4,5], [5,-4,3,-2,1]),
        
        # Same values
        ([1,1,1,1], [1,1,1,1]),
        ([2,2,2], [2,2,2]),
        
        # Large range
        ([100,200,300,400,500], [500,400,300,200,100]),
        
        # Sequential
        ([1,2,3], [3,2,1]),
        ([5,4,3,2,1], [1,2,3,4,5]),
        
        # Zero values
        ([0,1,2,3,0], [0,3,2,1,0]),
        ([0,0,0], [0,0,0]),
        
        # Mixed positive/negative
        ([-5,-4,-3,-2,-1,0,1,2,3,4,5], [5,4,3,2,1,0,-1,-2,-3,-4,-5]),
        
        # Large values
        ([5000,4999,4998], [4998,4999,5000]),
        ([-5000,-4999,-4998], [-4998,-4999,-5000]),
    ]
    
    # Test all implementations
    implementations = [
        ("Iterative", reverse_list_iterative),
        ("Recursive", reverse_list_recursive),
        ("Stack", reverse_list_stack),
        ("Two Pointers", reverse_list_two_pointers),
        ("Array", reverse_list_array),
        ("Recursive Tail", reverse_list_recursive_tail),
        ("Values", reverse_list_values),
        ("Swap Pairs", reverse_list_swap_pairs),
        ("Iterative Clean", reverse_list_iterative_clean),
        ("Divide Conquer", reverse_list_divide_conquer),
    ]
    
    print("Testing Reverse Linked List implementations:")
    print("=" * 60)
    
    for i, (values, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {values}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh linked list for each implementation
                head = create_linked_list(values)
                result_head = func(head)
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
        ("Small list", list(range(1, 21))),
        ("Medium list", list(range(1, 101))),
        ("Large list", list(range(1, 1001))),
        ("Reverse order", list(range(1000, 0, -1))),
        ("Same values", [42] * 500),
    ]
    
    for scenario_name, values in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                head = create_linked_list(values)
                
                start_time = time.time()
                result_head = func(head)
                end_time = time.time()
                
                result_length = len(linked_list_to_list(result_head))
                print(f"  {name}: {result_length} nodes in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_reverse_list() 