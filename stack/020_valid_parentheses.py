"""
20. Valid Parentheses

Problem:
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']',
determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "()[]{}"
Output: true

Example 3:
Input: s = "(]"
Output: false

Example 4:
Input: s = "([)]"
Output: false

Example 5:
Input: s = "{[]}"
Output: true

Time Complexity: O(n)
Space Complexity: O(n)
"""


def is_valid(s):
    """
    Check if parentheses are valid using stack.
    
    Args:
        s: String containing parentheses
    
    Returns:
        True if valid, False otherwise
    """
    if not s:
        return True
    
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # Opening bracket
            stack.append(char)
    
    # Stack should be empty if all brackets are matched
    return len(stack) == 0


def is_valid_alternative(s):
    """
    Alternative implementation using different approach.
    
    Args:
        s: String containing parentheses
    
    Returns:
        True if valid, False otherwise
    """
    if not s:
        return True
    
    stack = []
    pairs = {'(': ')', '{': '}', '[': ']'}
    
    for char in s:
        if char in pairs:
            # Opening bracket
            stack.append(char)
        else:
            # Closing bracket
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0


def is_valid_counter(s):
    """
    Implementation using counter approach for simple cases.
    Note: This only works for cases with single type of brackets.
    
    Args:
        s: String containing parentheses
    
    Returns:
        True if valid, False otherwise
    """
    if not s:
        return True
    
    # This approach only works for single type of brackets
    # For demonstration purposes with simple parentheses
    if all(c in '()' for c in s):
        counter = 0
        for char in s:
            if char == '(':
                counter += 1
            elif char == ')':
                counter -= 1
                if counter < 0:
                    return False
        return counter == 0
    
    # Fall back to stack approach for mixed brackets
    return is_valid(s)


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1 = "()"
    result1a = is_valid(s1)
    result1b = is_valid_alternative(s1)
    result1c = is_valid_counter(s1)
    print(f"Test 1 - Expected: True, Stack: {result1a}, Alternative: {result1b}, Counter: {result1c}")
    
    # Test case 2
    s2 = "()[]{}"
    result2a = is_valid(s2)
    result2b = is_valid_alternative(s2)
    result2c = is_valid_counter(s2)
    print(f"Test 2 - Expected: True, Stack: {result2a}, Alternative: {result2b}, Counter: {result2c}")
    
    # Test case 3
    s3 = "(]"
    result3a = is_valid(s3)
    result3b = is_valid_alternative(s3)
    result3c = is_valid_counter(s3)
    print(f"Test 3 - Expected: False, Stack: {result3a}, Alternative: {result3b}, Counter: {result3c}")
    
    # Test case 4
    s4 = "([)]"
    result4a = is_valid(s4)
    result4b = is_valid_alternative(s4)
    result4c = is_valid_counter(s4)
    print(f"Test 4 - Expected: False, Stack: {result4a}, Alternative: {result4b}, Counter: {result4c}")
    
    # Test case 5
    s5 = "{[]}"
    result5a = is_valid(s5)
    result5b = is_valid_alternative(s5)
    result5c = is_valid_counter(s5)
    print(f"Test 5 - Expected: True, Stack: {result5a}, Alternative: {result5b}, Counter: {result5c}")
    
    # Test case 6
    s6 = ""
    result6a = is_valid(s6)
    result6b = is_valid_alternative(s6)
    result6c = is_valid_counter(s6)
    print(f"Test 6 - Expected: True, Stack: {result6a}, Alternative: {result6b}, Counter: {result6c}")
    
    # Test case 7
    s7 = "((("
    result7a = is_valid(s7)
    result7b = is_valid_alternative(s7)
    result7c = is_valid_counter(s7)
    print(f"Test 7 - Expected: False, Stack: {result7a}, Alternative: {result7b}, Counter: {result7c}")
    
    # Test case 8
    s8 = ")))"
    result8a = is_valid(s8)
    result8b = is_valid_alternative(s8)
    result8c = is_valid_counter(s8)
    print(f"Test 8 - Expected: False, Stack: {result8a}, Alternative: {result8b}, Counter: {result8c}")
    
    # Test case 9
    s9 = "(())"
    result9a = is_valid(s9)
    result9b = is_valid_alternative(s9)
    result9c = is_valid_counter(s9)
    print(f"Test 9 - Expected: True, Stack: {result9a}, Alternative: {result9b}, Counter: {result9c}") 