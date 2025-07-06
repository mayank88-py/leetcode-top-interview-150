"""
224. Basic Calculator

Problem:
Given a string s representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation.

Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

Example 1:
Input: s = "1 + 1"
Output: 2

Example 2:
Input: s = " 2-1 + 2 "
Output: 3

Example 3:
Input: s = "(1+(4+5+2)-3)+(6+8)"
Output: 23

Constraints:
- s consists of digits, '+', '-', '(', ')', and ' '.
- s represents a valid expression.
- '+' is not used as a unary operation.
- '-' could be used as a unary operation (i.e., "-1" and "-(2 + 3)" is valid).
- There will be no two consecutive operators in the input.
- Every number and running calculation will fit in a signed 32-bit integer.

Time Complexity: O(n)
Space Complexity: O(n)
"""


def calculate(s):
    """
    Calculate the result of a basic arithmetic expression using stack.
    
    Args:
        s: String representing the expression
    
    Returns:
        Integer result of the expression
    """
    if not s:
        return 0
    
    stack = []
    result = 0
    current_number = 0
    sign = 1  # 1 for positive, -1 for negative
    
    for char in s:
        if char.isdigit():
            current_number = current_number * 10 + int(char)
        elif char == '+':
            result += sign * current_number
            current_number = 0
            sign = 1
        elif char == '-':
            result += sign * current_number
            current_number = 0
            sign = -1
        elif char == '(':
            # Push current result and sign onto stack
            stack.append(result)
            stack.append(sign)
            # Reset for the expression inside parentheses
            result = 0
            sign = 1
        elif char == ')':
            # Complete the current number
            result += sign * current_number
            current_number = 0
            
            # Pop the sign and previous result
            result *= stack.pop()  # This is the sign before '('
            result += stack.pop()  # This is the result before '('
        # Skip spaces
    
    # Add the last number
    result += sign * current_number
    return result


def calculate_recursive(s):
    """
    Calculate using recursive descent parsing.
    
    Args:
        s: String representing the expression
    
    Returns:
        Integer result of the expression
    """
    def helper(index):
        """
        Parse and evaluate expression starting from index.
        Returns (result, next_index)
        """
        result = 0
        current_number = 0
        sign = 1
        
        while index < len(s):
            char = s[index]
            
            if char.isdigit():
                current_number = current_number * 10 + int(char)
            elif char == '+':
                result += sign * current_number
                current_number = 0
                sign = 1
            elif char == '-':
                result += sign * current_number
                current_number = 0
                sign = -1
            elif char == '(':
                # Recursively evaluate the parentheses
                sub_result, index = helper(index + 1)
                current_number = sub_result
            elif char == ')':
                # End of current parentheses
                result += sign * current_number
                return result, index
            
            index += 1
        
        # Add the last number
        result += sign * current_number
        return result, index
    
    result, _ = helper(0)
    return result


def calculate_no_stack(s):
    """
    Calculate without using stack, handling parentheses with recursion.
    
    Args:
        s: String representing the expression
    
    Returns:
        Integer result of the expression
    """
    def evaluate(expression):
        """Evaluate expression without parentheses"""
        result = 0
        current_number = 0
        sign = 1
        
        for char in expression:
            if char.isdigit():
                current_number = current_number * 10 + int(char)
            elif char in ['+', '-']:
                result += sign * current_number
                current_number = 0
                sign = 1 if char == '+' else -1
            # Skip spaces
        
        result += sign * current_number
        return result
    
    # Remove spaces for easier processing
    s = s.replace(' ', '')
    
    # Handle parentheses recursively
    while '(' in s:
        # Find the innermost parentheses
        start = -1
        for i in range(len(s)):
            if s[i] == '(':
                start = i
            elif s[i] == ')':
                # Evaluate the expression inside parentheses
                inner_expr = s[start + 1:i]
                inner_result = evaluate(inner_expr)
                
                # Replace the parentheses and their content with the result
                s = s[:start] + str(inner_result) + s[i + 1:]
                break
    
    return evaluate(s)


def calculate_state_machine(s):
    """
    Calculate using state machine approach.
    
    Args:
        s: String representing the expression
    
    Returns:
        Integer result of the expression
    """
    stack = []
    result = 0
    current_number = 0
    sign = 1
    
    for char in s:
        if char.isdigit():
            current_number = current_number * 10 + int(char)
        elif char in ['+', '-']:
            result += sign * current_number
            current_number = 0
            sign = 1 if char == '+' else -1
        elif char == '(':
            # Save current state
            stack.append(result)
            stack.append(sign)
            # Reset for new context
            result = 0
            sign = 1
        elif char == ')':
            # Finish current number
            result += sign * current_number
            current_number = 0
            
            # Restore previous state
            result *= stack.pop()  # Previous sign
            result += stack.pop()  # Previous result
    
    # Handle last number
    result += sign * current_number
    return result


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1 = "1 + 1"
    result1a = calculate(s1)
    result1b = calculate_recursive(s1)
    result1c = calculate_no_stack(s1)
    result1d = calculate_state_machine(s1)
    print(f"Test 1 - Expected: 2, Stack: {result1a}, Recursive: {result1b}, No Stack: {result1c}, State Machine: {result1d}")
    
    # Test case 2
    s2 = " 2-1 + 2 "
    result2a = calculate(s2)
    result2b = calculate_recursive(s2)
    result2c = calculate_no_stack(s2)
    result2d = calculate_state_machine(s2)
    print(f"Test 2 - Expected: 3, Stack: {result2a}, Recursive: {result2b}, No Stack: {result2c}, State Machine: {result2d}")
    
    # Test case 3
    s3 = "(1+(4+5+2)-3)+(6+8)"
    result3a = calculate(s3)
    result3b = calculate_recursive(s3)
    result3c = calculate_no_stack(s3)
    result3d = calculate_state_machine(s3)
    print(f"Test 3 - Expected: 23, Stack: {result3a}, Recursive: {result3b}, No Stack: {result3c}, State Machine: {result3d}")
    
    # Test case 4 - Single number
    s4 = "42"
    result4a = calculate(s4)
    result4b = calculate_recursive(s4)
    result4c = calculate_no_stack(s4)
    result4d = calculate_state_machine(s4)
    print(f"Test 4 - Expected: 42, Stack: {result4a}, Recursive: {result4b}, No Stack: {result4c}, State Machine: {result4d}")
    
    # Test case 5 - Negative number
    s5 = "-2+ 1"
    result5a = calculate(s5)
    result5b = calculate_recursive(s5)
    result5c = calculate_no_stack(s5)
    result5d = calculate_state_machine(s5)
    print(f"Test 5 - Expected: -1, Stack: {result5a}, Recursive: {result5b}, No Stack: {result5c}, State Machine: {result5d}")
    
    # Test case 6 - Nested parentheses
    s6 = "2-(1-6)"
    result6a = calculate(s6)
    result6b = calculate_recursive(s6)
    result6c = calculate_no_stack(s6)
    result6d = calculate_state_machine(s6)
    print(f"Test 6 - Expected: 7, Stack: {result6a}, Recursive: {result6b}, No Stack: {result6c}, State Machine: {result6d}")
    
    # Test case 7 - Complex expression
    s7 = "1-(     -2)"
    result7a = calculate(s7)
    result7b = calculate_recursive(s7)
    result7c = calculate_no_stack(s7)
    result7d = calculate_state_machine(s7)
    print(f"Test 7 - Expected: 3, Stack: {result7a}, Recursive: {result7b}, No Stack: {result7c}, State Machine: {result7d}")
    
    # Test case 8 - Zero
    s8 = "0"
    result8a = calculate(s8)
    result8b = calculate_recursive(s8)
    result8c = calculate_no_stack(s8)
    result8d = calculate_state_machine(s8)
    print(f"Test 8 - Expected: 0, Stack: {result8a}, Recursive: {result8b}, No Stack: {result8c}, State Machine: {result8d}")
    
    # Test case 9 - Multiple parentheses
    s9 = "((2+3)*(4-1))"
    result9a = calculate(s9)
    result9b = calculate_recursive(s9)
    result9c = calculate_no_stack(s9)
    result9d = calculate_state_machine(s9)
    print(f"Test 9 - Expected: 15, Stack: {result9a}, Recursive: {result9b}, No Stack: {result9c}, State Machine: {result9d}")
    
    # Test case 10 - Large numbers
    s10 = "2147483647"
    result10a = calculate(s10)
    result10b = calculate_recursive(s10)
    result10c = calculate_no_stack(s10)
    result10d = calculate_state_machine(s10)
    print(f"Test 10 - Expected: 2147483647, Stack: {result10a}, Recursive: {result10b}, No Stack: {result10c}, State Machine: {result10d}") 