"""
6. Zigzag Conversion

Problem:
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this:

P   A   H   R
A P L S I I G
Y   I   N

And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows.

Example 1:
Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"

Example 2:
Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I

Example 3:
Input: s = "A", numRows = 1
Output: "A"

Time Complexity: O(n) where n is the length of string
Space Complexity: O(n) for the result string
"""


def convert(s, numRows):
    """
    Convert string to zigzag pattern using simulation.
    
    Args:
        s: Input string
        numRows: Number of rows in zigzag pattern
    
    Returns:
        String read line by line from zigzag pattern
    """
    if numRows == 1:
        return s
    
    rows = [''] * numRows
    current_row = 0
    going_down = False
    
    for char in s:
        rows[current_row] += char
        
        # Change direction when we hit top or bottom row
        if current_row == 0 or current_row == numRows - 1:
            going_down = not going_down
        
        # Move to next row
        current_row += 1 if going_down else -1
    
    return ''.join(rows)


def convert_mathematical(s, numRows):
    """
    Convert string to zigzag pattern using mathematical approach.
    
    Args:
        s: Input string
        numRows: Number of rows in zigzag pattern
    
    Returns:
        String read line by line from zigzag pattern
    """
    if numRows == 1:
        return s
    
    result = []
    n = len(s)
    cycle_len = 2 * numRows - 2
    
    for i in range(numRows):
        for j in range(0, n, cycle_len):
            # First character of the cycle
            if j + i < n:
                result.append(s[j + i])
            
            # Second character of the cycle (if not first or last row)
            if i != 0 and i != numRows - 1:
                second_char_index = j + cycle_len - i
                if second_char_index < n:
                    result.append(s[second_char_index])
    
    return ''.join(result)


def convert_by_rows(s, numRows):
    """
    Convert string to zigzag pattern by processing each row.
    
    Args:
        s: Input string
        numRows: Number of rows in zigzag pattern
    
    Returns:
        String read line by line from zigzag pattern
    """
    if numRows == 1:
        return s
    
    result = []
    n = len(s)
    
    for row in range(numRows):
        if row == 0 or row == numRows - 1:
            # First and last rows have characters at regular intervals
            step = 2 * (numRows - 1)
            for i in range(row, n, step):
                result.append(s[i])
        else:
            # Middle rows have characters at two different intervals
            step1 = 2 * (numRows - 1 - row)
            step2 = 2 * row
            
            i = row
            use_step1 = True
            
            while i < n:
                result.append(s[i])
                i += step1 if use_step1 else step2
                use_step1 = not use_step1
    
    return ''.join(result)


def convert_matrix(s, numRows):
    """
    Convert string to zigzag pattern using matrix approach.
    
    Args:
        s: Input string
        numRows: Number of rows in zigzag pattern
    
    Returns:
        String read line by line from zigzag pattern
    """
    if numRows == 1:
        return s
    
    n = len(s)
    cycle_len = 2 * numRows - 2
    num_cycles = (n + cycle_len - 1) // cycle_len
    num_cols = num_cycles * (numRows - 1)
    
    # Create matrix
    matrix = [['' for _ in range(num_cols)] for _ in range(numRows)]
    
    # Fill matrix
    char_index = 0
    for cycle in range(num_cycles):
        if char_index >= n:
            break
        
        # Fill down
        col = cycle * (numRows - 1)
        for row in range(numRows):
            if char_index < n:
                matrix[row][col] = s[char_index]
                char_index += 1
        
        # Fill diagonally up
        for i in range(1, numRows - 1):
            if char_index < n:
                matrix[numRows - 1 - i][col + i] = s[char_index]
                char_index += 1
    
    # Read row by row
    result = []
    for row in range(numRows):
        for col in range(num_cols):
            if matrix[row][col]:
                result.append(matrix[row][col])
    
    return ''.join(result)


def convert_list_comprehension(s, numRows):
    """
    Convert string to zigzag pattern using list comprehension.
    
    Args:
        s: Input string
        numRows: Number of rows in zigzag pattern
    
    Returns:
        String read line by line from zigzag pattern
    """
    if numRows == 1:
        return s
    
    n = len(s)
    cycle_len = 2 * numRows - 2
    
    result = []
    
    for row in range(numRows):
        if row == 0 or row == numRows - 1:
            # First and last rows
            result.extend([s[i] for i in range(row, n, cycle_len)])
        else:
            # Middle rows
            for start in range(row, n, cycle_len):
                result.append(s[start])
                diagonal_pos = start + cycle_len - 2 * row
                if diagonal_pos < n:
                    result.append(s[diagonal_pos])
    
    return ''.join(result)


def convert_recursive(s, numRows):
    """
    Convert string to zigzag pattern using recursion.
    
    Args:
        s: Input string
        numRows: Number of rows in zigzag pattern
    
    Returns:
        String read line by line from zigzag pattern
    """
    if numRows == 1:
        return s
    
    def get_chars_for_row(row):
        """Get all characters for a specific row"""
        chars = []
        n = len(s)
        cycle_len = 2 * numRows - 2
        
        if row == 0 or row == numRows - 1:
            # First and last rows
            i = row
            while i < n:
                chars.append(s[i])
                i += cycle_len
        else:
            # Middle rows
            i = row
            use_first_step = True
            
            while i < n:
                chars.append(s[i])
                
                if use_first_step:
                    i += cycle_len - 2 * row
                else:
                    i += 2 * row
                
                use_first_step = not use_first_step
        
        return chars
    
    result = []
    for row in range(numRows):
        result.extend(get_chars_for_row(row))
    
    return ''.join(result)


def convert_functional(s, numRows):
    """
    Convert string to zigzag pattern using functional programming.
    
    Args:
        s: Input string
        numRows: Number of rows in zigzag pattern
    
    Returns:
        String read line by line from zigzag pattern
    """
    if numRows == 1:
        return s
    
    from functools import reduce
    
    def add_char_to_row(rows, char_info):
        char, index = char_info
        cycle_len = 2 * numRows - 2
        position_in_cycle = index % cycle_len
        
        if position_in_cycle < numRows:
            row = position_in_cycle
        else:
            row = cycle_len - position_in_cycle
        
        rows[row] += char
        return rows
    
    initial_rows = [''] * numRows
    char_with_index = [(char, i) for i, char in enumerate(s)]
    
    rows = reduce(add_char_to_row, char_with_index, initial_rows)
    
    return ''.join(rows)


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1 = "PAYPALISHIRING"
    numRows1 = 3
    result1a = convert(s1, numRows1)
    result1b = convert_mathematical(s1, numRows1)
    result1c = convert_by_rows(s1, numRows1)
    result1d = convert_matrix(s1, numRows1)
    result1e = convert_list_comprehension(s1, numRows1)
    result1f = convert_recursive(s1, numRows1)
    result1g = convert_functional(s1, numRows1)
    print(f"Test 1 - String: '{s1}', Rows: {numRows1}, Expected: 'PAHNAPLSIIGYIR'")
    print(f"Simulation: '{result1a}', Mathematical: '{result1b}', ByRows: '{result1c}', Matrix: '{result1d}', ListComp: '{result1e}', Recursive: '{result1f}', Functional: '{result1g}'")
    print()
    
    # Test case 2
    s2 = "PAYPALISHIRING"
    numRows2 = 4
    result2a = convert(s2, numRows2)
    result2b = convert_mathematical(s2, numRows2)
    result2c = convert_by_rows(s2, numRows2)
    result2d = convert_matrix(s2, numRows2)
    result2e = convert_list_comprehension(s2, numRows2)
    result2f = convert_recursive(s2, numRows2)
    result2g = convert_functional(s2, numRows2)
    print(f"Test 2 - String: '{s2}', Rows: {numRows2}, Expected: 'PINALSIGYAHRPI'")
    print(f"Simulation: '{result2a}', Mathematical: '{result2b}', ByRows: '{result2c}', Matrix: '{result2d}', ListComp: '{result2e}', Recursive: '{result2f}', Functional: '{result2g}'")
    print()
    
    # Test case 3
    s3 = "A"
    numRows3 = 1
    result3a = convert(s3, numRows3)
    result3b = convert_mathematical(s3, numRows3)
    result3c = convert_by_rows(s3, numRows3)
    result3d = convert_matrix(s3, numRows3)
    result3e = convert_list_comprehension(s3, numRows3)
    result3f = convert_recursive(s3, numRows3)
    result3g = convert_functional(s3, numRows3)
    print(f"Test 3 - String: '{s3}', Rows: {numRows3}, Expected: 'A'")
    print(f"Simulation: '{result3a}', Mathematical: '{result3b}', ByRows: '{result3c}', Matrix: '{result3d}', ListComp: '{result3e}', Recursive: '{result3f}', Functional: '{result3g}'")
    print()
    
    # Test case 4
    s4 = "AB"
    numRows4 = 1
    result4a = convert(s4, numRows4)
    result4b = convert_mathematical(s4, numRows4)
    result4c = convert_by_rows(s4, numRows4)
    result4d = convert_matrix(s4, numRows4)
    result4e = convert_list_comprehension(s4, numRows4)
    result4f = convert_recursive(s4, numRows4)
    result4g = convert_functional(s4, numRows4)
    print(f"Test 4 - String: '{s4}', Rows: {numRows4}, Expected: 'AB'")
    print(f"Simulation: '{result4a}', Mathematical: '{result4b}', ByRows: '{result4c}', Matrix: '{result4d}', ListComp: '{result4e}', Recursive: '{result4f}', Functional: '{result4g}'")
    print()
    
    # Test case 5
    s5 = "ABCDE"
    numRows5 = 2
    result5a = convert(s5, numRows5)
    result5b = convert_mathematical(s5, numRows5)
    result5c = convert_by_rows(s5, numRows5)
    result5d = convert_matrix(s5, numRows5)
    result5e = convert_list_comprehension(s5, numRows5)
    result5f = convert_recursive(s5, numRows5)
    result5g = convert_functional(s5, numRows5)
    print(f"Test 5 - String: '{s5}', Rows: {numRows5}, Expected: 'ACEBD'")
    print(f"Simulation: '{result5a}', Mathematical: '{result5b}', ByRows: '{result5c}', Matrix: '{result5d}', ListComp: '{result5e}', Recursive: '{result5f}', Functional: '{result5g}'")
    print()
    
    # Test case 6
    s6 = "ABCDEFGHIJK"
    numRows6 = 3
    result6a = convert(s6, numRows6)
    result6b = convert_mathematical(s6, numRows6)
    result6c = convert_by_rows(s6, numRows6)
    result6d = convert_matrix(s6, numRows6)
    result6e = convert_list_comprehension(s6, numRows6)
    result6f = convert_recursive(s6, numRows6)
    result6g = convert_functional(s6, numRows6)
    print(f"Test 6 - String: '{s6}', Rows: {numRows6}, Expected: 'AGBFHCEIJDK'")
    print(f"Simulation: '{result6a}', Mathematical: '{result6b}', ByRows: '{result6c}', Matrix: '{result6d}', ListComp: '{result6e}', Recursive: '{result6f}', Functional: '{result6g}'")
    print()
    
    # Test case 7
    s7 = "ABCDEFGHIJKLMNOP"
    numRows7 = 5
    result7a = convert(s7, numRows7)
    result7b = convert_mathematical(s7, numRows7)
    result7c = convert_by_rows(s7, numRows7)
    result7d = convert_matrix(s7, numRows7)
    result7e = convert_list_comprehension(s7, numRows7)
    result7f = convert_recursive(s7, numRows7)
    result7g = convert_functional(s7, numRows7)
    print(f"Test 7 - String: '{s7}', Rows: {numRows7}, Expected: 'AIBHCGJDKFLMEONP'")
    print(f"Simulation: '{result7a}', Mathematical: '{result7b}', ByRows: '{result7c}', Matrix: '{result7d}', ListComp: '{result7e}', Recursive: '{result7f}', Functional: '{result7g}'")
    print()
    
    # Test case 8
    s8 = "ABC"
    numRows8 = 3
    result8a = convert(s8, numRows8)
    result8b = convert_mathematical(s8, numRows8)
    result8c = convert_by_rows(s8, numRows8)
    result8d = convert_matrix(s8, numRows8)
    result8e = convert_list_comprehension(s8, numRows8)
    result8f = convert_recursive(s8, numRows8)
    result8g = convert_functional(s8, numRows8)
    print(f"Test 8 - String: '{s8}', Rows: {numRows8}, Expected: 'ABC'")
    print(f"Simulation: '{result8a}', Mathematical: '{result8b}', ByRows: '{result8c}', Matrix: '{result8d}', ListComp: '{result8e}', Recursive: '{result8f}', Functional: '{result8g}'")
    print()
    
    # Test case 9
    s9 = "ABCDEFG"
    numRows9 = 4
    result9a = convert(s9, numRows9)
    result9b = convert_mathematical(s9, numRows9)
    result9c = convert_by_rows(s9, numRows9)
    result9d = convert_matrix(s9, numRows9)
    result9e = convert_list_comprehension(s9, numRows9)
    result9f = convert_recursive(s9, numRows9)
    result9g = convert_functional(s9, numRows9)
    print(f"Test 9 - String: '{s9}', Rows: {numRows9}, Expected: 'AGBFCED'")
    print(f"Simulation: '{result9a}', Mathematical: '{result9b}', ByRows: '{result9c}', Matrix: '{result9d}', ListComp: '{result9e}', Recursive: '{result9f}', Functional: '{result9g}'")
    print()
    
    # Test case 10
    s10 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numRows10 = 6
    result10a = convert(s10, numRows10)
    result10b = convert_mathematical(s10, numRows10)
    result10c = convert_by_rows(s10, numRows10)
    result10d = convert_matrix(s10, numRows10)
    result10e = convert_list_comprehension(s10, numRows10)
    result10f = convert_recursive(s10, numRows10)
    result10g = convert_functional(s10, numRows10)
    print(f"Test 10 - String: '{s10}', Rows: {numRows10}, Expected: 'AKBJLCIMDHNEGOFPQRSTVUXWYZ'")
    print(f"Simulation: '{result10a}', Mathematical: '{result10b}', ByRows: '{result10c}', Matrix: '{result10d}', ListComp: '{result10e}', Recursive: '{result10f}', Functional: '{result10g}'") 