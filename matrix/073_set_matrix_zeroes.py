"""
73. Set Matrix Zeroes

Problem:
Given an m x n integer matrix, if an element is 0, set its entire row and column to 0's.

You must do it in place.

Example 1:
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

Example 2:
Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

Follow up:
- A straightforward solution using O(mn) space is probably a bad idea.
- A simple improvement uses O(m + n) space, but still not the best solution.
- Could you devise a constant space solution?

Time Complexity: O(m*n) where m and n are matrix dimensions
Space Complexity: O(1) for optimal solution
"""


def set_zeroes(matrix):
    """
    Set matrix zeroes using first row and column as markers (O(1) space).
    
    Args:
        matrix: 2D list representing the matrix (modified in-place)
    """
    if not matrix or not matrix[0]:
        return
    
    m, n = len(matrix), len(matrix[0])
    
    # Check if first row and column have zeros
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))
    
    # Use first row and column as markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0  # Mark row
                matrix[0][j] = 0  # Mark column
    
    # Set zeros based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    # Handle first row
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    
    # Handle first column
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0


def set_zeroes_extra_space(matrix):
    """
    Set matrix zeroes using extra space to store zero positions.
    
    Args:
        matrix: 2D list representing the matrix (modified in-place)
    """
    if not matrix or not matrix[0]:
        return
    
    m, n = len(matrix), len(matrix[0])
    zero_rows = set()
    zero_cols = set()
    
    # Find all zero positions
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                zero_rows.add(i)
                zero_cols.add(j)
    
    # Set entire rows to zero
    for i in zero_rows:
        for j in range(n):
            matrix[i][j] = 0
    
    # Set entire columns to zero
    for j in zero_cols:
        for i in range(m):
            matrix[i][j] = 0


def set_zeroes_boolean_arrays(matrix):
    """
    Set matrix zeroes using boolean arrays (O(m+n) space).
    
    Args:
        matrix: 2D list representing the matrix (modified in-place)
    """
    if not matrix or not matrix[0]:
        return
    
    m, n = len(matrix), len(matrix[0])
    row_zeros = [False] * m
    col_zeros = [False] * n
    
    # Mark rows and columns that should be zero
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                row_zeros[i] = True
                col_zeros[j] = True
    
    # Set zeros based on markers
    for i in range(m):
        for j in range(n):
            if row_zeros[i] or col_zeros[j]:
                matrix[i][j] = 0


def set_zeroes_two_variables(matrix):
    """
    Set matrix zeroes using two variables to track first row/column.
    
    Args:
        matrix: 2D list representing the matrix (modified in-place)
    """
    if not matrix or not matrix[0]:
        return
    
    m, n = len(matrix), len(matrix[0])
    
    # Variables to track if first row/column should be zero
    first_row_zero = False
    first_col_zero = False
    
    # Check if first row has zero
    for j in range(n):
        if matrix[0][j] == 0:
            first_row_zero = True
            break
    
    # Check if first column has zero
    for i in range(m):
        if matrix[i][0] == 0:
            first_col_zero = True
            break
    
    # Use first row and column as markers for the rest
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[0][j] = 0
                matrix[i][0] = 0
    
    # Set zeros based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[0][j] == 0 or matrix[i][0] == 0:
                matrix[i][j] = 0
    
    # Handle first row and column
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0


def set_zeroes_single_pass(matrix):
    """
    Set matrix zeroes in a single pass with constant space.
    
    Args:
        matrix: 2D list representing the matrix (modified in-place)
    """
    if not matrix or not matrix[0]:
        return
    
    m, n = len(matrix), len(matrix[0])
    
    # Use a special marker for cells that should be zero
    marker = float('inf')
    
    # First pass: mark rows and columns
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                # Mark entire row
                for k in range(n):
                    if matrix[i][k] != 0:
                        matrix[i][k] = marker
                
                # Mark entire column
                for k in range(m):
                    if matrix[k][j] != 0:
                        matrix[k][j] = marker
    
    # Second pass: set markers to zero
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == marker:
                matrix[i][j] = 0


def set_zeroes_recursive(matrix):
    """
    Set matrix zeroes using recursive approach.
    
    Args:
        matrix: 2D list representing the matrix (modified in-place)
    """
    if not matrix or not matrix[0]:
        return
    
    def mark_zero(matrix, row, col):
        """Recursively mark row and column as zero"""
        m, n = len(matrix), len(matrix[0])
        
        # Mark row
        for j in range(n):
            if matrix[row][j] != 0:
                matrix[row][j] = 0
        
        # Mark column
        for i in range(m):
            if matrix[i][col] != 0:
                matrix[i][col] = 0
    
    m, n = len(matrix), len(matrix[0])
    zero_positions = []
    
    # Find all zero positions
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                zero_positions.append((i, j))
    
    # Mark zeros for each position
    for row, col in zero_positions:
        mark_zero(matrix, row, col)


# Test cases
if __name__ == "__main__":
    # Test case 1
    matrix1 = [[1,1,1],[1,0,1],[1,1,1]]
    expected1 = [[1,0,1],[0,0,0],[1,0,1]]
    matrix1_copy = [row[:] for row in matrix1]
    set_zeroes(matrix1_copy)
    print(f"Test 1 - Input: {matrix1}")
    print(f"Expected: {expected1}")
    print(f"Optimal: {matrix1_copy}")
    
    matrix1_copy = [row[:] for row in matrix1]
    set_zeroes_extra_space(matrix1_copy)
    print(f"ExtraSpace: {matrix1_copy}")
    
    matrix1_copy = [row[:] for row in matrix1]
    set_zeroes_boolean_arrays(matrix1_copy)
    print(f"BooleanArrays: {matrix1_copy}")
    
    matrix1_copy = [row[:] for row in matrix1]
    set_zeroes_two_variables(matrix1_copy)
    print(f"TwoVariables: {matrix1_copy}")
    
    matrix1_copy = [row[:] for row in matrix1]
    set_zeroes_single_pass(matrix1_copy)
    print(f"SinglePass: {matrix1_copy}")
    
    matrix1_copy = [row[:] for row in matrix1]
    set_zeroes_recursive(matrix1_copy)
    print(f"Recursive: {matrix1_copy}")
    print()
    
    # Test case 2
    matrix2 = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
    expected2 = [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
    matrix2_copy = [row[:] for row in matrix2]
    set_zeroes(matrix2_copy)
    print(f"Test 2 - Input: {matrix2}")
    print(f"Expected: {expected2}")
    print(f"Optimal: {matrix2_copy}")
    
    matrix2_copy = [row[:] for row in matrix2]
    set_zeroes_extra_space(matrix2_copy)
    print(f"ExtraSpace: {matrix2_copy}")
    
    matrix2_copy = [row[:] for row in matrix2]
    set_zeroes_boolean_arrays(matrix2_copy)
    print(f"BooleanArrays: {matrix2_copy}")
    
    matrix2_copy = [row[:] for row in matrix2]
    set_zeroes_two_variables(matrix2_copy)
    print(f"TwoVariables: {matrix2_copy}")
    
    matrix2_copy = [row[:] for row in matrix2]
    set_zeroes_single_pass(matrix2_copy)
    print(f"SinglePass: {matrix2_copy}")
    
    matrix2_copy = [row[:] for row in matrix2]
    set_zeroes_recursive(matrix2_copy)
    print(f"Recursive: {matrix2_copy}")
    print()
    
    # Test case 3 - Single element zero
    matrix3 = [[0]]
    expected3 = [[0]]
    matrix3_copy = [row[:] for row in matrix3]
    set_zeroes(matrix3_copy)
    print(f"Test 3 - Input: {matrix3}")
    print(f"Expected: {expected3}")
    print(f"Optimal: {matrix3_copy}")
    print()
    
    # Test case 4 - Single element non-zero
    matrix4 = [[1]]
    expected4 = [[1]]
    matrix4_copy = [row[:] for row in matrix4]
    set_zeroes(matrix4_copy)
    print(f"Test 4 - Input: {matrix4}")
    print(f"Expected: {expected4}")
    print(f"Optimal: {matrix4_copy}")
    print()
    
    # Test case 5 - No zeros
    matrix5 = [[1,2,3],[4,5,6],[7,8,9]]
    expected5 = [[1,2,3],[4,5,6],[7,8,9]]
    matrix5_copy = [row[:] for row in matrix5]
    set_zeroes(matrix5_copy)
    print(f"Test 5 - Input: {matrix5}")
    print(f"Expected: {expected5}")
    print(f"Optimal: {matrix5_copy}")
    print()
    
    # Test case 6 - All zeros
    matrix6 = [[0,0,0],[0,0,0],[0,0,0]]
    expected6 = [[0,0,0],[0,0,0],[0,0,0]]
    matrix6_copy = [row[:] for row in matrix6]
    set_zeroes(matrix6_copy)
    print(f"Test 6 - Input: {matrix6}")
    print(f"Expected: {expected6}")
    print(f"Optimal: {matrix6_copy}")
    print()
    
    # Test case 7 - First row has zero
    matrix7 = [[0,1,2],[3,4,5],[6,7,8]]
    expected7 = [[0,0,0],[0,4,5],[0,7,8]]
    matrix7_copy = [row[:] for row in matrix7]
    set_zeroes(matrix7_copy)
    print(f"Test 7 - Input: {matrix7}")
    print(f"Expected: {expected7}")
    print(f"Optimal: {matrix7_copy}")
    print()
    
    # Test case 8 - First column has zero
    matrix8 = [[1,2,3],[0,4,5],[6,7,8]]
    expected8 = [[1,2,3],[0,0,0],[6,7,8]]
    matrix8_copy = [row[:] for row in matrix8]
    set_zeroes(matrix8_copy)
    print(f"Test 8 - Input: {matrix8}")
    print(f"Expected: {expected8}")
    print(f"Optimal: {matrix8_copy}")
    print()
    
    # Test case 9 - Corner zero
    matrix9 = [[0,2,3],[4,5,6],[7,8,9]]
    expected9 = [[0,0,0],[0,5,6],[0,8,9]]
    matrix9_copy = [row[:] for row in matrix9]
    set_zeroes(matrix9_copy)
    print(f"Test 9 - Input: {matrix9}")
    print(f"Expected: {expected9}")
    print(f"Optimal: {matrix9_copy}")
    print()
    
    # Test case 10 - Multiple zeros
    matrix10 = [[1,0,3],[0,5,6],[7,8,0]]
    expected10 = [[0,0,0],[0,0,0],[0,0,0]]
    matrix10_copy = [row[:] for row in matrix10]
    set_zeroes(matrix10_copy)
    print(f"Test 10 - Input: {matrix10}")
    print(f"Expected: {expected10}")
    print(f"Optimal: {matrix10_copy}")
    print()
    
    # Test case 11 - Single row
    matrix11 = [[1,0,3,4]]
    expected11 = [[0,0,0,0]]
    matrix11_copy = [row[:] for row in matrix11]
    set_zeroes(matrix11_copy)
    print(f"Test 11 - Input: {matrix11}")
    print(f"Expected: {expected11}")
    print(f"Optimal: {matrix11_copy}")
    print()
    
    # Test case 12 - Single column
    matrix12 = [[1],[0],[3]]
    expected12 = [[0],[0],[0]]
    matrix12_copy = [row[:] for row in matrix12]
    set_zeroes(matrix12_copy)
    print(f"Test 12 - Input: {matrix12}")
    print(f"Expected: {expected12}")
    print(f"Optimal: {matrix12_copy}") 