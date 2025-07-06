"""
54. Spiral Matrix

Problem:
Given an m x n matrix, return all elements of the matrix in spiral order.

Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]

Example 2:
Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]

Time Complexity: O(m*n) where m and n are matrix dimensions
Space Complexity: O(1) excluding the result array
"""


def spiral_order(matrix):
    """
    Return elements of matrix in spiral order using boundary approach.
    
    Args:
        matrix: 2D list representing the matrix
    
    Returns:
        List of elements in spiral order
    """
    if not matrix or not matrix[0]:
        return []
    
    m, n = len(matrix), len(matrix[0])
    result = []
    
    # Define boundaries
    top, bottom = 0, m - 1
    left, right = 0, n - 1
    
    while top <= bottom and left <= right:
        # Traverse right along top row
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Traverse down along right column
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Traverse left along bottom row (if we still have rows)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Traverse up along left column (if we still have columns)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result


def spiral_order_direction_vector(matrix):
    """
    Return elements of matrix in spiral order using direction vectors.
    
    Args:
        matrix: 2D list representing the matrix
    
    Returns:
        List of elements in spiral order
    """
    if not matrix or not matrix[0]:
        return []
    
    m, n = len(matrix), len(matrix[0])
    result = []
    visited = [[False] * n for _ in range(m)]
    
    # Direction vectors: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_idx = 0
    
    row, col = 0, 0
    
    for _ in range(m * n):
        result.append(matrix[row][col])
        visited[row][col] = True
        
        # Calculate next position
        dr, dc = directions[direction_idx]
        next_row, next_col = row + dr, col + dc
        
        # Check if we need to change direction
        if (next_row < 0 or next_row >= m or 
            next_col < 0 or next_col >= n or 
            visited[next_row][next_col]):
            direction_idx = (direction_idx + 1) % 4
            dr, dc = directions[direction_idx]
            next_row, next_col = row + dr, col + dc
        
        row, col = next_row, next_col
    
    return result


def spiral_order_recursive(matrix):
    """
    Return elements of matrix in spiral order using recursion.
    
    Args:
        matrix: 2D list representing the matrix
    
    Returns:
        List of elements in spiral order
    """
    def spiral_helper(matrix, top, bottom, left, right):
        if top > bottom or left > right:
            return []
        
        result = []
        
        # Traverse right along top row
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        
        # Traverse down along right column
        for row in range(top + 1, bottom + 1):
            result.append(matrix[row][right])
        
        # Traverse left along bottom row (if we have more than one row)
        if top < bottom:
            for col in range(right - 1, left - 1, -1):
                result.append(matrix[bottom][col])
        
        # Traverse up along left column (if we have more than one column)
        if left < right:
            for row in range(bottom - 1, top, -1):
                result.append(matrix[row][left])
        
        # Recursively process inner matrix
        result.extend(spiral_helper(matrix, top + 1, bottom - 1, left + 1, right - 1))
        
        return result
    
    if not matrix or not matrix[0]:
        return []
    
    return spiral_helper(matrix, 0, len(matrix) - 1, 0, len(matrix[0]) - 1)


def spiral_order_layer_by_layer(matrix):
    """
    Return elements of matrix in spiral order processing layer by layer.
    
    Args:
        matrix: 2D list representing the matrix
    
    Returns:
        List of elements in spiral order
    """
    if not matrix or not matrix[0]:
        return []
    
    m, n = len(matrix), len(matrix[0])
    result = []
    
    # Process each layer from outside to inside
    for layer in range(min(m, n) // 2):
        # Top row
        for col in range(layer, n - layer):
            result.append(matrix[layer][col])
        
        # Right column (excluding top corner)
        for row in range(layer + 1, m - layer):
            result.append(matrix[row][n - 1 - layer])
        
        # Bottom row (excluding right corner, if not same as top row)
        if layer < m - 1 - layer:
            for col in range(n - 2 - layer, layer - 1, -1):
                result.append(matrix[m - 1 - layer][col])
        
        # Left column (excluding both corners, if not same as right column)
        if layer < n - 1 - layer:
            for row in range(m - 2 - layer, layer, -1):
                result.append(matrix[row][layer])
    
    return result


def spiral_order_state_machine(matrix):
    """
    Return elements of matrix in spiral order using state machine approach.
    
    Args:
        matrix: 2D list representing the matrix
    
    Returns:
        List of elements in spiral order
    """
    if not matrix or not matrix[0]:
        return []
    
    m, n = len(matrix), len(matrix[0])
    result = []
    visited = [[False] * n for _ in range(m)]
    
    # States: 0=right, 1=down, 2=left, 3=up
    state = 0
    row, col = 0, 0
    
    for _ in range(m * n):
        result.append(matrix[row][col])
        visited[row][col] = True
        
        # Try to continue in current direction
        if state == 0:  # Moving right
            if col + 1 < n and not visited[row][col + 1]:
                col += 1
            else:
                state = 1
                row += 1
        elif state == 1:  # Moving down
            if row + 1 < m and not visited[row + 1][col]:
                row += 1
            else:
                state = 2
                col -= 1
        elif state == 2:  # Moving left
            if col - 1 >= 0 and not visited[row][col - 1]:
                col -= 1
            else:
                state = 3
                row -= 1
        elif state == 3:  # Moving up
            if row - 1 >= 0 and not visited[row - 1][col]:
                row -= 1
            else:
                state = 0
                col += 1
    
    return result


# Test cases
if __name__ == "__main__":
    # Test case 1
    matrix1 = [[1,2,3],[4,5,6],[7,8,9]]
    result1a = spiral_order(matrix1)
    result1b = spiral_order_direction_vector(matrix1)
    result1c = spiral_order_recursive(matrix1)
    result1d = spiral_order_layer_by_layer(matrix1)
    result1e = spiral_order_state_machine(matrix1)
    print(f"Test 1 - Input: {matrix1}")
    print(f"Expected: [1,2,3,6,9,8,7,4,5]")
    print(f"Boundary: {result1a}")
    print(f"Direction: {result1b}")
    print(f"Recursive: {result1c}")
    print(f"Layer: {result1d}")
    print(f"StateMachine: {result1e}")
    print()
    
    # Test case 2
    matrix2 = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    result2a = spiral_order(matrix2)
    result2b = spiral_order_direction_vector(matrix2)
    result2c = spiral_order_recursive(matrix2)
    result2d = spiral_order_layer_by_layer(matrix2)
    result2e = spiral_order_state_machine(matrix2)
    print(f"Test 2 - Input: {matrix2}")
    print(f"Expected: [1,2,3,4,8,12,11,10,9,5,6,7]")
    print(f"Boundary: {result2a}")
    print(f"Direction: {result2b}")
    print(f"Recursive: {result2c}")
    print(f"Layer: {result2d}")
    print(f"StateMachine: {result2e}")
    print()
    
    # Test case 3 - Single row
    matrix3 = [[1,2,3,4]]
    result3a = spiral_order(matrix3)
    result3b = spiral_order_direction_vector(matrix3)
    result3c = spiral_order_recursive(matrix3)
    result3d = spiral_order_layer_by_layer(matrix3)
    result3e = spiral_order_state_machine(matrix3)
    print(f"Test 3 - Input: {matrix3}")
    print(f"Expected: [1,2,3,4]")
    print(f"Boundary: {result3a}")
    print(f"Direction: {result3b}")
    print(f"Recursive: {result3c}")
    print(f"Layer: {result3d}")
    print(f"StateMachine: {result3e}")
    print()
    
    # Test case 4 - Single column
    matrix4 = [[1],[2],[3],[4]]
    result4a = spiral_order(matrix4)
    result4b = spiral_order_direction_vector(matrix4)
    result4c = spiral_order_recursive(matrix4)
    result4d = spiral_order_layer_by_layer(matrix4)
    result4e = spiral_order_state_machine(matrix4)
    print(f"Test 4 - Input: {matrix4}")
    print(f"Expected: [1,2,3,4]")
    print(f"Boundary: {result4a}")
    print(f"Direction: {result4b}")
    print(f"Recursive: {result4c}")
    print(f"Layer: {result4d}")
    print(f"StateMachine: {result4e}")
    print()
    
    # Test case 5 - Single element
    matrix5 = [[1]]
    result5a = spiral_order(matrix5)
    result5b = spiral_order_direction_vector(matrix5)
    result5c = spiral_order_recursive(matrix5)
    result5d = spiral_order_layer_by_layer(matrix5)
    result5e = spiral_order_state_machine(matrix5)
    print(f"Test 5 - Input: {matrix5}")
    print(f"Expected: [1]")
    print(f"Boundary: {result5a}")
    print(f"Direction: {result5b}")
    print(f"Recursive: {result5c}")
    print(f"Layer: {result5d}")
    print(f"StateMachine: {result5e}")
    print()
    
    # Test case 6 - Empty matrix
    matrix6 = []
    result6a = spiral_order(matrix6)
    result6b = spiral_order_direction_vector(matrix6)
    result6c = spiral_order_recursive(matrix6)
    result6d = spiral_order_layer_by_layer(matrix6)
    result6e = spiral_order_state_machine(matrix6)
    print(f"Test 6 - Input: {matrix6}")
    print(f"Expected: []")
    print(f"Boundary: {result6a}")
    print(f"Direction: {result6b}")
    print(f"Recursive: {result6c}")
    print(f"Layer: {result6d}")
    print(f"StateMachine: {result6e}")
    print()
    
    # Test case 7 - 2x2 matrix
    matrix7 = [[1,2],[3,4]]
    result7a = spiral_order(matrix7)
    result7b = spiral_order_direction_vector(matrix7)
    result7c = spiral_order_recursive(matrix7)
    result7d = spiral_order_layer_by_layer(matrix7)
    result7e = spiral_order_state_machine(matrix7)
    print(f"Test 7 - Input: {matrix7}")
    print(f"Expected: [1,2,4,3]")
    print(f"Boundary: {result7a}")
    print(f"Direction: {result7b}")
    print(f"Recursive: {result7c}")
    print(f"Layer: {result7d}")
    print(f"StateMachine: {result7e}")
    print()
    
    # Test case 8 - 1x5 matrix
    matrix8 = [[1,2,3,4,5]]
    result8a = spiral_order(matrix8)
    result8b = spiral_order_direction_vector(matrix8)
    result8c = spiral_order_recursive(matrix8)
    result8d = spiral_order_layer_by_layer(matrix8)
    result8e = spiral_order_state_machine(matrix8)
    print(f"Test 8 - Input: {matrix8}")
    print(f"Expected: [1,2,3,4,5]")
    print(f"Boundary: {result8a}")
    print(f"Direction: {result8b}")
    print(f"Recursive: {result8c}")
    print(f"Layer: {result8d}")
    print(f"StateMachine: {result8e}")
    print()
    
    # Test case 9 - 5x1 matrix
    matrix9 = [[1],[2],[3],[4],[5]]
    result9a = spiral_order(matrix9)
    result9b = spiral_order_direction_vector(matrix9)
    result9c = spiral_order_recursive(matrix9)
    result9d = spiral_order_layer_by_layer(matrix9)
    result9e = spiral_order_state_machine(matrix9)
    print(f"Test 9 - Input: {matrix9}")
    print(f"Expected: [1,2,3,4,5]")
    print(f"Boundary: {result9a}")
    print(f"Direction: {result9b}")
    print(f"Recursive: {result9c}")
    print(f"Layer: {result9d}")
    print(f"StateMachine: {result9e}")
    print()
    
    # Test case 10 - 4x4 matrix
    matrix10 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    result10a = spiral_order(matrix10)
    result10b = spiral_order_direction_vector(matrix10)
    result10c = spiral_order_recursive(matrix10)
    result10d = spiral_order_layer_by_layer(matrix10)
    result10e = spiral_order_state_machine(matrix10)
    print(f"Test 10 - Input: {matrix10}")
    print(f"Expected: [1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10]")
    print(f"Boundary: {result10a}")
    print(f"Direction: {result10b}")
    print(f"Recursive: {result10c}")
    print(f"Layer: {result10d}")
    print(f"StateMachine: {result10e}") 