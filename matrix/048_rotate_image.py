"""
48. Rotate Image

Problem:
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. 
DO NOT allocate another 2D matrix and do the rotation.

Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:
Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

Time Complexity: O(n²) where n is the matrix dimension
Space Complexity: O(1) for in-place rotation
"""


def rotate(matrix):
    """
    Rotate matrix 90 degrees clockwise using transpose and reverse.
    
    Args:
        matrix: n x n 2D list representing the matrix (modified in-place)
    """
    n = len(matrix)
    
    # Step 1: Transpose the matrix (swap matrix[i][j] with matrix[j][i])
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Step 2: Reverse each row
    for i in range(n):
        matrix[i].reverse()


def rotate_layer_by_layer(matrix):
    """
    Rotate matrix 90 degrees clockwise by rotating each layer.
    
    Args:
        matrix: n x n 2D list representing the matrix (modified in-place)
    """
    n = len(matrix)
    
    # Process each layer from outside to inside
    for layer in range(n // 2):
        first = layer
        last = n - 1 - layer
        
        # Rotate elements in current layer
        for i in range(first, last):
            offset = i - first
            
            # Save top element
            top = matrix[first][i]
            
            # top = left
            matrix[first][i] = matrix[last - offset][first]
            
            # left = bottom
            matrix[last - offset][first] = matrix[last][last - offset]
            
            # bottom = right
            matrix[last][last - offset] = matrix[i][last]
            
            # right = top
            matrix[i][last] = top


def rotate_four_way_swap(matrix):
    """
    Rotate matrix 90 degrees clockwise using four-way element swap.
    
    Args:
        matrix: n x n 2D list representing the matrix (modified in-place)
    """
    n = len(matrix)
    
    # Process each quadrant
    for i in range(n // 2):
        for j in range(i, n - i - 1):
            # Store current element
            temp = matrix[i][j]
            
            # Move elements in clockwise direction
            matrix[i][j] = matrix[n - 1 - j][i]
            matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
            matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
            matrix[j][n - 1 - i] = temp


def rotate_recursive(matrix):
    """
    Rotate matrix 90 degrees clockwise using recursion.
    
    Args:
        matrix: n x n 2D list representing the matrix (modified in-place)
    """
    def rotate_ring(matrix, start, end):
        """Rotate elements in a ring from start to end index"""
        if start >= end:
            return
        
        for i in range(start, end):
            # Calculate positions
            top = matrix[start][i]
            right = matrix[i][end]
            bottom = matrix[end][end - (i - start)]
            left = matrix[end - (i - start)][start]
            
            # Rotate elements
            matrix[start][i] = left
            matrix[i][end] = top
            matrix[end][end - (i - start)] = right
            matrix[end - (i - start)][start] = bottom
        
        # Recursively rotate inner ring
        rotate_ring(matrix, start + 1, end - 1)
    
    n = len(matrix)
    rotate_ring(matrix, 0, n - 1)


def rotate_with_coordinates(matrix):
    """
    Rotate matrix 90 degrees clockwise using coordinate transformation.
    
    Args:
        matrix: n x n 2D list representing the matrix (modified in-place)
    """
    n = len(matrix)
    
    # Create temporary matrix to store rotated values
    temp = [[0] * n for _ in range(n)]
    
    # Apply rotation transformation: (i, j) -> (j, n-1-i)
    for i in range(n):
        for j in range(n):
            temp[j][n - 1 - i] = matrix[i][j]
    
    # Copy back to original matrix
    for i in range(n):
        for j in range(n):
            matrix[i][j] = temp[i][j]


def rotate_90_degrees_original(matrix):
    """
    Create a copy of the original matrix and rotate it.
    Note: This violates the in-place requirement but included for completeness.
    
    Args:
        matrix: n x n 2D list representing the matrix (modified in-place)
    """
    n = len(matrix)
    
    # Create a copy of the original matrix
    original = [row[:] for row in matrix]
    
    # Rotate: new[j][n-1-i] = original[i][j]
    for i in range(n):
        for j in range(n):
            matrix[j][n - 1 - i] = original[i][j]


# Test cases
if __name__ == "__main__":
    # Test case 1
    matrix1 = [[1,2,3],[4,5,6],[7,8,9]]
    expected1 = [[7,4,1],[8,5,2],[9,6,3]]
    matrix1_copy = [row[:] for row in matrix1]
    rotate(matrix1_copy)
    print(f"Test 1 - Input: {matrix1}")
    print(f"Expected: {expected1}")
    print(f"TransposeReverse: {matrix1_copy}")
    
    matrix1_copy = [row[:] for row in matrix1]
    rotate_layer_by_layer(matrix1_copy)
    print(f"LayerByLayer: {matrix1_copy}")
    
    matrix1_copy = [row[:] for row in matrix1]
    rotate_four_way_swap(matrix1_copy)
    print(f"FourWaySwap: {matrix1_copy}")
    
    matrix1_copy = [row[:] for row in matrix1]
    rotate_recursive(matrix1_copy)
    print(f"Recursive: {matrix1_copy}")
    
    matrix1_copy = [row[:] for row in matrix1]
    rotate_with_coordinates(matrix1_copy)
    print(f"Coordinates: {matrix1_copy}")
    print()
    
    # Test case 2
    matrix2 = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    expected2 = [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
    matrix2_copy = [row[:] for row in matrix2]
    rotate(matrix2_copy)
    print(f"Test 2 - Input: {matrix2}")
    print(f"Expected: {expected2}")
    print(f"TransposeReverse: {matrix2_copy}")
    
    matrix2_copy = [row[:] for row in matrix2]
    rotate_layer_by_layer(matrix2_copy)
    print(f"LayerByLayer: {matrix2_copy}")
    
    matrix2_copy = [row[:] for row in matrix2]
    rotate_four_way_swap(matrix2_copy)
    print(f"FourWaySwap: {matrix2_copy}")
    
    matrix2_copy = [row[:] for row in matrix2]
    rotate_recursive(matrix2_copy)
    print(f"Recursive: {matrix2_copy}")
    
    matrix2_copy = [row[:] for row in matrix2]
    rotate_with_coordinates(matrix2_copy)
    print(f"Coordinates: {matrix2_copy}")
    print()
    
    # Test case 3 - Single element
    matrix3 = [[1]]
    expected3 = [[1]]
    matrix3_copy = [row[:] for row in matrix3]
    rotate(matrix3_copy)
    print(f"Test 3 - Input: {matrix3}")
    print(f"Expected: {expected3}")
    print(f"TransposeReverse: {matrix3_copy}")
    
    matrix3_copy = [row[:] for row in matrix3]
    rotate_layer_by_layer(matrix3_copy)
    print(f"LayerByLayer: {matrix3_copy}")
    
    matrix3_copy = [row[:] for row in matrix3]
    rotate_four_way_swap(matrix3_copy)
    print(f"FourWaySwap: {matrix3_copy}")
    
    matrix3_copy = [row[:] for row in matrix3]
    rotate_recursive(matrix3_copy)
    print(f"Recursive: {matrix3_copy}")
    
    matrix3_copy = [row[:] for row in matrix3]
    rotate_with_coordinates(matrix3_copy)
    print(f"Coordinates: {matrix3_copy}")
    print()
    
    # Test case 4 - 2x2 matrix
    matrix4 = [[1,2],[3,4]]
    expected4 = [[3,1],[4,2]]
    matrix4_copy = [row[:] for row in matrix4]
    rotate(matrix4_copy)
    print(f"Test 4 - Input: {matrix4}")
    print(f"Expected: {expected4}")
    print(f"TransposeReverse: {matrix4_copy}")
    
    matrix4_copy = [row[:] for row in matrix4]
    rotate_layer_by_layer(matrix4_copy)
    print(f"LayerByLayer: {matrix4_copy}")
    
    matrix4_copy = [row[:] for row in matrix4]
    rotate_four_way_swap(matrix4_copy)
    print(f"FourWaySwap: {matrix4_copy}")
    
    matrix4_copy = [row[:] for row in matrix4]
    rotate_recursive(matrix4_copy)
    print(f"Recursive: {matrix4_copy}")
    
    matrix4_copy = [row[:] for row in matrix4]
    rotate_with_coordinates(matrix4_copy)
    print(f"Coordinates: {matrix4_copy}")
    print()
    
    # Test case 5 - 5x5 matrix
    matrix5 = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]
    expected5 = [[21,16,11,6,1],[22,17,12,7,2],[23,18,13,8,3],[24,19,14,9,4],[25,20,15,10,5]]
    matrix5_copy = [row[:] for row in matrix5]
    rotate(matrix5_copy)
    print(f"Test 5 - Input: {matrix5}")
    print(f"Expected: {expected5}")
    print(f"TransposeReverse: {matrix5_copy}")
    
    matrix5_copy = [row[:] for row in matrix5]
    rotate_layer_by_layer(matrix5_copy)
    print(f"LayerByLayer: {matrix5_copy}")
    
    matrix5_copy = [row[:] for row in matrix5]
    rotate_four_way_swap(matrix5_copy)
    print(f"FourWaySwap: {matrix5_copy}")
    
    matrix5_copy = [row[:] for row in matrix5]
    rotate_recursive(matrix5_copy)
    print(f"Recursive: {matrix5_copy}")
    
    matrix5_copy = [row[:] for row in matrix5]
    rotate_with_coordinates(matrix5_copy)
    print(f"Coordinates: {matrix5_copy}")
    print()
    
    # Test case 6 - 6x6 matrix
    matrix6 = [[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25,26,27,28,29,30],[31,32,33,34,35,36]]
    expected6 = [[31,25,19,13,7,1],[32,26,20,14,8,2],[33,27,21,15,9,3],[34,28,22,16,10,4],[35,29,23,17,11,5],[36,30,24,18,12,6]]
    matrix6_copy = [row[:] for row in matrix6]
    rotate(matrix6_copy)
    print(f"Test 6 - Input: {matrix6}")
    print(f"Expected: {expected6}")
    print(f"TransposeReverse: {matrix6_copy}")
    
    matrix6_copy = [row[:] for row in matrix6]
    rotate_layer_by_layer(matrix6_copy)
    print(f"LayerByLayer: {matrix6_copy}")
    
    matrix6_copy = [row[:] for row in matrix6]
    rotate_four_way_swap(matrix6_copy)
    print(f"FourWaySwap: {matrix6_copy}")
    
    matrix6_copy = [row[:] for row in matrix6]
    rotate_recursive(matrix6_copy)
    print(f"Recursive: {matrix6_copy}")
    
    matrix6_copy = [row[:] for row in matrix6]
    rotate_with_coordinates(matrix6_copy)
    print(f"Coordinates: {matrix6_copy}")
    print()
    
    # Test multiple rotations
    print("Testing multiple rotations (should return to original after 4 rotations):")
    matrix_test = [[1,2,3],[4,5,6],[7,8,9]]
    original = [row[:] for row in matrix_test]
    print(f"Original: {original}")
    
    for i in range(4):
        rotate(matrix_test)
        print(f"After {i+1} rotation(s): {matrix_test}")
    
    print(f"Back to original: {matrix_test == original}")
    print()
    
    # Test anti-clockwise rotation (3 clockwise rotations)
    print("Testing anti-clockwise rotation (3 clockwise rotations):")
    matrix_anti = [[1,2,3],[4,5,6],[7,8,9]]
    print(f"Original: {matrix_anti}")
    
    # Rotate 3 times for anti-clockwise
    for _ in range(3):
        rotate(matrix_anti)
    print(f"After anti-clockwise rotation: {matrix_anti}")
    
    # Expected anti-clockwise: [[3,6,9],[2,5,8],[1,4,7]]
    expected_anti = [[3,6,9],[2,5,8],[1,4,7]]
    print(f"Expected anti-clockwise: {expected_anti}")
    print(f"Matches expected: {matrix_anti == expected_anti}") 