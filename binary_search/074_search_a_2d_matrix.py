"""
74. Search a 2D Matrix

You are given an m x n integer matrix matrix with the following two properties:
- Each row is sorted in non-decreasing order.
- The first integer of each row is greater than the last integer of the previous row.

Given an integer target, return true if target is in matrix or false otherwise.

You must write an algorithm in O(log(m * n)) time complexity.

Example 1:
Input: matrix = [[1,4,7,11],[2,5,8,12],[3,6,9,16],[10,13,14,17]], target = 5
Output: true

Example 2:
Input: matrix = [[1,4,7,11],[2,5,8,12],[3,6,9,16],[10,13,14,17]], target = 13
Output: true

Example 3:
Input: matrix = [[1,4,7,11],[2,5,8,12],[3,6,9,16],[10,13,14,17]], target = 20
Output: false

Constraints:
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 100
- -10^4 <= matrix[i][j], target <= 10^4
"""

def search_matrix_treat_as_1d(matrix, target):
    """
    Approach 1: Treat 2D Matrix as 1D Array
    Time Complexity: O(log(m * n))
    Space Complexity: O(1)
    
    Since matrix is sorted row by row, we can treat it as a single sorted array
    and use binary search with coordinate transformation.
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        # Convert 1D index to 2D coordinates
        row, col = divmod(mid, n)
        mid_val = matrix[row][col]
        
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False


def search_matrix_two_pass(matrix, target):
    """
    Approach 2: Two-Pass Binary Search
    Time Complexity: O(log m + log n)
    Space Complexity: O(1)
    
    First find the correct row, then search within that row.
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    
    # Find the correct row
    top, bottom = 0, m - 1
    while top <= bottom:
        mid_row = top + (bottom - top) // 2
        
        if matrix[mid_row][0] <= target <= matrix[mid_row][n - 1]:
            # Found the correct row, now search within it
            left, right = 0, n - 1
            while left <= right:
                mid_col = left + (right - left) // 2
                mid_val = matrix[mid_row][mid_col]
                
                if mid_val == target:
                    return True
                elif mid_val < target:
                    left = mid_col + 1
                else:
                    right = mid_col - 1
            return False
        elif matrix[mid_row][0] > target:
            bottom = mid_row - 1
        else:
            top = mid_row + 1
    
    return False


def search_matrix_staircase(matrix, target):
    """
    Approach 3: Staircase Search
    Time Complexity: O(m + n)
    Space Complexity: O(1)
    
    Start from top-right corner and move left or down based on comparison.
    This works because of the sorted property.
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    row, col = 0, n - 1
    
    while row < m and col >= 0:
        current = matrix[row][col]
        
        if current == target:
            return True
        elif current > target:
            col -= 1  # Move left
        else:
            row += 1  # Move down
    
    return False


def search_matrix_optimized_row_search(matrix, target):
    """
    Approach 4: Optimized Row Search
    Time Complexity: O(log m + log n)
    Space Complexity: O(1)
    
    Use binary search to find row more efficiently by checking last element.
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    
    # Binary search for the correct row
    left, right = 0, m - 1
    target_row = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if matrix[mid][0] <= target <= matrix[mid][n - 1]:
            target_row = mid
            break
        elif matrix[mid][0] > target:
            right = mid - 1
        else:
            left = mid + 1
    
    if target_row == -1:
        return False
    
    # Binary search within the row
    left, right = 0, n - 1
    while left <= right:
        mid = left + (right - left) // 2
        mid_val = matrix[target_row][mid]
        
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False


def test_search_matrix():
    """Test all approaches with various test cases."""
    
    test_cases = [
        # (matrix, target, expected)
        ([[1, 4, 7, 11], [2, 5, 8, 12], [3, 6, 9, 16], [10, 13, 14, 17]], 5, True),
        ([[1, 4, 7, 11], [2, 5, 8, 12], [3, 6, 9, 16], [10, 13, 14, 17]], 13, True),
        ([[1, 4, 7, 11], [2, 5, 8, 12], [3, 6, 9, 16], [10, 13, 14, 17]], 20, False),
        ([[1, 4, 7, 11], [2, 5, 8, 12], [3, 6, 9, 16], [10, 13, 14, 17]], 1, True),
        ([[1, 4, 7, 11], [2, 5, 8, 12], [3, 6, 9, 16], [10, 13, 14, 17]], 17, True),
        ([[1]], 1, True),
        ([[1]], 2, False),
        ([[1, 3, 5]], 3, True),
        ([[1], [3], [5]], 3, True),
        ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 6, True),
        ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 15, False),
    ]
    
    approaches = [
        ("Treat as 1D", search_matrix_treat_as_1d),
        ("Two-Pass Binary Search", search_matrix_two_pass),
        ("Staircase Search", search_matrix_staircase),
        ("Optimized Row Search", search_matrix_optimized_row_search),
    ]
    
    for i, (matrix, target, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Matrix: {matrix}")
        print(f"Target: {target}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(matrix, target)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_search_matrix() 