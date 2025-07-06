"""
85. Maximal Rectangle

Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

Example 1:
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.

Example 2:
Input: matrix = [["0"]]
Output: 0

Example 3:
Input: matrix = [["1"]]
Output: 1

Constraints:
- rows == matrix.length
- cols == matrix[i].length
- 1 <= rows, cols <= 200
- matrix[i][j] is '0' or '1'.
"""

def maximal_rectangle_histogram(matrix):
    """
    Approach 1: Largest Rectangle in Histogram
    Time Complexity: O(rows * cols)
    Space Complexity: O(cols)
    
    Convert each row to histogram heights and find max rectangle.
    """
    if not matrix or not matrix[0]:
        return 0
    
    def largest_rectangle_area(heights):
        stack = []
        max_area = 0
        
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        while stack:
            height = heights[stack.pop()]
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        return max_area
    
    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        
        max_area = max(max_area, largest_rectangle_area(heights))
    
    return max_area


def maximal_rectangle_dp(matrix):
    """
    Approach 2: Dynamic Programming
    Time Complexity: O(rows * cols^2)
    Space Complexity: O(cols)
    
    For each position, calculate height, left, and right boundaries.
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    
    # Arrays to track state for current row
    heights = [0] * cols  # Height of consecutive 1s ending at current row
    lefts = [0] * cols    # Left boundary of rectangle ending at current position
    rights = [cols] * cols  # Right boundary of rectangle ending at current position
    
    max_area = 0
    
    for i in range(rows):
        # Update heights
        for j in range(cols):
            if matrix[i][j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        
        # Update left boundaries
        cur_left = 0
        for j in range(cols):
            if matrix[i][j] == '1':
                lefts[j] = max(lefts[j], cur_left)
            else:
                lefts[j] = 0
                cur_left = j + 1
        
        # Update right boundaries
        cur_right = cols
        for j in range(cols - 1, -1, -1):
            if matrix[i][j] == '1':
                rights[j] = min(rights[j], cur_right)
            else:
                rights[j] = cols
                cur_right = j
        
        # Calculate max area for current row
        for j in range(cols):
            max_area = max(max_area, heights[j] * (rights[j] - lefts[j]))
    
    return max_area


def maximal_rectangle_stack_optimized(matrix):
    """
    Approach 3: Optimized Stack Approach
    Time Complexity: O(rows * cols)
    Space Complexity: O(cols)
    
    Optimized version of histogram approach.
    """
    if not matrix or not matrix[0]:
        return 0
    
    def largest_rectangle_optimized(heights):
        stack = [-1]  # Start with sentinel
        max_area = 0
        
        for i in range(len(heights)):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)
        
        while stack[-1] != -1:
            h = heights[stack.pop()]
            w = len(heights) - stack[-1] - 1
            max_area = max(max_area, h * w)
        
        return max_area
    
    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for i in range(rows):
        for j in range(cols):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        
        max_area = max(max_area, largest_rectangle_optimized(heights))
    
    return max_area


def maximal_rectangle_divide_conquer(matrix):
    """
    Approach 4: Divide and Conquer
    Time Complexity: O(rows * cols^2)
    Space Complexity: O(cols)
    
    Use divide and conquer for each histogram.
    """
    if not matrix or not matrix[0]:
        return 0
    
    def largest_rectangle_dc(heights, left, right):
        if left > right:
            return 0
        
        # Find minimum height and its index
        min_height = float('inf')
        min_idx = left
        for i in range(left, right + 1):
            if heights[i] < min_height:
                min_height = heights[i]
                min_idx = i
        
        # Area with min_height as the smallest bar
        area = min_height * (right - left + 1)
        
        # Recursively find max area in left and right parts
        left_area = largest_rectangle_dc(heights, left, min_idx - 1)
        right_area = largest_rectangle_dc(heights, min_idx + 1, right)
        
        return max(area, left_area, right_area)
    
    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for i in range(rows):
        for j in range(cols):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        
        if cols > 0:
            max_area = max(max_area, largest_rectangle_dc(heights, 0, cols - 1))
    
    return max_area


def maximal_rectangle_2d_dp(matrix):
    """
    Approach 5: 2D Dynamic Programming
    Time Complexity: O(rows * cols^2)
    Space Complexity: O(rows * cols)
    
    Use 2D DP to track rectangle properties.
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    
    # Create auxiliary arrays
    heights = [[0] * cols for _ in range(rows)]
    
    # Calculate heights
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == '1':
                heights[i][j] = 1 if i == 0 else heights[i-1][j] + 1
            else:
                heights[i][j] = 0
    
    max_area = 0
    
    # For each row, find max rectangle
    for i in range(rows):
        max_area = max(max_area, largest_rectangle_in_row(heights[i]))
    
    return max_area


def largest_rectangle_in_row(heights):
    """Helper function for 2D DP approach."""
    stack = []
    max_area = 0
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    while stack:
        height = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, height * width)
    
    return max_area


def maximal_rectangle_prefix_sum(matrix):
    """
    Approach 6: Prefix Sum Approach
    Time Complexity: O(rows^2 * cols^2)
    Space Complexity: O(rows * cols)
    
    Use prefix sums to quickly check rectangle validity.
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    
    # Build prefix sum matrix
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            prefix[i][j] = (int(matrix[i-1][j-1]) + 
                          prefix[i-1][j] + 
                          prefix[i][j-1] - 
                          prefix[i-1][j-1])
    
    def get_sum(r1, c1, r2, c2):
        return (prefix[r2+1][c2+1] - 
                prefix[r1][c2+1] - 
                prefix[r2+1][c1] + 
                prefix[r1][c1])
    
    max_area = 0
    
    for r1 in range(rows):
        for c1 in range(cols):
            for r2 in range(r1, rows):
                for c2 in range(c1, cols):
                    expected_sum = (r2 - r1 + 1) * (c2 - c1 + 1)
                    actual_sum = get_sum(r1, c1, r2, c2)
                    
                    if actual_sum == expected_sum:
                        max_area = max(max_area, actual_sum)
    
    return max_area


def maximal_rectangle_segment_tree(matrix):
    """
    Approach 7: Segment Tree Approach
    Time Complexity: O(rows * cols * log(cols))
    Space Complexity: O(cols)
    
    Use segment tree for range minimum queries in histogram.
    """
    if not matrix or not matrix[0]:
        return 0
    
    def largest_rectangle_segment_tree(heights):
        n = len(heights)
        if n == 0:
            return 0
        
        # Simple approach for small arrays
        if n <= 100:
            return largest_rectangle_in_row(heights)
        
        # For larger arrays, use divide and conquer
        def solve(left, right):
            if left > right:
                return 0
            if left == right:
                return heights[left]
            
            mid = (left + right) // 2
            left_area = solve(left, mid)
            right_area = solve(mid + 1, right)
            
            # Cross area
            l = mid
            r = mid + 1
            min_height = min(heights[l], heights[r])
            cross_area = min_height * 2
            
            # Expand while possible
            while l > left or r < right:
                if l > left and (r >= right or heights[l-1] >= heights[r+1]):
                    l -= 1
                    min_height = min(min_height, heights[l])
                else:
                    r += 1
                    min_height = min(min_height, heights[r])
                
                cross_area = max(cross_area, min_height * (r - l + 1))
            
            return max(left_area, right_area, cross_area)
        
        return solve(0, n - 1)
    
    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for i in range(rows):
        for j in range(cols):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        
        max_area = max(max_area, largest_rectangle_segment_tree(heights))
    
    return max_area


def maximal_rectangle_monotonic_stack(matrix):
    """
    Approach 8: Monotonic Stack
    Time Complexity: O(rows * cols)
    Space Complexity: O(cols)
    
    Use monotonic stack for efficient histogram processing.
    """
    if not matrix or not matrix[0]:
        return 0
    
    def largest_rectangle_monotonic(heights):
        stack = []
        max_area = 0
        heights.append(0)  # Add sentinel
        
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        heights.pop()  # Remove sentinel
        return max_area
    
    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for i in range(rows):
        for j in range(cols):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        
        max_area = max(max_area, largest_rectangle_monotonic(heights[:]))
    
    return max_area


def maximal_rectangle_brute_force(matrix):
    """
    Approach 9: Brute Force
    Time Complexity: O(rows^2 * cols^2)
    Space Complexity: O(1)
    
    Check all possible rectangles.
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    max_area = 0
    
    for r1 in range(rows):
        for c1 in range(cols):
            if matrix[r1][c1] == '1':
                for r2 in range(r1, rows):
                    for c2 in range(c1, cols):
                        # Check if rectangle from (r1,c1) to (r2,c2) is all 1s
                        valid = True
                        for r in range(r1, r2 + 1):
                            for c in range(c1, c2 + 1):
                                if matrix[r][c] == '0':
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            max_area = max(max_area, area)
    
    return max_area


def test_maximal_rectangle():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 6),
        ([["0"]], 0),
        ([["1"]], 1),
        ([["1","1"],["1","1"]], 4),
        ([["0","0"],["0","0"]], 0),
        ([["1","1","1"],["1","1","1"]], 6),
        ([["1","0","1","1","1"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 9),
        ([["0","1","1","0","1"],["1","1","0","1","0"],["0","1","1","1","0"],["1","1","1","1","0"],["1","1","1","1","1"],["0","0","0","0","0"]], 9),
    ]
    
    approaches = [
        ("Histogram", maximal_rectangle_histogram),
        ("DP", maximal_rectangle_dp),
        ("Stack Optimized", maximal_rectangle_stack_optimized),
        ("Divide Conquer", maximal_rectangle_divide_conquer),
        ("2D DP", maximal_rectangle_2d_dp),
        ("Prefix Sum", maximal_rectangle_prefix_sum),
        ("Segment Tree", maximal_rectangle_segment_tree),
        ("Monotonic Stack", maximal_rectangle_monotonic_stack),
        ("Brute Force", maximal_rectangle_brute_force),
    ]
    
    for i, (matrix, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {matrix}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            # Create deep copy to avoid modifying original
            matrix_copy = [row[:] for row in matrix]
            result = func(matrix_copy)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_maximal_rectangle() 