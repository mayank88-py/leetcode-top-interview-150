"""
42. Trapping Rain Water

Problem:
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it can trap after raining.

Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
In this case, 6 units of rain water (blue section) are being trapped.

Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9

Time Complexity: O(n) for optimal solution
Space Complexity: O(1) for two-pointer approach
"""


def trap(height):
    """
    Calculate trapped rainwater using two-pointer approach.
    
    Args:
        height: List of heights representing elevation map
    
    Returns:
        Amount of water that can be trapped
    """
    if not height or len(height) < 3:
        return 0
    
    left = 0
    right = len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water


def trap_dynamic_programming(height):
    """
    Calculate trapped rainwater using dynamic programming approach.
    
    Args:
        height: List of heights representing elevation map
    
    Returns:
        Amount of water that can be trapped
    """
    if not height or len(height) < 3:
        return 0
    
    n = len(height)
    
    # Calculate maximum height to the left of each position
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])
    
    # Calculate maximum height to the right of each position
    right_max = [0] * n
    right_max[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])
    
    # Calculate trapped water at each position
    water = 0
    for i in range(n):
        water_level = min(left_max[i], right_max[i])
        if water_level > height[i]:
            water += water_level - height[i]
    
    return water


def trap_stack(height):
    """
    Calculate trapped rainwater using stack approach.
    
    Args:
        height: List of heights representing elevation map
    
    Returns:
        Amount of water that can be trapped
    """
    if not height or len(height) < 3:
        return 0
    
    stack = []
    water = 0
    
    for i in range(len(height)):
        while stack and height[i] > height[stack[-1]]:
            top = stack.pop()
            
            if not stack:
                break
            
            # Calculate trapped water
            width = i - stack[-1] - 1
            bounded_height = min(height[i], height[stack[-1]]) - height[top]
            water += width * bounded_height
        
        stack.append(i)
    
    return water


def trap_brute_force(height):
    """
    Calculate trapped rainwater using brute force approach.
    
    Args:
        height: List of heights representing elevation map
    
    Returns:
        Amount of water that can be trapped
    """
    if not height or len(height) < 3:
        return 0
    
    n = len(height)
    water = 0
    
    for i in range(1, n-1):
        # Find maximum height to the left
        left_max = 0
        for j in range(i):
            left_max = max(left_max, height[j])
        
        # Find maximum height to the right
        right_max = 0
        for j in range(i+1, n):
            right_max = max(right_max, height[j])
        
        # Calculate water level at current position
        water_level = min(left_max, right_max)
        if water_level > height[i]:
            water += water_level - height[i]
    
    return water


def trap_divide_conquer(height):
    """
    Calculate trapped rainwater using divide and conquer approach.
    
    Args:
        height: List of heights representing elevation map
    
    Returns:
        Amount of water that can be trapped
    """
    if not height or len(height) < 3:
        return 0
    
    def find_max_index(arr, start, end):
        """Find index of maximum element in range"""
        max_idx = start
        for i in range(start + 1, end + 1):
            if arr[i] > arr[max_idx]:
                max_idx = i
        return max_idx
    
    def trap_helper(start, end):
        """Recursively calculate trapped water"""
        if start >= end:
            return 0
        
        # Find the highest bar in current range
        max_idx = find_max_index(height, start, end)
        
        water = 0
        
        # Calculate water on the left side
        current_max = 0
        for i in range(start, max_idx):
            current_max = max(current_max, height[i])
            water += max(0, current_max - height[i])
        
        # Calculate water on the right side
        current_max = 0
        for i in range(end, max_idx, -1):
            current_max = max(current_max, height[i])
            water += max(0, current_max - height[i])
        
        # Recursively solve for left and right parts
        water += trap_helper(start, max_idx - 1)
        water += trap_helper(max_idx + 1, end)
        
        return water
    
    return trap_helper(0, len(height) - 1)


def trap_segment_tree(height):
    """
    Calculate trapped rainwater using segment tree approach.
    
    Args:
        height: List of heights representing elevation map
    
    Returns:
        Amount of water that can be trapped
    """
    if not height or len(height) < 3:
        return 0
    
    n = len(height)
    
    # Build segment tree for range maximum queries
    class SegmentTree:
        def __init__(self, arr):
            self.n = len(arr)
            self.tree = [0] * (4 * self.n)
            self.build(arr, 0, 0, self.n - 1)
        
        def build(self, arr, node, start, end):
            if start == end:
                self.tree[node] = arr[start]
            else:
                mid = (start + end) // 2
                self.build(arr, 2*node+1, start, mid)
                self.build(arr, 2*node+2, mid+1, end)
                self.tree[node] = max(self.tree[2*node+1], self.tree[2*node+2])
        
        def query(self, node, start, end, l, r):
            if r < start or end < l:
                return 0
            if l <= start and end <= r:
                return self.tree[node]
            mid = (start + end) // 2
            left_max = self.query(2*node+1, start, mid, l, r)
            right_max = self.query(2*node+2, mid+1, end, l, r)
            return max(left_max, right_max)
        
        def range_max(self, l, r):
            return self.query(0, 0, self.n-1, l, r)
    
    st = SegmentTree(height)
    water = 0
    
    for i in range(1, n-1):
        left_max = st.range_max(0, i-1)
        right_max = st.range_max(i+1, n-1)
        water_level = min(left_max, right_max)
        if water_level > height[i]:
            water += water_level - height[i]
    
    return water


# Test cases
if __name__ == "__main__":
    # Test case 1
    height1 = [0,1,0,2,1,0,1,3,2,1,2,1]
    result1a = trap(height1)
    result1b = trap_dynamic_programming(height1)
    result1c = trap_stack(height1)
    result1d = trap_brute_force(height1)
    result1e = trap_divide_conquer(height1)
    result1f = trap_segment_tree(height1)
    print(f"Test 1 - Height: {height1}, Expected: 6")
    print(f"TwoPointer: {result1a}, DP: {result1b}, Stack: {result1c}, BruteForce: {result1d}, DivideConquer: {result1e}, SegmentTree: {result1f}")
    print()
    
    # Test case 2
    height2 = [4,2,0,3,2,5]
    result2a = trap(height2)
    result2b = trap_dynamic_programming(height2)
    result2c = trap_stack(height2)
    result2d = trap_brute_force(height2)
    result2e = trap_divide_conquer(height2)
    result2f = trap_segment_tree(height2)
    print(f"Test 2 - Height: {height2}, Expected: 9")
    print(f"TwoPointer: {result2a}, DP: {result2b}, Stack: {result2c}, BruteForce: {result2d}, DivideConquer: {result2e}, SegmentTree: {result2f}")
    print()
    
    # Test case 3 - No water can be trapped
    height3 = [1,2,3,4,5]
    result3a = trap(height3)
    result3b = trap_dynamic_programming(height3)
    result3c = trap_stack(height3)
    result3d = trap_brute_force(height3)
    result3e = trap_divide_conquer(height3)
    result3f = trap_segment_tree(height3)
    print(f"Test 3 - Height: {height3}, Expected: 0")
    print(f"TwoPointer: {result3a}, DP: {result3b}, Stack: {result3c}, BruteForce: {result3d}, DivideConquer: {result3e}, SegmentTree: {result3f}")
    print()
    
    # Test case 4 - Decreasing heights
    height4 = [5,4,3,2,1]
    result4a = trap(height4)
    result4b = trap_dynamic_programming(height4)
    result4c = trap_stack(height4)
    result4d = trap_brute_force(height4)
    result4e = trap_divide_conquer(height4)
    result4f = trap_segment_tree(height4)
    print(f"Test 4 - Height: {height4}, Expected: 0")
    print(f"TwoPointer: {result4a}, DP: {result4b}, Stack: {result4c}, BruteForce: {result4d}, DivideConquer: {result4e}, SegmentTree: {result4f}")
    print()
    
    # Test case 5 - Simple valley
    height5 = [3,0,2]
    result5a = trap(height5)
    result5b = trap_dynamic_programming(height5)
    result5c = trap_stack(height5)
    result5d = trap_brute_force(height5)
    result5e = trap_divide_conquer(height5)
    result5f = trap_segment_tree(height5)
    print(f"Test 5 - Height: {height5}, Expected: 2")
    print(f"TwoPointer: {result5a}, DP: {result5b}, Stack: {result5c}, BruteForce: {result5d}, DivideConquer: {result5e}, SegmentTree: {result5f}")
    print()
    
    # Test case 6 - Single element
    height6 = [5]
    result6a = trap(height6)
    result6b = trap_dynamic_programming(height6)
    result6c = trap_stack(height6)
    result6d = trap_brute_force(height6)
    result6e = trap_divide_conquer(height6)
    result6f = trap_segment_tree(height6)
    print(f"Test 6 - Height: {height6}, Expected: 0")
    print(f"TwoPointer: {result6a}, DP: {result6b}, Stack: {result6c}, BruteForce: {result6d}, DivideConquer: {result6e}, SegmentTree: {result6f}")
    print()
    
    # Test case 7 - Two elements
    height7 = [2,1]
    result7a = trap(height7)
    result7b = trap_dynamic_programming(height7)
    result7c = trap_stack(height7)
    result7d = trap_brute_force(height7)
    result7e = trap_divide_conquer(height7)
    result7f = trap_segment_tree(height7)
    print(f"Test 7 - Height: {height7}, Expected: 0")
    print(f"TwoPointer: {result7a}, DP: {result7b}, Stack: {result7c}, BruteForce: {result7d}, DivideConquer: {result7e}, SegmentTree: {result7f}")
    print()
    
    # Test case 8 - All zeros
    height8 = [0,0,0,0]
    result8a = trap(height8)
    result8b = trap_dynamic_programming(height8)
    result8c = trap_stack(height8)
    result8d = trap_brute_force(height8)
    result8e = trap_divide_conquer(height8)
    result8f = trap_segment_tree(height8)
    print(f"Test 8 - Height: {height8}, Expected: 0")
    print(f"TwoPointer: {result8a}, DP: {result8b}, Stack: {result8c}, BruteForce: {result8d}, DivideConquer: {result8e}, SegmentTree: {result8f}")
    print()
    
    # Test case 9 - Peak in middle
    height9 = [1,2,3,2,1]
    result9a = trap(height9)
    result9b = trap_dynamic_programming(height9)
    result9c = trap_stack(height9)
    result9d = trap_brute_force(height9)
    result9e = trap_divide_conquer(height9)
    result9f = trap_segment_tree(height9)
    print(f"Test 9 - Height: {height9}, Expected: 0")
    print(f"TwoPointer: {result9a}, DP: {result9b}, Stack: {result9c}, BruteForce: {result9d}, DivideConquer: {result9e}, SegmentTree: {result9f}")
    print()
    
    # Test case 10 - Complex pattern
    height10 = [2,1,2,1,2]
    result10a = trap(height10)
    result10b = trap_dynamic_programming(height10)
    result10c = trap_stack(height10)
    result10d = trap_brute_force(height10)
    result10e = trap_divide_conquer(height10)
    result10f = trap_segment_tree(height10)
    print(f"Test 10 - Height: {height10}, Expected: 2")
    print(f"TwoPointer: {result10a}, DP: {result10b}, Stack: {result10c}, BruteForce: {result10d}, DivideConquer: {result10e}, SegmentTree: {result10f}") 