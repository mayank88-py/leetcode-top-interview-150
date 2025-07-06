"""
11. Container With Most Water

Problem:
You are given an integer array height of length n. There are n vertical lines drawn such that
the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

Example 1:
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7].
In this case, the max area of water (blue section) the container can contain is 49.

Example 2:
Input: height = [1,1]
Output: 1

Time Complexity: O(n)
Space Complexity: O(1)
"""


def max_area(height):
    """
    Find maximum area using two pointers approach.
    
    Args:
        height: List of heights
    
    Returns:
        Maximum area that can be contained
    """
    if not height or len(height) < 2:
        return 0
    
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        # Calculate current area
        width = right - left
        current_area = min(height[left], height[right]) * width
        max_area = max(max_area, current_area)
        
        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area


def max_area_brute_force(height):
    """
    Find maximum area using brute force approach.
    
    Args:
        height: List of heights
    
    Returns:
        Maximum area that can be contained
    """
    if not height or len(height) < 2:
        return 0
    
    max_area = 0
    
    for i in range(len(height)):
        for j in range(i + 1, len(height)):
            width = j - i
            current_area = min(height[i], height[j]) * width
            max_area = max(max_area, current_area)
    
    return max_area


# Test cases
if __name__ == "__main__":
    # Test case 1
    height1 = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    result1a = max_area(height1)
    result1b = max_area_brute_force(height1)
    print(f"Test 1 - Expected: 49, Two Pointers: {result1a}, Brute Force: {result1b}")
    
    # Test case 2
    height2 = [1, 1]
    result2a = max_area(height2)
    result2b = max_area_brute_force(height2)
    print(f"Test 2 - Expected: 1, Two Pointers: {result2a}, Brute Force: {result2b}")
    
    # Test case 3
    height3 = [4, 3, 2, 1, 4]
    result3a = max_area(height3)
    result3b = max_area_brute_force(height3)
    print(f"Test 3 - Expected: 16, Two Pointers: {result3a}, Brute Force: {result3b}")
    
    # Test case 4
    height4 = [1, 2, 1]
    result4a = max_area(height4)
    result4b = max_area_brute_force(height4)
    print(f"Test 4 - Expected: 2, Two Pointers: {result4a}, Brute Force: {result4b}")
    
    # Test case 5
    height5 = [1, 2, 4, 3]
    result5a = max_area(height5)
    result5b = max_area_brute_force(height5)
    print(f"Test 5 - Expected: 4, Two Pointers: {result5a}, Brute Force: {result5b}")
    
    # Test case 6
    height6 = [2, 1]
    result6a = max_area(height6)
    result6b = max_area_brute_force(height6)
    print(f"Test 6 - Expected: 1, Two Pointers: {result6a}, Brute Force: {result6b}")
    
    # Test case 7
    height7 = [1, 3, 2, 5, 25, 24, 5]
    result7a = max_area(height7)
    result7b = max_area_brute_force(height7)
    print(f"Test 7 - Expected: 24, Two Pointers: {result7a}, Brute Force: {result7b}") 