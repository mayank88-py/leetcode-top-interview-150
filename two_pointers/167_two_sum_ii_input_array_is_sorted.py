"""
167. Two Sum II - Input Array Is Sorted

Problem:
Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order,
find two numbers such that they add up to a specific target number.
Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the same element twice.

Example 1:
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].

Example 2:
Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 4 is 6. Therefore, index1 = 1, index2 = 3. We return [1, 3].

Example 3:
Input: numbers = [-1,0], target = -1
Output: [1,2]
Explanation: The sum of -1 and 0 is -1. Therefore, index1 = 1, index2 = 2. We return [1, 2].

Time Complexity: O(n)
Space Complexity: O(1)
"""


def two_sum(numbers, target):
    """
    Find two numbers that add up to target using two pointers.
    
    Args:
        numbers: Sorted array of integers
        target: Target sum
    
    Returns:
        List of two indices (1-indexed) where numbers sum to target
    """
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            return [left + 1, right + 1]  # Convert to 1-indexed
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []  # No solution found (shouldn't happen according to problem)


def two_sum_binary_search(numbers, target):
    """
    Find two numbers that add up to target using binary search.
    
    Args:
        numbers: Sorted array of integers
        target: Target sum
    
    Returns:
        List of two indices (1-indexed) where numbers sum to target
    """
    for i in range(len(numbers)):
        complement = target - numbers[i]
        
        # Binary search for complement in the remaining array
        left, right = i + 1, len(numbers) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if numbers[mid] == complement:
                return [i + 1, mid + 1]  # Convert to 1-indexed
            elif numbers[mid] < complement:
                left = mid + 1
            else:
                right = mid - 1
    
    return []  # No solution found


def two_sum_hashmap(numbers, target):
    """
    Find two numbers that add up to target using hashmap.
    
    Args:
        numbers: Sorted array of integers
        target: Target sum
    
    Returns:
        List of two indices (1-indexed) where numbers sum to target
    """
    num_to_index = {}
    
    for i, num in enumerate(numbers):
        complement = target - num
        
        if complement in num_to_index:
            return [num_to_index[complement] + 1, i + 1]  # Convert to 1-indexed
        
        num_to_index[num] = i
    
    return []  # No solution found


# Test cases
if __name__ == "__main__":
    # Test case 1
    numbers1 = [2, 7, 11, 15]
    target1 = 9
    result1a = two_sum(numbers1, target1)
    result1b = two_sum_binary_search(numbers1, target1)
    result1c = two_sum_hashmap(numbers1, target1)
    print(f"Test 1 - Expected: [1,2], Two Pointers: {result1a}, Binary Search: {result1b}, HashMap: {result1c}")
    
    # Test case 2
    numbers2 = [2, 3, 4]
    target2 = 6
    result2a = two_sum(numbers2, target2)
    result2b = two_sum_binary_search(numbers2, target2)
    result2c = two_sum_hashmap(numbers2, target2)
    print(f"Test 2 - Expected: [1,3], Two Pointers: {result2a}, Binary Search: {result2b}, HashMap: {result2c}")
    
    # Test case 3
    numbers3 = [-1, 0]
    target3 = -1
    result3a = two_sum(numbers3, target3)
    result3b = two_sum_binary_search(numbers3, target3)
    result3c = two_sum_hashmap(numbers3, target3)
    print(f"Test 3 - Expected: [1,2], Two Pointers: {result3a}, Binary Search: {result3b}, HashMap: {result3c}")
    
    # Test case 4
    numbers4 = [1, 2, 3, 4, 4, 9, 56, 90]
    target4 = 8
    result4a = two_sum(numbers4, target4)
    result4b = two_sum_binary_search(numbers4, target4)
    result4c = two_sum_hashmap(numbers4, target4)
    print(f"Test 4 - Expected: [4,5], Two Pointers: {result4a}, Binary Search: {result4b}, HashMap: {result4c}")
    
    # Test case 5
    numbers5 = [5, 25, 75]
    target5 = 100
    result5a = two_sum(numbers5, target5)
    result5b = two_sum_binary_search(numbers5, target5)
    result5c = two_sum_hashmap(numbers5, target5)
    print(f"Test 5 - Expected: [2,3], Two Pointers: {result5a}, Binary Search: {result5b}, HashMap: {result5c}") 