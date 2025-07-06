"""
238. Product of Array Except Self

Problem:
Given an integer array nums, return an array answer such that answer[i] is equal to the product
of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
You must write an algorithm that runs in O(n) time and without using the division operation.

Example 1:
Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Example 2:
Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]

Follow up: Can you solve the problem in O(1) extra space complexity? 
(The output array does not count as extra space for space complexity analysis.)

Time Complexity: O(n)
Space Complexity: O(1) excluding output array
"""


def product_except_self(nums):
    """
    Calculate product of array except self using left and right pass.
    
    Args:
        nums: List of integers
    
    Returns:
        List where each element is product of all others
    """
    if not nums:
        return []
    
    n = len(nums)
    result = [1] * n
    
    # First pass: calculate left products
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]
    
    # Second pass: multiply by right products
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result


def product_except_self_with_arrays(nums):
    """
    Calculate product using separate left and right arrays.
    
    Args:
        nums: List of integers
    
    Returns:
        List where each element is product of all others
    """
    if not nums:
        return []
    
    n = len(nums)
    left_products = [1] * n
    right_products = [1] * n
    result = [1] * n
    
    # Calculate left products
    for i in range(1, n):
        left_products[i] = left_products[i - 1] * nums[i - 1]
    
    # Calculate right products
    for i in range(n - 2, -1, -1):
        right_products[i] = right_products[i + 1] * nums[i + 1]
    
    # Calculate final result
    for i in range(n):
        result[i] = left_products[i] * right_products[i]
    
    return result


def product_except_self_division(nums):
    """
    Calculate product using division (not allowed in actual problem).
    Included for educational purposes only.
    
    Args:
        nums: List of integers
    
    Returns:
        List where each element is product of all others
    """
    if not nums:
        return []
    
    # Count zeros and calculate product of non-zero elements
    zero_count = nums.count(0)
    
    if zero_count > 1:
        # More than one zero means all results are zero
        return [0] * len(nums)
    elif zero_count == 1:
        # Exactly one zero
        total_product = 1
        for num in nums:
            if num != 0:
                total_product *= num
        
        result = []
        for num in nums:
            if num == 0:
                result.append(total_product)
            else:
                result.append(0)
        return result
    else:
        # No zeros
        total_product = 1
        for num in nums:
            total_product *= num
        
        return [total_product // num for num in nums]


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [1, 2, 3, 4]
    result1a = product_except_self(nums1)
    result1b = product_except_self_with_arrays(nums1)
    result1c = product_except_self_division(nums1)
    print(f"Test 1 - Expected: [24,12,8,6]")
    print(f"         Optimal: {result1a}")
    print(f"         Arrays: {result1b}")
    print(f"         Division: {result1c}")
    
    # Test case 2
    nums2 = [-1, 1, 0, -3, 3]
    result2a = product_except_self(nums2)
    result2b = product_except_self_with_arrays(nums2)
    result2c = product_except_self_division(nums2)
    print(f"Test 2 - Expected: [0,0,9,0,0]")
    print(f"         Optimal: {result2a}")
    print(f"         Arrays: {result2b}")
    print(f"         Division: {result2c}")
    
    # Test case 3 - Single element
    nums3 = [1]
    result3a = product_except_self(nums3)
    result3b = product_except_self_with_arrays(nums3)
    result3c = product_except_self_division(nums3)
    print(f"Test 3 - Expected: [1]")
    print(f"         Optimal: {result3a}")
    print(f"         Arrays: {result3b}")
    print(f"         Division: {result3c}")
    
    # Test case 4 - Two elements
    nums4 = [2, 3]
    result4a = product_except_self(nums4)
    result4b = product_except_self_with_arrays(nums4)
    result4c = product_except_self_division(nums4)
    print(f"Test 4 - Expected: [3,2]")
    print(f"         Optimal: {result4a}")
    print(f"         Arrays: {result4b}")
    print(f"         Division: {result4c}")
    
    # Test case 5 - Multiple zeros
    nums5 = [0, 0, 2, 3]
    result5a = product_except_self(nums5)
    result5b = product_except_self_with_arrays(nums5)
    result5c = product_except_self_division(nums5)
    print(f"Test 5 - Expected: [0,0,0,0]")
    print(f"         Optimal: {result5a}")
    print(f"         Arrays: {result5b}")
    print(f"         Division: {result5c}")
    
    # Test case 6 - Negative numbers
    nums6 = [-2, -3, 4, -5]
    result6a = product_except_self(nums6)
    result6b = product_except_self_with_arrays(nums6)
    result6c = product_except_self_division(nums6)
    print(f"Test 6 - Expected: [60,40,-30,24]")
    print(f"         Optimal: {result6a}")
    print(f"         Arrays: {result6b}")
    print(f"         Division: {result6c}")
    
    # Test case 7 - Empty array
    nums7 = []
    result7a = product_except_self(nums7)
    result7b = product_except_self_with_arrays(nums7)
    result7c = product_except_self_division(nums7)
    print(f"Test 7 - Expected: []")
    print(f"         Optimal: {result7a}")
    print(f"         Arrays: {result7b}")
    print(f"         Division: {result7c}") 