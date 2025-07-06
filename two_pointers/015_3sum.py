"""
15. 3Sum

Problem:
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that
i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].

Example 2:
Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.

Example 3:
Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0.

Time Complexity: O(n^2)
Space Complexity: O(1) excluding the result array
"""


def three_sum(nums):
    """
    Find all unique triplets that sum to zero.
    
    Args:
        nums: List of integers
    
    Returns:
        List of triplets that sum to zero
    """
    if not nums or len(nums) < 3:
        return []
    
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for the first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates for the second element
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                
                # Skip duplicates for the third element
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    
    return result


def three_sum_brute_force(nums):
    """
    Find all unique triplets that sum to zero using brute force.
    
    Args:
        nums: List of integers
    
    Returns:
        List of triplets that sum to zero
    """
    if not nums or len(nums) < 3:
        return []
    
    result = []
    
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                if nums[i] + nums[j] + nums[k] == 0:
                    triplet = sorted([nums[i], nums[j], nums[k]])
                    if triplet not in result:
                        result.append(triplet)
    
    return result


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [-1, 0, 1, 2, -1, -4]
    result1a = three_sum(nums1)
    result1b = three_sum_brute_force(nums1)
    print(f"Test 1 - Expected: [[-1,-1,2],[-1,0,1]], Two Pointers: {result1a}")
    print(f"         Brute Force: {result1b}")
    
    # Test case 2
    nums2 = [0, 1, 1]
    result2a = three_sum(nums2)
    result2b = three_sum_brute_force(nums2)
    print(f"Test 2 - Expected: [], Two Pointers: {result2a}, Brute Force: {result2b}")
    
    # Test case 3
    nums3 = [0, 0, 0]
    result3a = three_sum(nums3)
    result3b = three_sum_brute_force(nums3)
    print(f"Test 3 - Expected: [[0,0,0]], Two Pointers: {result3a}, Brute Force: {result3b}")
    
    # Test case 4
    nums4 = [-2, 0, 1, 1, 2]
    result4a = three_sum(nums4)
    result4b = three_sum_brute_force(nums4)
    print(f"Test 4 - Expected: [[-2,0,2],[-2,1,1]], Two Pointers: {result4a}")
    print(f"         Brute Force: {result4b}")
    
    # Test case 5
    nums5 = [-1, 0, 1, 0]
    result5a = three_sum(nums5)
    result5b = three_sum_brute_force(nums5)
    print(f"Test 5 - Expected: [[-1,0,1]], Two Pointers: {result5a}, Brute Force: {result5b}")
    
    # Test case 6
    nums6 = [1, 2, -2, -1]
    result6a = three_sum(nums6)
    result6b = three_sum_brute_force(nums6)
    print(f"Test 6 - Expected: [], Two Pointers: {result6a}, Brute Force: {result6b}")
    
    # Test case 7
    nums7 = [3, 0, -2, -1, 1, 2]
    result7a = three_sum(nums7)
    result7b = three_sum_brute_force(nums7)
    print(f"Test 7 - Expected: [[-2,-1,3],[-2,0,2],[-1,0,1]], Two Pointers: {result7a}")
    print(f"         Brute Force: {result7b}")
    
    # Test case 8 - Empty array
    nums8 = []
    result8a = three_sum(nums8)
    result8b = three_sum_brute_force(nums8)
    print(f"Test 8 - Expected: [], Two Pointers: {result8a}, Brute Force: {result8b}")
    
    # Test case 9
    nums9 = [1, 2]
    result9a = three_sum(nums9)
    result9b = three_sum_brute_force(nums9)
    print(f"Test 9 - Expected: [], Two Pointers: {result9a}, Brute Force: {result9b}") 