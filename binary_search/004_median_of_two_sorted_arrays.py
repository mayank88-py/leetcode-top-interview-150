"""
4. Median of Two Sorted Arrays

Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).

Example 1:
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.

Example 2:
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.

Constraints:
- nums1.length == m
- nums2.length == n
- 0 <= m <= 1000
- 0 <= n <= 1000
- 1 <= m + n <= 2000
- -10^6 <= nums1[i], nums2[i] <= 10^6
"""

def find_median_sorted_arrays_binary_search(nums1, nums2):
    """
    Approach 1: Binary Search on Smaller Array
    Time Complexity: O(log(min(m, n)))
    Space Complexity: O(1)
    
    Use binary search on the smaller array to find the correct partition.
    """
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    total = m + n
    half = total // 2
    
    left, right = 0, m
    
    while left <= right:
        # Partition points
        partition1 = (left + right) // 2
        partition2 = half - partition1
        
        # Elements on the left side of partition
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        
        # Elements on the right side of partition
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        # Check if we found the correct partition
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found the correct partition
            if total % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            else:
                return float(min(min_right1, min_right2))
        
        elif max_left1 > min_right2:
            # We have too many elements from nums1, move left
            right = partition1 - 1
        else:
            # We have too few elements from nums1, move right
            left = partition1 + 1
    
    raise ValueError("Input arrays are not sorted")


def find_median_sorted_arrays_merge(nums1, nums2):
    """
    Approach 2: Merge Until Median
    Time Complexity: O((m + n) / 2) = O(m + n)
    Space Complexity: O(1)
    
    Merge arrays until we reach the median position(s).
    """
    m, n = len(nums1), len(nums2)
    total = m + n
    target = total // 2
    
    i = j = 0
    prev = curr = 0
    
    for count in range(target + 1):
        prev = curr
        
        if i < m and j < n:
            if nums1[i] <= nums2[j]:
                curr = nums1[i]
                i += 1
            else:
                curr = nums2[j]
                j += 1
        elif i < m:
            curr = nums1[i]
            i += 1
        else:
            curr = nums2[j]
            j += 1
    
    if total % 2 == 0:
        return (prev + curr) / 2.0
    else:
        return float(curr)


def find_median_sorted_arrays_recursive(nums1, nums2):
    """
    Approach 3: Recursive Binary Search
    Time Complexity: O(log(min(m, n)))
    Space Complexity: O(log(min(m, n))) - recursion depth
    
    Recursive implementation of binary search approach.
    """
    def find_kth_element(nums1, nums2, k):
        """Find the k-th smallest element in merged arrays (1-indexed)."""
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        if not nums1:
            return nums2[k - 1]
        
        if k == 1:
            return min(nums1[0], nums2[0])
        
        # Binary search
        i = min(k // 2, len(nums1))
        j = k - i
        
        if j > len(nums2):
            j = len(nums2)
            i = k - j
        
        if nums1[i - 1] < nums2[j - 1]:
            return find_kth_element(nums1[i:], nums2, k - i)
        else:
            return find_kth_element(nums1, nums2[j:], k - j)
    
    total = len(nums1) + len(nums2)
    
    if total % 2 == 1:
        return float(find_kth_element(nums1, nums2, total // 2 + 1))
    else:
        left = find_kth_element(nums1, nums2, total // 2)
        right = find_kth_element(nums1, nums2, total // 2 + 1)
        return (left + right) / 2.0


def find_median_sorted_arrays_full_merge(nums1, nums2):
    """
    Approach 4: Full Merge (Not Optimal)
    Time Complexity: O(m + n)
    Space Complexity: O(m + n)
    
    Merge both arrays completely and find median.
    """
    merged = []
    i = j = 0
    
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            merged.append(nums1[i])
            i += 1
        else:
            merged.append(nums2[j])
            j += 1
    
    # Add remaining elements
    merged.extend(nums1[i:])
    merged.extend(nums2[j:])
    
    n = len(merged)
    if n % 2 == 0:
        return (merged[n // 2 - 1] + merged[n // 2]) / 2.0
    else:
        return float(merged[n // 2])


def find_median_sorted_arrays_optimized(nums1, nums2):
    """
    Approach 5: Optimized Binary Search
    Time Complexity: O(log(min(m, n)))
    Space Complexity: O(1)
    
    Optimized version with better handling of edge cases.
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    
    # Handle empty arrays
    if m == 0:
        if n % 2 == 0:
            return (nums2[n // 2 - 1] + nums2[n // 2]) / 2.0
        else:
            return float(nums2[n // 2])
    
    total = m + n
    half = (total + 1) // 2  # This handles both odd and even cases
    
    low, high = 0, m
    
    while low <= high:
        cut1 = (low + high) // 2
        cut2 = half - cut1
        
        left1 = float('-inf') if cut1 == 0 else nums1[cut1 - 1]
        left2 = float('-inf') if cut2 == 0 else nums2[cut2 - 1]
        
        right1 = float('inf') if cut1 == m else nums1[cut1]
        right2 = float('inf') if cut2 == n else nums2[cut2]
        
        if left1 <= right2 and left2 <= right1:
            if total % 2 == 0:
                return (max(left1, left2) + min(right1, right2)) / 2.0
            else:
                return float(max(left1, left2))
        elif left1 > right2:
            high = cut1 - 1
        else:
            low = cut1 + 1
    
    return 1.0  # Should never reach here


def test_find_median_sorted_arrays():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([1, 3], [2], 2.0),
        ([1, 2], [3, 4], 2.5),
        ([0, 0], [0, 0], 0.0),
        ([], [1], 1.0),
        ([2], [], 2.0),
        ([1, 2, 3], [4, 5, 6], 3.5),
        ([1], [2, 3, 4], 2.5),
        ([1, 3, 5], [2, 4, 6], 3.5),
        ([1, 2], [1, 2], 1.5),
        ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], 5.5),
        ([1, 3, 8, 9, 15], [7, 11, 19, 21, 25], 9.0),
    ]
    
    approaches = [
        ("Binary Search", find_median_sorted_arrays_binary_search),
        ("Merge Until Median", find_median_sorted_arrays_merge),
        ("Recursive Binary Search", find_median_sorted_arrays_recursive),
        ("Full Merge", find_median_sorted_arrays_full_merge),
        ("Optimized Binary Search", find_median_sorted_arrays_optimized),
    ]
    
    for i, (nums1, nums2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: nums1 = {nums1}, nums2 = {nums2}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(nums1, nums2)
            status = "✓" if abs(result - expected) < 1e-5 else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_find_median_sorted_arrays() 