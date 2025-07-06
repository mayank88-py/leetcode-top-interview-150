"""
215. Kth Largest Element in an Array

Problem:
Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

You must solve it in O(n) time complexity.

Example 1:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

Example 2:
Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4

Time Complexity: O(n) average case for quickselect
Space Complexity: O(log n) for recursion stack
"""


def find_kth_largest_quickselect(nums, k):
    """
    Quickselect algorithm - optimal average case.
    
    Time Complexity: O(n) average, O(n^2) worst case
    Space Complexity: O(log n) average for recursion stack
    
    Algorithm:
    1. Use quickselect (variant of quicksort)
    2. Partition array around pivot
    3. If pivot position equals target, return pivot
    4. Otherwise recurse on appropriate side
    5. Target position = len(nums) - k (for kth largest)
    """
    import random
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        # Move pivot to end
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        
        # Move pivot to final position
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index
    
    def quickselect(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        # Choose random pivot for better average performance
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return quickselect(left, pivot_index - 1, k_smallest)
        else:
            return quickselect(pivot_index + 1, right, k_smallest)
    
    # Convert kth largest to kth smallest
    return quickselect(0, len(nums) - 1, len(nums) - k)


def find_kth_largest_heap(nums, k):
    """
    Min heap approach.
    
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    
    Algorithm:
    1. Maintain a min heap of size k
    2. Add elements one by one
    3. If heap size > k, remove minimum
    4. Final heap top is kth largest element
    """
    import heapq
    
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]


def find_kth_largest_sort(nums, k):
    """
    Simple sorting approach.
    
    Time Complexity: O(n log n)
    Space Complexity: O(1) if in-place sort, O(n) otherwise
    
    Algorithm:
    1. Sort the array in descending order
    2. Return element at index k-1
    """
    nums.sort(reverse=True)
    return nums[k - 1]


def find_kth_largest_counting_sort(nums, k):
    """
    Counting sort approach (when range is limited).
    
    Time Complexity: O(n + range)
    Space Complexity: O(range)
    
    Algorithm:
    1. Count frequency of each number
    2. Iterate from largest to smallest
    3. Decrease k by frequency until k <= 0
    4. Return current number
    """
    min_val = min(nums)
    max_val = max(nums)
    
    # If range is too large, fall back to other method
    if max_val - min_val > 100000:
        return find_kth_largest_quickselect(nums, k)
    
    # Count frequencies
    count = [0] * (max_val - min_val + 1)
    for num in nums:
        count[num - min_val] += 1
    
    # Find kth largest
    for i in range(len(count) - 1, -1, -1):
        k -= count[i]
        if k <= 0:
            return i + min_val
    
    return -1  # Should never reach here


def find_kth_largest_iterative_quickselect(nums, k):
    """
    Iterative quickselect to avoid recursion stack.
    
    Time Complexity: O(n) average case
    Space Complexity: O(1)
    
    Algorithm:
    1. Same as recursive quickselect but iterative
    2. Use loop instead of recursion
    3. Update left/right bounds based on pivot position
    """
    import random
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index
    
    left, right = 0, len(nums) - 1
    k_smallest = len(nums) - k
    
    while left <= right:
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1
    
    return nums[k_smallest]


def find_kth_largest_median_of_medians(nums, k):
    """
    Deterministic quickselect using median of medians.
    
    Time Complexity: O(n) worst case guaranteed
    Space Complexity: O(log n)
    
    Algorithm:
    1. Use median-of-medians to choose good pivot
    2. Guarantees O(n) worst-case performance
    3. More complex but theoretically optimal
    """
    def median_of_medians(arr, left, right):
        if right - left < 5:
            return sorted(arr[left:right+1])[len(arr[left:right+1])//2]
        
        medians = []
        for i in range(left, right + 1, 5):
            group_right = min(i + 4, right)
            group = sorted(arr[i:group_right+1])
            medians.append(group[len(group)//2])
        
        return median_of_medians(medians, 0, len(medians) - 1)
    
    def partition(left, right, pivot_value):
        # Find pivot index
        pivot_index = -1
        for i in range(left, right + 1):
            if nums[i] == pivot_value:
                pivot_index = i
                break
        
        # Move pivot to end
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index
    
    def quickselect(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        pivot_value = median_of_medians(nums, left, right)
        pivot_index = partition(left, right, pivot_value)
        
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return quickselect(left, pivot_index - 1, k_smallest)
        else:
            return quickselect(pivot_index + 1, right, k_smallest)
    
    return quickselect(0, len(nums) - 1, len(nums) - k)


def test_kth_largest():
    """Test all implementations with various test cases."""
    
    test_cases = [
        ([3,2,1,5,6,4], 2, 5),
        ([3,2,3,1,2,4,5,5,6], 4, 4),
        ([1], 1, 1),
        ([1,2], 1, 2),
        ([1,2], 2, 1),
        ([7,10,4,3,20,15], 3, 10),
        ([2,1,3,3], 2, 3)
    ]
    
    implementations = [
        ("Quickselect", find_kth_largest_quickselect),
        ("Min Heap", find_kth_largest_heap),
        ("Sort", find_kth_largest_sort),
        ("Counting Sort", find_kth_largest_counting_sort),
        ("Iterative Quickselect", find_kth_largest_iterative_quickselect),
        ("Median of Medians", find_kth_largest_median_of_medians)
    ]
    
    print("Testing Kth Largest Element in Array...")
    
    for i, (nums, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}, k = {k}")
        print(f"Expected: {expected}")
        
        for impl_name, impl_func in implementations:
            # Make a copy since some implementations modify the array
            nums_copy = nums[:]
            result = impl_func(nums_copy, k)
            
            is_correct = result == expected
            print(f"{impl_name:20} | Result: {result} | {'✓' if is_correct else '✗'}")


if __name__ == "__main__":
    test_kth_largest() 