"""
215. Kth Largest Element in an Array

Given an integer array nums and an integer k, return the k-th largest element in the array.

Note that it is the k-th largest element in the sorted order, not the k-th distinct element.

Can you solve it without sorting?

Example 1:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

Example 2:
Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4

Constraints:
- 1 <= k <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
"""

def find_kth_largest_min_heap(nums, k):
    """
    Approach 1: Min Heap of Size K
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    
    Maintain a min heap of size k. The root will be the kth largest element.
    """
    # Implementing min heap manually
    class MinHeap:
        def __init__(self):
            self.heap = []
        
        def push(self, val):
            self.heap.append(val)
            self._heapify_up(len(self.heap) - 1)
        
        def pop(self):
            if not self.heap:
                return None
            
            # Swap first and last elements
            self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
            result = self.heap.pop()
            
            if self.heap:
                self._heapify_down(0)
            
            return result
        
        def peek(self):
            return self.heap[0] if self.heap else None
        
        def size(self):
            return len(self.heap)
        
        def _heapify_up(self, idx):
            parent = (idx - 1) // 2
            if parent >= 0 and self.heap[parent] > self.heap[idx]:
                self.heap[parent], self.heap[idx] = self.heap[idx], self.heap[parent]
                self._heapify_up(parent)
        
        def _heapify_down(self, idx):
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
                smallest = left
            
            if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
                smallest = right
            
            if smallest != idx:
                self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
                self._heapify_down(smallest)
    
    heap = MinHeap()
    
    for num in nums:
        heap.push(num)
        if heap.size() > k:
            heap.pop()
    
    return heap.peek()


def find_kth_largest_max_heap(nums, k):
    """
    Approach 2: Max Heap (Extract K Times)
    Time Complexity: O(n + k log n)
    Space Complexity: O(n)
    
    Build max heap and extract k times.
    """
    # Implementing max heap manually
    class MaxHeap:
        def __init__(self, arr):
            self.heap = arr[:]
            self._build_heap()
        
        def extract_max(self):
            if not self.heap:
                return None
            
            # Swap first and last elements
            self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
            result = self.heap.pop()
            
            if self.heap:
                self._heapify_down(0)
            
            return result
        
        def _build_heap(self):
            # Start from last non-leaf node
            for i in range(len(self.heap) // 2 - 1, -1, -1):
                self._heapify_down(i)
        
        def _heapify_down(self, idx):
            largest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < len(self.heap) and self.heap[left] > self.heap[largest]:
                largest = left
            
            if right < len(self.heap) and self.heap[right] > self.heap[largest]:
                largest = right
            
            if largest != idx:
                self.heap[idx], self.heap[largest] = self.heap[largest], self.heap[idx]
                self._heapify_down(largest)
    
    heap = MaxHeap(nums)
    
    for _ in range(k - 1):
        heap.extract_max()
    
    return heap.extract_max()


def find_kth_largest_quickselect(nums, k):
    """
    Approach 3: Quickselect Algorithm
    Time Complexity: O(n) average, O(n^2) worst case
    Space Complexity: O(1)
    
    Use quickselect to find kth largest element without fully sorting.
    """
    def quickselect(left, right, target_idx):
        if left == right:
            return nums[left]
        
        # Choose pivot and partition
        pivot_idx = partition(left, right)
        
        if pivot_idx == target_idx:
            return nums[pivot_idx]
        elif pivot_idx > target_idx:
            return quickselect(left, pivot_idx - 1, target_idx)
        else:
            return quickselect(pivot_idx + 1, right, target_idx)
    
    def partition(left, right):
        # Choose rightmost element as pivot
        pivot = nums[right]
        i = left
        
        for j in range(left, right):
            if nums[j] >= pivot:  # >= for descending order
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        
        nums[i], nums[right] = nums[right], nums[i]
        return i
    
    # Find kth largest = (k-1)th index in descending order
    return quickselect(0, len(nums) - 1, k - 1)


def find_kth_largest_sorting(nums, k):
    """
    Approach 4: Simple Sorting
    Time Complexity: O(n log n)
    Space Complexity: O(1) if in-place sorting
    
    Sort array and return kth largest element.
    """
    nums.sort(reverse=True)
    return nums[k - 1]


def find_kth_largest_counting_sort(nums, k):
    """
    Approach 5: Counting Sort (when range is limited)
    Time Complexity: O(n + range)
    Space Complexity: O(range)
    
    Use counting sort when the range of values is limited.
    """
    # Find range
    min_val, max_val = min(nums), max(nums)
    range_size = max_val - min_val + 1
    
    # Count frequencies
    count = [0] * range_size
    for num in nums:
        count[num - min_val] += 1
    
    # Find kth largest
    remaining = k
    for i in range(range_size - 1, -1, -1):
        remaining -= count[i]
        if remaining <= 0:
            return i + min_val
    
    return -1  # Should never reach here


def find_kth_largest_randomized_quickselect(nums, k):
    """
    Approach 6: Randomized Quickselect
    Time Complexity: O(n) average
    Space Complexity: O(1)
    
    Use randomized pivot selection for better average performance.
    """
    import random
    
    def quickselect(left, right, target_idx):
        if left == right:
            return nums[left]
        
        # Randomize pivot selection
        random_idx = random.randint(left, right)
        nums[random_idx], nums[right] = nums[right], nums[random_idx]
        
        pivot_idx = partition(left, right)
        
        if pivot_idx == target_idx:
            return nums[pivot_idx]
        elif pivot_idx > target_idx:
            return quickselect(left, pivot_idx - 1, target_idx)
        else:
            return quickselect(pivot_idx + 1, right, target_idx)
    
    def partition(left, right):
        pivot = nums[right]
        i = left
        
        for j in range(left, right):
            if nums[j] >= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        
        nums[i], nums[right] = nums[right], nums[i]
        return i
    
    return quickselect(0, len(nums) - 1, k - 1)


def test_find_kth_largest():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([3, 2, 1, 5, 6, 4], 2, 5),
        ([3, 2, 3, 1, 2, 4, 5, 5, 6], 4, 4),
        ([1], 1, 1),
        ([1, 2], 1, 2),
        ([1, 2], 2, 1),
        ([7, 10, 4, 3, 20, 15], 3, 10),
        ([1, 2, 3, 4, 5], 1, 5),
        ([1, 2, 3, 4, 5], 5, 1),
        ([2, 1, 3, 5, 6, 4], 2, 5),
        ([3, 2, 1, 5, 6, 4], 4, 3),
    ]
    
    approaches = [
        ("Min Heap", find_kth_largest_min_heap),
        ("Max Heap", find_kth_largest_max_heap),
        ("Quickselect", find_kth_largest_quickselect),
        ("Sorting", find_kth_largest_sorting),
        ("Counting Sort", find_kth_largest_counting_sort),
        ("Randomized Quickselect", find_kth_largest_randomized_quickselect),
    ]
    
    for i, (nums, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: nums = {nums}, k = {k}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            # Create a copy for functions that modify the array
            nums_copy = nums.copy()
            result = func(nums_copy, k)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_find_kth_largest() 