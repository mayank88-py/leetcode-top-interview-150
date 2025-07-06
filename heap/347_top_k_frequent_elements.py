"""
347. Top K Frequent Elements

Given an integer array nums and an integer k, return the k most frequent elements. 
You may return the answer in any order.

Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:
Input: nums = [1], k = 1
Output: [1]

Constraints:
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
- k is in the range [1, the number of unique elements in the array].
- It is guaranteed that the answer is unique.

Follow up: Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
"""

def top_k_frequent_min_heap(nums, k):
    """
    Approach 1: Min Heap of Size K
    Time Complexity: O(n log k)
    Space Complexity: O(n + k)
    
    Use frequency map and min heap to keep top k frequent elements.
    """
    # Count frequencies
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1
    
    # Manual min heap implementation (based on frequency)
    class MinHeap:
        def __init__(self):
            self.heap = []  # [(frequency, number), ...]
        
        def push(self, item):
            self.heap.append(item)
            self._heapify_up(len(self.heap) - 1)
        
        def pop(self):
            if not self.heap:
                return None
            
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
            if parent >= 0 and self.heap[parent][0] > self.heap[idx][0]:
                self.heap[parent], self.heap[idx] = self.heap[idx], self.heap[parent]
                self._heapify_up(parent)
        
        def _heapify_down(self, idx):
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < len(self.heap) and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            
            if right < len(self.heap) and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            
            if smallest != idx:
                self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
                self._heapify_down(smallest)
    
    heap = MinHeap()
    
    for num, freq in freq_map.items():
        heap.push((freq, num))
        if heap.size() > k:
            heap.pop()
    
    # Extract results
    result = []
    while heap.size() > 0:
        _, num = heap.pop()
        result.append(num)
    
    return result


def top_k_frequent_max_heap(nums, k):
    """
    Approach 2: Max Heap (Extract K Times)
    Time Complexity: O(n + m log m) where m is unique elements
    Space Complexity: O(n)
    
    Build max heap of all elements by frequency and extract k times.
    """
    # Count frequencies
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1
    
    # Manual max heap implementation
    class MaxHeap:
        def __init__(self, items):
            self.heap = items[:]  # [(frequency, number), ...]
            self._build_heap()
        
        def extract_max(self):
            if not self.heap:
                return None
            
            self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
            result = self.heap.pop()
            
            if self.heap:
                self._heapify_down(0)
            
            return result
        
        def _build_heap(self):
            for i in range(len(self.heap) // 2 - 1, -1, -1):
                self._heapify_down(i)
        
        def _heapify_down(self, idx):
            largest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < len(self.heap) and self.heap[left][0] > self.heap[largest][0]:
                largest = left
            
            if right < len(self.heap) and self.heap[right][0] > self.heap[largest][0]:
                largest = right
            
            if largest != idx:
                self.heap[idx], self.heap[largest] = self.heap[largest], self.heap[idx]
                self._heapify_down(largest)
    
    # Create heap with frequency-number pairs
    heap_items = [(freq, num) for num, freq in freq_map.items()]
    heap = MaxHeap(heap_items)
    
    # Extract k most frequent
    result = []
    for _ in range(k):
        _, num = heap.extract_max()
        result.append(num)
    
    return result


def top_k_frequent_bucket_sort(nums, k):
    """
    Approach 3: Bucket Sort
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Use bucket sort where index represents frequency.
    """
    # Count frequencies
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1
    
    # Create buckets for each frequency
    # Maximum frequency can be len(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in freq_map.items():
        buckets[freq].append(num)
    
    # Collect results from highest frequency buckets
    result = []
    for freq in range(len(buckets) - 1, 0, -1):
        for num in buckets[freq]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result


def top_k_frequent_sorting(nums, k):
    """
    Approach 4: Sorting by Frequency
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Sort elements by frequency and return top k.
    """
    # Count frequencies
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1
    
    # Sort by frequency (descending)
    sorted_items = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
    
    return [num for num, _ in sorted_items[:k]]


def top_k_frequent_quickselect(nums, k):
    """
    Approach 5: Quickselect on Frequencies
    Time Complexity: O(n) average
    Space Complexity: O(n)
    
    Use quickselect to find kth most frequent element.
    """
    # Count frequencies
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1
    
    unique_nums = list(freq_map.keys())
    
    def quickselect(left, right, target_idx):
        if left == right:
            return
        
        pivot_idx = partition(left, right)
        
        if pivot_idx == target_idx:
            return
        elif pivot_idx > target_idx:
            quickselect(left, pivot_idx - 1, target_idx)
        else:
            quickselect(pivot_idx + 1, right, target_idx)
    
    def partition(left, right):
        # Choose rightmost element as pivot
        pivot_freq = freq_map[unique_nums[right]]
        i = left
        
        for j in range(left, right):
            # Sort in descending order of frequency
            if freq_map[unique_nums[j]] >= pivot_freq:
                unique_nums[i], unique_nums[j] = unique_nums[j], unique_nums[i]
                i += 1
        
        unique_nums[i], unique_nums[right] = unique_nums[right], unique_nums[i]
        return i
    
    # Find k most frequent elements
    quickselect(0, len(unique_nums) - 1, k - 1)
    
    return unique_nums[:k]


def top_k_frequent_counter_optimized(nums, k):
    """
    Approach 6: Optimized with Early Termination
    Time Complexity: O(n log k)
    Space Complexity: O(n)
    
    Optimized approach with early termination for large arrays.
    """
    if k == len(set(nums)):
        return list(set(nums))
    
    # Count frequencies
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1
    
    # If we need all unique elements
    if k >= len(freq_map):
        return list(freq_map.keys())
    
    # Use bucket sort for small k, heap for large k
    if k <= 3:
        return top_k_frequent_bucket_sort(nums, k)
    else:
        return top_k_frequent_min_heap(nums, k)


def test_top_k_frequent():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([1, 1, 1, 2, 2, 3], 2, {1, 2}),
        ([1], 1, {1}),
        ([1, 2], 2, {1, 2}),
        ([1, 1, 1, 1], 1, {1}),
        ([1, 2, 3, 1, 2, 3, 1, 2, 1], 2, {1, 2}),
        ([4, 1, -1, 2, -1, 2, 3], 2, {-1, 2}),
        ([1, 1, 2, 2, 3, 3], 3, {1, 2, 3}),
        ([5, 3, 1, 1, 1, 3, 73, 1], 2, {1, 3}),
        ([1, 2, 3, 4, 5], 3, {1, 2, 3}),  # All same frequency
        ([7, 7], 1, {7}),
    ]
    
    approaches = [
        ("Min Heap", top_k_frequent_min_heap),
        ("Max Heap", top_k_frequent_max_heap),
        ("Bucket Sort", top_k_frequent_bucket_sort),
        ("Sorting", top_k_frequent_sorting),
        ("Quickselect", top_k_frequent_quickselect),
        ("Optimized", top_k_frequent_counter_optimized),
    ]
    
    for i, (nums, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: nums = {nums}, k = {k}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(nums, k)
            result_set = set(result)
            status = "✓" if result_set == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_top_k_frequent() 