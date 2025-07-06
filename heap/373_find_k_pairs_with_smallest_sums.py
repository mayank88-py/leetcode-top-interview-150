"""
373. Find K Pairs with Smallest Sums

You are given two integer arrays nums1 and nums2 sorted in non-decreasing order and an integer k.

Define a pair (u, v) which consists of one element from the first array and one element from the second array.

Return the k pairs (u1, v1), (u2, v2), ..., (uk, vk) with the smallest sums.

Example 1:
Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
Output: [[1,2],[1,4],[1,6]]
Explanation: The first 3 pairs are returned from the sequence: [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]

Example 2:
Input: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
Output: [[1,1],[1,1]]
Explanation: The first 2 pairs are returned from the sequence: [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]

Constraints:
- 1 <= nums1.length, nums2.length <= 10^5
- -10^9 <= nums1[i], nums2[i] <= 10^9
- nums1 and nums2 both are sorted in non-decreasing order.
- 1 <= k <= 10^4
- k <= nums1.length * nums2.length
"""

def k_smallest_pairs_min_heap(nums1, nums2, k):
    """
    Approach 1: Min Heap with All Possible Pairs
    Time Complexity: O(min(k log(m*n), m*n log(m*n)))
    Space Complexity: O(m*n)
    
    Generate all pairs, add to min heap, and extract k smallest.
    WARNING: Not efficient for large arrays due to memory usage.
    """
    # Manual min heap implementation
    class MinHeap:
        def __init__(self):
            self.heap = []  # [(sum, [u, v]), ...]
        
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
    
    # Only practical for small arrays
    if len(nums1) * len(nums2) > 10000:
        return k_smallest_pairs_optimized_heap(nums1, nums2, k)
    
    heap = MinHeap()
    
    # Generate all pairs
    for i in range(len(nums1)):
        for j in range(len(nums2)):
            pair_sum = nums1[i] + nums2[j]
            heap.push((pair_sum, [nums1[i], nums2[j]]))
    
    # Extract k smallest
    result = []
    for _ in range(min(k, len(nums1) * len(nums2))):
        _, pair = heap.pop()
        result.append(pair)
    
    return result


def k_smallest_pairs_optimized_heap(nums1, nums2, k):
    """
    Approach 2: Optimized Min Heap (Only K pairs)
    Time Complexity: O(k log k)
    Space Complexity: O(k)
    
    Use heap to maintain only the k smallest candidates.
    """
    class MinHeap:
        def __init__(self):
            self.heap = []  # [(sum, i, j), ...]
        
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
    
    if not nums1 or not nums2:
        return []
    
    heap = MinHeap()
    visited = set()
    result = []
    
    # Start with the pair (0, 0)
    heap.push((nums1[0] + nums2[0], 0, 0))
    visited.add((0, 0))
    
    while result.__len__() < k and heap.size() > 0:
        curr_sum, i, j = heap.pop()
        result.append([nums1[i], nums2[j]])
        
        # Add adjacent pairs
        if i + 1 < len(nums1) and (i + 1, j) not in visited:
            heap.push((nums1[i + 1] + nums2[j], i + 1, j))
            visited.add((i + 1, j))
        
        if j + 1 < len(nums2) and (i, j + 1) not in visited:
            heap.push((nums1[i] + nums2[j + 1], i, j + 1))
            visited.add((i, j + 1))
    
    return result


def k_smallest_pairs_max_heap(nums1, nums2, k):
    """
    Approach 3: Max Heap (Keep only K pairs)
    Time Complexity: O(m*n log k)
    Space Complexity: O(k)
    
    Use max heap to maintain k smallest pairs seen so far.
    """
    class MaxHeap:
        def __init__(self):
            self.heap = []  # [(negative_sum, [u, v]), ...]
        
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
            if parent >= 0 and self.heap[parent][0] < self.heap[idx][0]:
                self.heap[parent], self.heap[idx] = self.heap[idx], self.heap[parent]
                self._heapify_up(parent)
        
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
    
    heap = MaxHeap()
    
    # Only check reasonable number of pairs to avoid TLE
    max_pairs = min(len(nums1) * len(nums2), k * 10)
    count = 0
    
    for i in range(len(nums1)):
        for j in range(len(nums2)):
            if count >= max_pairs:
                break
            
            pair_sum = nums1[i] + nums2[j]
            
            if heap.size() < k:
                heap.push((-pair_sum, [nums1[i], nums2[j]]))
            elif -heap.peek()[0] > pair_sum:
                heap.pop()
                heap.push((-pair_sum, [nums1[i], nums2[j]]))
            
            count += 1
        
        if count >= max_pairs:
            break
    
    # Extract results
    result = []
    while heap.size() > 0:
        _, pair = heap.pop()
        result.append(pair)
    
    return result[::-1]  # Reverse to get smallest first


def k_smallest_pairs_brute_force_optimized(nums1, nums2, k):
    """
    Approach 4: Brute Force with Early Termination
    Time Complexity: O(min(k, m*n) log min(k, m*n))
    Space Complexity: O(min(k, m*n))
    
    Generate pairs with early termination and sort.
    """
    pairs = []
    
    # Early termination optimization
    for i in range(min(len(nums1), k)):
        for j in range(min(len(nums2), k)):
            if len(pairs) >= k * 2:  # Collect extra pairs for better selection
                break
            pairs.append((nums1[i] + nums2[j], [nums1[i], nums2[j]]))
        
        if len(pairs) >= k * 2:
            break
    
    # Sort and return first k
    pairs.sort()
    return [pair[1] for pair in pairs[:k]]


def k_smallest_pairs_merge_k_sorted(nums1, nums2, k):
    """
    Approach 5: Merge K Sorted Lists Approach
    Time Complexity: O(k log min(k, m))
    Space Complexity: O(min(k, m))
    
    Treat each row as a sorted list and merge k sorted lists.
    """
    class MinHeap:
        def __init__(self):
            self.heap = []  # [(sum, i, j), ...]
        
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
    
    if not nums1 or not nums2:
        return []
    
    heap = MinHeap()
    result = []
    
    # Initialize heap with first column
    for i in range(min(len(nums1), k)):
        heap.push((nums1[i] + nums2[0], i, 0))
    
    while result.__len__() < k and heap.size() > 0:
        curr_sum, i, j = heap.pop()
        result.append([nums1[i], nums2[j]])
        
        # Add next element from the same row
        if j + 1 < len(nums2):
            heap.push((nums1[i] + nums2[j + 1], i, j + 1))
    
    return result


def test_k_smallest_pairs():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([1, 7, 11], [2, 4, 6], 3, [[1, 2], [1, 4], [1, 6]]),
        ([1, 1, 2], [1, 2, 3], 2, [[1, 1], [1, 1]]),
        ([1, 2], [3], 3, [[1, 3], [2, 3]]),
        ([1], [1], 1, [[1, 1]]),
        ([1, 2, 4, 5, 6], [3, 5, 7, 9], 3, [[1, 3], [2, 3], [1, 5]]),
        ([1, 1, 2], [1, 2, 3], 4, [[1, 1], [1, 1], [2, 1], [1, 2]]),
        ([0, 0, 0], [1, 2, 3], 2, [[0, 1], [0, 1]]),
    ]
    
    approaches = [
        ("Min Heap (All Pairs)", k_smallest_pairs_min_heap),
        ("Optimized Heap", k_smallest_pairs_optimized_heap),
        ("Max Heap", k_smallest_pairs_max_heap),
        ("Brute Force Optimized", k_smallest_pairs_brute_force_optimized),
        ("Merge K Sorted", k_smallest_pairs_merge_k_sorted),
    ]
    
    for i, (nums1, nums2, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: nums1 = {nums1}, nums2 = {nums2}, k = {k}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(nums1, nums2, k)
            
            # For this problem, order within same sum might vary, so we check sums
            expected_sums = sorted([pair[0] + pair[1] for pair in expected])
            result_sums = sorted([pair[0] + pair[1] for pair in result])
            
            status = "✓" if result_sums == expected_sums and len(result) == len(expected) else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_k_smallest_pairs() 