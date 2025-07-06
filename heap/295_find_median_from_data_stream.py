"""
295. Find Median from Data Stream

The median is the middle value in an ordered integer list. If the size of the list is even, 
there is no middle value, and the median is the mean of the two middle values.

Implement the MedianFinder class:
- MedianFinder() initializes the MedianFinder object.
- void addNum(int num) adds the integer num from the data stream to the data structure.
- double findMedian() returns the median of all elements so far.

Example 1:
Input:
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]

Output:
[null, null, null, 1.5, null, 2.0]

Explanation:
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr = [1, 2, 3]
medianFinder.findMedian(); // return 2.0

Constraints:
- -10^5 <= num <= 10^5
- There will be at least one element in the data structure before calling findMedian.
- At most 5 * 10^4 calls will be made to addNum and findMedian.

Follow up:
- If all integer numbers from the stream are in the range [0, 100], how would you optimize it?
- If 99% of all integer numbers from the stream are in the range [0, 100], how would you optimize it?
"""

class MedianFinderTwoHeaps:
    """
    Approach 1: Two Heaps (Max Heap + Min Heap)
    Time Complexity: O(log n) for addNum, O(1) for findMedian
    Space Complexity: O(n)
    
    Use max heap for smaller half and min heap for larger half.
    """
    
    def __init__(self):
        # Max heap for smaller half (negate values for max heap behavior)
        self.max_heap = []
        # Min heap for larger half
        self.min_heap = []
    
    def addNum(self, num):
        # Add to max heap first
        self._push_max_heap(num)
        
        # Balance: move largest from max heap to min heap
        if self.max_heap:
            val = self._pop_max_heap()
            self._push_min_heap(val)
        
        # Rebalance if min heap has more than 1 extra element
        if len(self.min_heap) > len(self.max_heap) + 1:
            val = self._pop_min_heap()
            self._push_max_heap(val)
    
    def findMedian(self):
        if len(self.min_heap) > len(self.max_heap):
            return float(self.min_heap[0])
        else:
            return (self.min_heap[0] + (-self.max_heap[0])) / 2.0
    
    def _push_max_heap(self, val):
        self.max_heap.append(-val)
        self._heapify_up_max(len(self.max_heap) - 1)
    
    def _pop_max_heap(self):
        if not self.max_heap:
            return None
        self.max_heap[0], self.max_heap[-1] = self.max_heap[-1], self.max_heap[0]
        result = -self.max_heap.pop()
        if self.max_heap:
            self._heapify_down_max(0)
        return result
    
    def _push_min_heap(self, val):
        self.min_heap.append(val)
        self._heapify_up_min(len(self.min_heap) - 1)
    
    def _pop_min_heap(self):
        if not self.min_heap:
            return None
        self.min_heap[0], self.min_heap[-1] = self.min_heap[-1], self.min_heap[0]
        result = self.min_heap.pop()
        if self.min_heap:
            self._heapify_down_min(0)
        return result
    
    def _heapify_up_max(self, idx):
        parent = (idx - 1) // 2
        if parent >= 0 and self.max_heap[parent] > self.max_heap[idx]:
            self.max_heap[parent], self.max_heap[idx] = self.max_heap[idx], self.max_heap[parent]
            self._heapify_up_max(parent)
    
    def _heapify_down_max(self, idx):
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2
        
        if left < len(self.max_heap) and self.max_heap[left] < self.max_heap[smallest]:
            smallest = left
        if right < len(self.max_heap) and self.max_heap[right] < self.max_heap[smallest]:
            smallest = right
        
        if smallest != idx:
            self.max_heap[idx], self.max_heap[smallest] = self.max_heap[smallest], self.max_heap[idx]
            self._heapify_down_max(smallest)
    
    def _heapify_up_min(self, idx):
        parent = (idx - 1) // 2
        if parent >= 0 and self.min_heap[parent] > self.min_heap[idx]:
            self.min_heap[parent], self.min_heap[idx] = self.min_heap[idx], self.min_heap[parent]
            self._heapify_up_min(parent)
    
    def _heapify_down_min(self, idx):
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2
        
        if left < len(self.min_heap) and self.min_heap[left] < self.min_heap[smallest]:
            smallest = left
        if right < len(self.min_heap) and self.min_heap[right] < self.min_heap[smallest]:
            smallest = right
        
        if smallest != idx:
            self.min_heap[idx], self.min_heap[smallest] = self.min_heap[smallest], self.min_heap[idx]
            self._heapify_down_min(smallest)


class MedianFinderSortedArray:
    """
    Approach 2: Sorted Array
    Time Complexity: O(n) for addNum, O(1) for findMedian
    Space Complexity: O(n)
    
    Maintain a sorted array and use binary search for insertion.
    """
    
    def __init__(self):
        self.nums = []
    
    def addNum(self, num):
        # Binary search for insertion position
        left, right = 0, len(self.nums)
        
        while left < right:
            mid = (left + right) // 2
            if self.nums[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        self.nums.insert(left, num)
    
    def findMedian(self):
        n = len(self.nums)
        if n % 2 == 1:
            return float(self.nums[n // 2])
        else:
            return (self.nums[n // 2 - 1] + self.nums[n // 2]) / 2.0


class MedianFinderSimpleList:
    """
    Approach 3: Simple List with Sorting
    Time Complexity: O(n log n) for addNum, O(1) for findMedian
    Space Complexity: O(n)
    
    Simple approach using list and sorting for each median query.
    """
    
    def __init__(self):
        self.nums = []
    
    def addNum(self, num):
        self.nums.append(num)
    
    def findMedian(self):
        sorted_nums = sorted(self.nums)
        n = len(sorted_nums)
        if n % 2 == 1:
            return float(sorted_nums[n // 2])
        else:
            return (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2.0


class MedianFinderBuckets:
    """
    Approach 4: Bucket Approach (For limited range)
    Time Complexity: O(1) for addNum, O(range) for findMedian
    Space Complexity: O(range)
    
    Use buckets when numbers are in a limited range [0, 100].
    """
    
    def __init__(self, max_val=200):  # Extended range to handle negative numbers
        self.buckets = [0] * (max_val + 1)
        self.offset = 100  # Offset for negative numbers
        self.count = 0
    
    def addNum(self, num):
        self.buckets[num + self.offset] += 1
        self.count += 1
    
    def findMedian(self):
        target = self.count // 2
        curr_count = 0
        
        if self.count % 2 == 1:
            # Odd number of elements
            for i, freq in enumerate(self.buckets):
                curr_count += freq
                if curr_count > target:
                    return float(i - self.offset)
        else:
            # Even number of elements
            first_median = None
            for i, freq in enumerate(self.buckets):
                curr_count += freq
                if first_median is None and curr_count >= target:
                    first_median = i - self.offset
                if curr_count > target:
                    return (first_median + (i - self.offset)) / 2.0
                if curr_count == target:
                    first_median = i - self.offset
        
        return 0.0


class MedianFinderOptimized:
    """
    Approach 5: Optimized Two Heaps with Size Tracking
    Time Complexity: O(log n) for addNum, O(1) for findMedian
    Space Complexity: O(n)
    
    Optimized version with better balancing logic.
    """
    
    def __init__(self):
        self.max_heap = []  # For smaller half
        self.min_heap = []  # For larger half
        self.max_heap_size = 0
        self.min_heap_size = 0
    
    def addNum(self, num):
        # Always add to max_heap first
        if self.max_heap_size == 0 or num <= -self.max_heap[0]:
            self._push_max_heap(num)
        else:
            self._push_min_heap(num)
        
        # Balance heaps
        self._balance()
    
    def findMedian(self):
        if self.max_heap_size == self.min_heap_size:
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0
        else:
            return float(-self.max_heap[0])
    
    def _balance(self):
        # Ensure max_heap has at most 1 more element than min_heap
        if self.max_heap_size > self.min_heap_size + 1:
            val = self._pop_max_heap()
            self._push_min_heap(val)
        elif self.min_heap_size > self.max_heap_size:
            val = self._pop_min_heap()
            self._push_max_heap(val)
    
    def _push_max_heap(self, val):
        self.max_heap.append(-val)
        self.max_heap_size += 1
        self._heapify_up_max(self.max_heap_size - 1)
    
    def _pop_max_heap(self):
        if self.max_heap_size == 0:
            return None
        self.max_heap[0], self.max_heap[self.max_heap_size - 1] = \
            self.max_heap[self.max_heap_size - 1], self.max_heap[0]
        result = -self.max_heap.pop()
        self.max_heap_size -= 1
        if self.max_heap_size > 0:
            self._heapify_down_max(0)
        return result
    
    def _push_min_heap(self, val):
        self.min_heap.append(val)
        self.min_heap_size += 1
        self._heapify_up_min(self.min_heap_size - 1)
    
    def _pop_min_heap(self):
        if self.min_heap_size == 0:
            return None
        self.min_heap[0], self.min_heap[self.min_heap_size - 1] = \
            self.min_heap[self.min_heap_size - 1], self.min_heap[0]
        result = self.min_heap.pop()
        self.min_heap_size -= 1
        if self.min_heap_size > 0:
            self._heapify_down_min(0)
        return result
    
    def _heapify_up_max(self, idx):
        parent = (idx - 1) // 2
        if parent >= 0 and self.max_heap[parent] > self.max_heap[idx]:
            self.max_heap[parent], self.max_heap[idx] = self.max_heap[idx], self.max_heap[parent]
            self._heapify_up_max(parent)
    
    def _heapify_down_max(self, idx):
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2
        
        if left < self.max_heap_size and self.max_heap[left] < self.max_heap[smallest]:
            smallest = left
        if right < self.max_heap_size and self.max_heap[right] < self.max_heap[smallest]:
            smallest = right
        
        if smallest != idx:
            self.max_heap[idx], self.max_heap[smallest] = self.max_heap[smallest], self.max_heap[idx]
            self._heapify_down_max(smallest)
    
    def _heapify_up_min(self, idx):
        parent = (idx - 1) // 2
        if parent >= 0 and self.min_heap[parent] > self.min_heap[idx]:
            self.min_heap[parent], self.min_heap[idx] = self.min_heap[idx], self.min_heap[parent]
            self._heapify_up_min(parent)
    
    def _heapify_down_min(self, idx):
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2
        
        if left < self.min_heap_size and self.min_heap[left] < self.min_heap[smallest]:
            smallest = left
        if right < self.min_heap_size and self.min_heap[right] < self.min_heap[smallest]:
            smallest = right
        
        if smallest != idx:
            self.min_heap[idx], self.min_heap[smallest] = self.min_heap[smallest], self.min_heap[idx]
            self._heapify_down_min(smallest)


def test_median_finder():
    """Test all approaches with various test cases."""
    
    test_cases = [
        # (operations, arguments, expected_results)
        (["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"],
         [[], [1], [2], [], [3], []],
         [None, None, None, 1.5, None, 2.0]),
        
        (["MedianFinder", "addNum", "findMedian", "addNum", "findMedian", "addNum", "findMedian"],
         [[], [1], [], [2], [], [3], []],
         [None, None, 1.0, None, 1.5, None, 2.0]),
        
        (["MedianFinder", "addNum", "addNum", "addNum", "addNum", "findMedian"],
         [[], [6], [10], [2], [6], []],
         [None, None, None, None, None, 6.0]),
    ]
    
    implementations = [
        ("Two Heaps", MedianFinderTwoHeaps),
        ("Sorted Array", MedianFinderSortedArray),
        ("Simple List", MedianFinderSimpleList),
        ("Buckets", MedianFinderBuckets),
        ("Optimized Two Heaps", MedianFinderOptimized),
    ]
    
    for test_idx, (operations, arguments, expected) in enumerate(test_cases):
        print(f"\nTest Case {test_idx + 1}:")
        print(f"Operations: {operations}")
        print(f"Arguments: {arguments}")
        print(f"Expected: {expected}")
        
        for name, MedianFinderClass in implementations:
            print(f"\n{name}:")
            mf = None
            results = []
            
            for i, (op, arg) in enumerate(zip(operations, arguments)):
                if op == "MedianFinder":
                    mf = MedianFinderClass()
                    results.append(None)
                elif op == "addNum":
                    mf.addNum(arg[0])
                    results.append(None)
                elif op == "findMedian":
                    result = mf.findMedian()
                    results.append(result)
            
            # Check results
            all_correct = True
            for i, (result, exp) in enumerate(zip(results, expected)):
                if exp is not None:
                    if abs(result - exp) > 1e-5:
                        all_correct = False
                        break
            
            status = "✓" if all_correct else "✗"
            print(f"{status} Results: {results}")


if __name__ == "__main__":
    test_median_finder() 