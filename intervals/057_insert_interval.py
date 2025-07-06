"""
57. Insert Interval

Problem:
You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.

Example 1:
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Example 2:
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]

Time Complexity: O(n) where n is the number of intervals
Space Complexity: O(n) for the result array
"""


def insert(intervals, newInterval):
    """
    Insert interval using three-phase approach.
    
    Args:
        intervals: List of non-overlapping sorted intervals
        newInterval: New interval to insert [start, end]
    
    Returns:
        List of intervals after insertion and merging
    """
    result = []
    i = 0
    n = len(intervals)
    
    # Phase 1: Add all intervals that end before newInterval starts
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1
    
    # Phase 2: Merge all overlapping intervals with newInterval
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    
    result.append(newInterval)
    
    # Phase 3: Add all remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result


def insert_binary_search(intervals, newInterval):
    """
    Insert interval using binary search to find insertion points.
    
    Args:
        intervals: List of non-overlapping sorted intervals
        newInterval: New interval to insert [start, end]
    
    Returns:
        List of intervals after insertion and merging
    """
    def find_insertion_point(target, is_start):
        """Find where to insert target using binary search"""
        left, right = 0, len(intervals)
        
        while left < right:
            mid = (left + right) // 2
            
            if is_start:
                # Finding insertion point for start
                if intervals[mid][1] < target:
                    left = mid + 1
                else:
                    right = mid
            else:
                # Finding insertion point for end
                if intervals[mid][0] <= target:
                    left = mid + 1
                else:
                    right = mid
        
        return left
    
    # Find which intervals to merge
    start_idx = find_insertion_point(newInterval[0], True)
    end_idx = find_insertion_point(newInterval[1], False)
    
    # Merge overlapping intervals
    if start_idx < len(intervals):
        newInterval[0] = min(newInterval[0], intervals[start_idx][0])
    if end_idx > 0:
        newInterval[1] = max(newInterval[1], intervals[end_idx - 1][1])
    
    # Build result
    result = intervals[:start_idx]
    result.append(newInterval)
    result.extend(intervals[end_idx:])
    
    return result


def insert_iterative(intervals, newInterval):
    """
    Insert interval using iterative approach.
    
    Args:
        intervals: List of non-overlapping sorted intervals
        newInterval: New interval to insert [start, end]
    
    Returns:
        List of intervals after insertion and merging
    """
    result = []
    inserted = False
    
    for interval in intervals:
        if interval[1] < newInterval[0]:
            # Current interval ends before new interval starts
            result.append(interval)
        elif interval[0] > newInterval[1]:
            # Current interval starts after new interval ends
            if not inserted:
                result.append(newInterval)
                inserted = True
            result.append(interval)
        else:
            # Overlapping intervals, merge
            newInterval[0] = min(newInterval[0], interval[0])
            newInterval[1] = max(newInterval[1], interval[1])
    
    # If newInterval hasn't been inserted yet
    if not inserted:
        result.append(newInterval)
    
    return result


def insert_stack(intervals, newInterval):
    """
    Insert interval using stack approach.
    
    Args:
        intervals: List of non-overlapping sorted intervals
        newInterval: New interval to insert [start, end]
    
    Returns:
        List of intervals after insertion and merging
    """
    stack = []
    
    # Add newInterval to intervals and sort
    all_intervals = intervals + [newInterval]
    all_intervals.sort(key=lambda x: x[0])
    
    for interval in all_intervals:
        if not stack or stack[-1][1] < interval[0]:
            stack.append(interval)
        else:
            # Merge with top of stack
            stack[-1][1] = max(stack[-1][1], interval[1])
    
    return stack


def insert_two_pointers(intervals, newInterval):
    """
    Insert interval using two pointers approach.
    
    Args:
        intervals: List of non-overlapping sorted intervals
        newInterval: New interval to insert [start, end]
    
    Returns:
        List of intervals after insertion and merging
    """
    result = []
    left = 0
    
    # Find position to insert newInterval
    while left < len(intervals) and intervals[left][0] < newInterval[0]:
        left += 1
    
    # Insert newInterval at correct position
    intervals.insert(left, newInterval)
    
    # Merge overlapping intervals
    for i, interval in enumerate(intervals):
        if not result or result[-1][1] < interval[0]:
            result.append(interval)
        else:
            result[-1][1] = max(result[-1][1], interval[1])
    
    return result


def insert_functional(intervals, newInterval):
    """
    Insert interval using functional programming approach.
    
    Args:
        intervals: List of non-overlapping sorted intervals
        newInterval: New interval to insert [start, end]
    
    Returns:
        List of intervals after insertion and merging
    """
    from functools import reduce
    
    def merge_with_new_interval(acc, interval):
        """Merge current interval with accumulated result"""
        result, new_int, inserted = acc
        
        if new_int is None:
            # New interval already processed
            result.append(interval)
        elif interval[1] < new_int[0]:
            # Current interval ends before new interval starts
            result.append(interval)
        elif interval[0] > new_int[1]:
            # Current interval starts after new interval ends
            if not inserted:
                result.append(new_int)
                inserted = True
            result.append(interval)
            new_int = None
        else:
            # Overlapping intervals, merge
            new_int[0] = min(new_int[0], interval[0])
            new_int[1] = max(new_int[1], interval[1])
        
        return result, new_int, inserted
    
    result, remaining_new, inserted = reduce(merge_with_new_interval, intervals, ([], newInterval[:], False))
    
    # Add remaining new interval if not inserted
    if remaining_new is not None:
        result.append(remaining_new)
    
    return result


def insert_recursive(intervals, newInterval):
    """
    Insert interval using recursive approach.
    
    Args:
        intervals: List of non-overlapping sorted intervals
        newInterval: New interval to insert [start, end]
    
    Returns:
        List of intervals after insertion and merging
    """
    def insert_helper(idx, current_new):
        """Recursively insert and merge intervals"""
        if idx >= len(intervals):
            return [current_new] if current_new else []
        
        interval = intervals[idx]
        
        if current_new is None:
            # New interval already processed
            return [interval] + insert_helper(idx + 1, None)
        elif interval[1] < current_new[0]:
            # Current interval ends before new interval starts
            return [interval] + insert_helper(idx + 1, current_new)
        elif interval[0] > current_new[1]:
            # Current interval starts after new interval ends
            return [current_new, interval] + insert_helper(idx + 1, None)
        else:
            # Overlapping intervals, merge
            merged = [min(current_new[0], interval[0]), max(current_new[1], interval[1])]
            return insert_helper(idx + 1, merged)
    
    return insert_helper(0, newInterval[:])


def insert_sweep_line(intervals, newInterval):
    """
    Insert interval using sweep line algorithm.
    
    Args:
        intervals: List of non-overlapping sorted intervals
        newInterval: New interval to insert [start, end]
    
    Returns:
        List of intervals after insertion and merging
    """
    # Create events for sweep line
    events = []
    
    # Add events for existing intervals
    for start, end in intervals:
        events.append((start, 1, 'existing'))   # Start event
        events.append((end, -1, 'existing'))    # End event
    
    # Add events for new interval
    events.append((newInterval[0], 1, 'new'))
    events.append((newInterval[1], -1, 'new'))
    
    # Sort events by time, with start events before end events
    events.sort(key=lambda x: (x[0], -x[1]))
    
    result = []
    active_count = 0
    start_time = None
    
    for time, delta, _ in events:
        if active_count == 0 and delta == 1:
            # Start of a new merged interval
            start_time = time
        
        active_count += delta
        
        if active_count == 0 and delta == -1:
            # End of current merged interval
            result.append([start_time, time])
    
    return result


def insert_divide_conquer(intervals, newInterval):
    """
    Insert interval using divide and conquer approach.
    
    Args:
        intervals: List of non-overlapping sorted intervals
        newInterval: New interval to insert [start, end]
    
    Returns:
        List of intervals after insertion and merging
    """
    def merge_intervals(left, right):
        """Merge two lists of intervals"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i][0] <= right[j][0]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        
        # Merge overlapping intervals
        merged = []
        for interval in result:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        
        return merged
    
    return merge_intervals(intervals, [newInterval])


# Test cases
if __name__ == "__main__":
    # Test case 1
    intervals1 = [[1,3],[6,9]]
    newInterval1 = [2,5]
    result1a = insert(intervals1.copy(), newInterval1.copy())
    result1b = insert_binary_search(intervals1.copy(), newInterval1.copy())
    result1c = insert_iterative(intervals1.copy(), newInterval1.copy())
    result1d = insert_stack(intervals1.copy(), newInterval1.copy())
    result1e = insert_two_pointers(intervals1.copy(), newInterval1.copy())
    result1f = insert_functional(intervals1.copy(), newInterval1.copy())
    result1g = insert_recursive(intervals1.copy(), newInterval1.copy())
    result1h = insert_sweep_line(intervals1.copy(), newInterval1.copy())
    result1i = insert_divide_conquer(intervals1.copy(), newInterval1.copy())
    print(f"Test 1 - Intervals: {intervals1}, New: {newInterval1}, Expected: [[1,5],[6,9]]")
    print(f"ThreePhase: {result1a}, BinarySearch: {result1b}, Iterative: {result1c}, Stack: {result1d}, TwoPointers: {result1e}, Functional: {result1f}, Recursive: {result1g}, SweepLine: {result1h}, DivideConquer: {result1i}")
    print()
    
    # Test case 2
    intervals2 = [[1,2],[3,5],[6,7],[8,10],[12,16]]
    newInterval2 = [4,8]
    result2a = insert(intervals2.copy(), newInterval2.copy())
    result2b = insert_binary_search(intervals2.copy(), newInterval2.copy())
    result2c = insert_iterative(intervals2.copy(), newInterval2.copy())
    result2d = insert_stack(intervals2.copy(), newInterval2.copy())
    result2e = insert_two_pointers(intervals2.copy(), newInterval2.copy())
    result2f = insert_functional(intervals2.copy(), newInterval2.copy())
    result2g = insert_recursive(intervals2.copy(), newInterval2.copy())
    result2h = insert_sweep_line(intervals2.copy(), newInterval2.copy())
    result2i = insert_divide_conquer(intervals2.copy(), newInterval2.copy())
    print(f"Test 2 - Intervals: {intervals2}, New: {newInterval2}, Expected: [[1,2],[3,10],[12,16]]")
    print(f"ThreePhase: {result2a}, BinarySearch: {result2b}, Iterative: {result2c}, Stack: {result2d}, TwoPointers: {result2e}, Functional: {result2f}, Recursive: {result2g}, SweepLine: {result2h}, DivideConquer: {result2i}")
    print()
    
    # Test case 3 - Insert at beginning
    intervals3 = [[3,5],[6,9]]
    newInterval3 = [1,2]
    result3a = insert(intervals3.copy(), newInterval3.copy())
    result3b = insert_binary_search(intervals3.copy(), newInterval3.copy())
    result3c = insert_iterative(intervals3.copy(), newInterval3.copy())
    result3d = insert_stack(intervals3.copy(), newInterval3.copy())
    result3e = insert_two_pointers(intervals3.copy(), newInterval3.copy())
    result3f = insert_functional(intervals3.copy(), newInterval3.copy())
    result3g = insert_recursive(intervals3.copy(), newInterval3.copy())
    result3h = insert_sweep_line(intervals3.copy(), newInterval3.copy())
    result3i = insert_divide_conquer(intervals3.copy(), newInterval3.copy())
    print(f"Test 3 - Intervals: {intervals3}, New: {newInterval3}, Expected: [[1,2],[3,5],[6,9]]")
    print(f"ThreePhase: {result3a}, BinarySearch: {result3b}, Iterative: {result3c}, Stack: {result3d}, TwoPointers: {result3e}, Functional: {result3f}, Recursive: {result3g}, SweepLine: {result3h}, DivideConquer: {result3i}")
    print()
    
    # Test case 4 - Insert at end
    intervals4 = [[1,3],[6,9]]
    newInterval4 = [10,12]
    result4a = insert(intervals4.copy(), newInterval4.copy())
    result4b = insert_binary_search(intervals4.copy(), newInterval4.copy())
    result4c = insert_iterative(intervals4.copy(), newInterval4.copy())
    result4d = insert_stack(intervals4.copy(), newInterval4.copy())
    result4e = insert_two_pointers(intervals4.copy(), newInterval4.copy())
    result4f = insert_functional(intervals4.copy(), newInterval4.copy())
    result4g = insert_recursive(intervals4.copy(), newInterval4.copy())
    result4h = insert_sweep_line(intervals4.copy(), newInterval4.copy())
    result4i = insert_divide_conquer(intervals4.copy(), newInterval4.copy())
    print(f"Test 4 - Intervals: {intervals4}, New: {newInterval4}, Expected: [[1,3],[6,9],[10,12]]")
    print(f"ThreePhase: {result4a}, BinarySearch: {result4b}, Iterative: {result4c}, Stack: {result4d}, TwoPointers: {result4e}, Functional: {result4f}, Recursive: {result4g}, SweepLine: {result4h}, DivideConquer: {result4i}")
    print()
    
    # Test case 5 - Empty intervals
    intervals5 = []
    newInterval5 = [5,7]
    result5a = insert(intervals5.copy(), newInterval5.copy())
    result5b = insert_binary_search(intervals5.copy(), newInterval5.copy())
    result5c = insert_iterative(intervals5.copy(), newInterval5.copy())
    result5d = insert_stack(intervals5.copy(), newInterval5.copy())
    result5e = insert_two_pointers(intervals5.copy(), newInterval5.copy())
    result5f = insert_functional(intervals5.copy(), newInterval5.copy())
    result5g = insert_recursive(intervals5.copy(), newInterval5.copy())
    result5h = insert_sweep_line(intervals5.copy(), newInterval5.copy())
    result5i = insert_divide_conquer(intervals5.copy(), newInterval5.copy())
    print(f"Test 5 - Intervals: {intervals5}, New: {newInterval5}, Expected: [[5,7]]")
    print(f"ThreePhase: {result5a}, BinarySearch: {result5b}, Iterative: {result5c}, Stack: {result5d}, TwoPointers: {result5e}, Functional: {result5f}, Recursive: {result5g}, SweepLine: {result5h}, DivideConquer: {result5i}")
    print()
    
    # Test case 6 - Merge all intervals
    intervals6 = [[1,2],[3,5],[6,7],[8,10]]
    newInterval6 = [0,11]
    result6a = insert(intervals6.copy(), newInterval6.copy())
    result6b = insert_binary_search(intervals6.copy(), newInterval6.copy())
    result6c = insert_iterative(intervals6.copy(), newInterval6.copy())
    result6d = insert_stack(intervals6.copy(), newInterval6.copy())
    result6e = insert_two_pointers(intervals6.copy(), newInterval6.copy())
    result6f = insert_functional(intervals6.copy(), newInterval6.copy())
    result6g = insert_recursive(intervals6.copy(), newInterval6.copy())
    result6h = insert_sweep_line(intervals6.copy(), newInterval6.copy())
    result6i = insert_divide_conquer(intervals6.copy(), newInterval6.copy())
    print(f"Test 6 - Intervals: {intervals6}, New: {newInterval6}, Expected: [[0,11]]")
    print(f"ThreePhase: {result6a}, BinarySearch: {result6b}, Iterative: {result6c}, Stack: {result6d}, TwoPointers: {result6e}, Functional: {result6f}, Recursive: {result6g}, SweepLine: {result6h}, DivideConquer: {result6i}")
    print()
    
    # Test case 7 - No overlap
    intervals7 = [[1,3],[6,9]]
    newInterval7 = [4,5]
    result7a = insert(intervals7.copy(), newInterval7.copy())
    result7b = insert_binary_search(intervals7.copy(), newInterval7.copy())
    result7c = insert_iterative(intervals7.copy(), newInterval7.copy())
    result7d = insert_stack(intervals7.copy(), newInterval7.copy())
    result7e = insert_two_pointers(intervals7.copy(), newInterval7.copy())
    result7f = insert_functional(intervals7.copy(), newInterval7.copy())
    result7g = insert_recursive(intervals7.copy(), newInterval7.copy())
    result7h = insert_sweep_line(intervals7.copy(), newInterval7.copy())
    result7i = insert_divide_conquer(intervals7.copy(), newInterval7.copy())
    print(f"Test 7 - Intervals: {intervals7}, New: {newInterval7}, Expected: [[1,3],[4,5],[6,9]]")
    print(f"ThreePhase: {result7a}, BinarySearch: {result7b}, Iterative: {result7c}, Stack: {result7d}, TwoPointers: {result7e}, Functional: {result7f}, Recursive: {result7g}, SweepLine: {result7h}, DivideConquer: {result7i}")
    print()
    
    # Test case 8 - Single interval overlap
    intervals8 = [[1,5]]
    newInterval8 = [2,3]
    result8a = insert(intervals8.copy(), newInterval8.copy())
    result8b = insert_binary_search(intervals8.copy(), newInterval8.copy())
    result8c = insert_iterative(intervals8.copy(), newInterval8.copy())
    result8d = insert_stack(intervals8.copy(), newInterval8.copy())
    result8e = insert_two_pointers(intervals8.copy(), newInterval8.copy())
    result8f = insert_functional(intervals8.copy(), newInterval8.copy())
    result8g = insert_recursive(intervals8.copy(), newInterval8.copy())
    result8h = insert_sweep_line(intervals8.copy(), newInterval8.copy())
    result8i = insert_divide_conquer(intervals8.copy(), newInterval8.copy())
    print(f"Test 8 - Intervals: {intervals8}, New: {newInterval8}, Expected: [[1,5]]")
    print(f"ThreePhase: {result8a}, BinarySearch: {result8b}, Iterative: {result8c}, Stack: {result8d}, TwoPointers: {result8e}, Functional: {result8f}, Recursive: {result8g}, SweepLine: {result8h}, DivideConquer: {result8i}")
    print()
    
    # Test case 9 - Touching intervals
    intervals9 = [[1,3],[6,9]]
    newInterval9 = [3,6]
    result9a = insert(intervals9.copy(), newInterval9.copy())
    result9b = insert_binary_search(intervals9.copy(), newInterval9.copy())
    result9c = insert_iterative(intervals9.copy(), newInterval9.copy())
    result9d = insert_stack(intervals9.copy(), newInterval9.copy())
    result9e = insert_two_pointers(intervals9.copy(), newInterval9.copy())
    result9f = insert_functional(intervals9.copy(), newInterval9.copy())
    result9g = insert_recursive(intervals9.copy(), newInterval9.copy())
    result9h = insert_sweep_line(intervals9.copy(), newInterval9.copy())
    result9i = insert_divide_conquer(intervals9.copy(), newInterval9.copy())
    print(f"Test 9 - Intervals: {intervals9}, New: {newInterval9}, Expected: [[1,9]]")
    print(f"ThreePhase: {result9a}, BinarySearch: {result9b}, Iterative: {result9c}, Stack: {result9d}, TwoPointers: {result9e}, Functional: {result9f}, Recursive: {result9g}, SweepLine: {result9h}, DivideConquer: {result9i}")
    print()
    
    # Test case 10 - Single point interval
    intervals10 = [[1,3],[6,9]]
    newInterval10 = [4,4]
    result10a = insert(intervals10.copy(), newInterval10.copy())
    result10b = insert_binary_search(intervals10.copy(), newInterval10.copy())
    result10c = insert_iterative(intervals10.copy(), newInterval10.copy())
    result10d = insert_stack(intervals10.copy(), newInterval10.copy())
    result10e = insert_two_pointers(intervals10.copy(), newInterval10.copy())
    result10f = insert_functional(intervals10.copy(), newInterval10.copy())
    result10g = insert_recursive(intervals10.copy(), newInterval10.copy())
    result10h = insert_sweep_line(intervals10.copy(), newInterval10.copy())
    result10i = insert_divide_conquer(intervals10.copy(), newInterval10.copy())
    print(f"Test 10 - Intervals: {intervals10}, New: {newInterval10}, Expected: [[1,3],[4,4],[6,9]]")
    print(f"ThreePhase: {result10a}, BinarySearch: {result10b}, Iterative: {result10c}, Stack: {result10d}, TwoPointers: {result10e}, Functional: {result10f}, Recursive: {result10g}, SweepLine: {result10h}, DivideConquer: {result10i}") 