"""
56. Merge Intervals

Problem:
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.

Time Complexity: O(n log n) due to sorting
Space Complexity: O(1) excluding the output array
"""


def merge(intervals):
    """
    Merge overlapping intervals using sorting approach.
    
    Args:
        intervals: List of intervals [start, end]
    
    Returns:
        List of merged intervals
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # If current interval overlaps with the last merged interval
        if current[0] <= last[1]:
            # Merge intervals by updating the end time
            last[1] = max(last[1], current[1])
        else:
            # No overlap, add current interval
            merged.append(current)
    
    return merged


def merge_stack(intervals):
    """
    Merge overlapping intervals using stack approach.
    
    Args:
        intervals: List of intervals [start, end]
    
    Returns:
        List of merged intervals
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    stack = []
    
    for interval in intervals:
        if not stack or stack[-1][1] < interval[0]:
            # No overlap with top of stack
            stack.append(interval)
        else:
            # Overlap with top of stack, merge
            stack[-1][1] = max(stack[-1][1], interval[1])
    
    return stack


def merge_in_place(intervals):
    """
    Merge overlapping intervals in place.
    
    Args:
        intervals: List of intervals [start, end]
    
    Returns:
        List of merged intervals
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    write_idx = 0
    
    for read_idx in range(1, len(intervals)):
        current = intervals[read_idx]
        last_merged = intervals[write_idx]
        
        # If current interval overlaps with last merged
        if current[0] <= last_merged[1]:
            # Merge by updating end time
            intervals[write_idx][1] = max(last_merged[1], current[1])
        else:
            # No overlap, move to next position
            write_idx += 1
            intervals[write_idx] = current
    
    return intervals[:write_idx + 1]


def merge_two_pass(intervals):
    """
    Merge overlapping intervals using two pass approach.
    
    Args:
        intervals: List of intervals [start, end]
    
    Returns:
        List of merged intervals
    """
    if not intervals:
        return []
    
    # First pass: sort intervals
    intervals.sort(key=lambda x: x[0])
    
    # Second pass: merge overlapping intervals
    result = []
    for interval in intervals:
        if not result or result[-1][1] < interval[0]:
            result.append(interval[:])  # Create a copy
        else:
            result[-1][1] = max(result[-1][1], interval[1])
    
    return result


def merge_recursive(intervals):
    """
    Merge overlapping intervals using recursive approach.
    
    Args:
        intervals: List of intervals [start, end]
    
    Returns:
        List of merged intervals
    """
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    
    def merge_helper(merged, remaining):
        """Recursively merge intervals"""
        if not remaining:
            return merged
        
        current = remaining[0]
        
        if not merged or merged[-1][1] < current[0]:
            # No overlap
            return merge_helper(merged + [current], remaining[1:])
        else:
            # Overlap, merge with last interval
            merged[-1][1] = max(merged[-1][1], current[1])
            return merge_helper(merged, remaining[1:])
    
    return merge_helper([], intervals)


def merge_functional(intervals):
    """
    Merge overlapping intervals using functional programming.
    
    Args:
        intervals: List of intervals [start, end]
    
    Returns:
        List of merged intervals
    """
    if not intervals:
        return []
    
    from functools import reduce
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    def merge_two_intervals(merged, current):
        """Merge current interval with merged list"""
        if not merged or merged[-1][1] < current[0]:
            return merged + [current]
        else:
            merged[-1][1] = max(merged[-1][1], current[1])
            return merged
    
    return reduce(merge_two_intervals, sorted_intervals, [])


def merge_sweep_line(intervals):
    """
    Merge overlapping intervals using sweep line approach.
    
    Args:
        intervals: List of intervals [start, end]
    
    Returns:
        List of merged intervals
    """
    if not intervals:
        return []
    
    # Create events for sweep line
    events = []
    for start, end in intervals:
        events.append((start, 1))   # Start event
        events.append((end + 1, -1))  # End event (end+1 to handle touching intervals)
    
    # Sort events by time, with end events before start events at same time
    events.sort(key=lambda x: (x[0], x[1]))
    
    result = []
    active_count = 0
    start_time = None
    
    for time, delta in events:
        if active_count == 0 and delta == 1:
            # Start of a new merged interval
            start_time = time
        
        active_count += delta
        
        if active_count == 0 and delta == -1:
            # End of current merged interval
            result.append([start_time, time - 1])
    
    return result


def merge_union_find(intervals):
    """
    Merge overlapping intervals using Union-Find approach.
    
    Args:
        intervals: List of intervals [start, end]
    
    Returns:
        List of merged intervals
    """
    if not intervals:
        return []
    
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
    
    n = len(intervals)
    uf = UnionFind(n)
    
    # Union overlapping intervals
    for i in range(n):
        for j in range(i + 1, n):
            if (intervals[i][0] <= intervals[j][1] and 
                intervals[j][0] <= intervals[i][1]):
                uf.union(i, j)
    
    # Group intervals by their root
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(intervals[i])
    
    # Merge each group
    result = []
    for group in groups.values():
        min_start = min(interval[0] for interval in group)
        max_end = max(interval[1] for interval in group)
        result.append([min_start, max_end])
    
    return sorted(result)


# Test cases
if __name__ == "__main__":
    # Test case 1
    intervals1 = [[1,3],[2,6],[8,10],[15,18]]
    result1a = merge(intervals1.copy())
    result1b = merge_stack(intervals1.copy())
    result1c = merge_in_place(intervals1.copy())
    result1d = merge_two_pass(intervals1.copy())
    result1e = merge_recursive(intervals1.copy())
    result1f = merge_functional(intervals1.copy())
    result1g = merge_sweep_line(intervals1.copy())
    result1h = merge_union_find(intervals1.copy())
    print(f"Test 1 - Input: {intervals1}, Expected: [[1,6],[8,10],[15,18]]")
    print(f"Sort: {result1a}, Stack: {result1b}, InPlace: {result1c}, TwoPass: {result1d}, Recursive: {result1e}, Functional: {result1f}, SweepLine: {result1g}, UnionFind: {result1h}")
    print()
    
    # Test case 2
    intervals2 = [[1,4],[4,5]]
    result2a = merge(intervals2.copy())
    result2b = merge_stack(intervals2.copy())
    result2c = merge_in_place(intervals2.copy())
    result2d = merge_two_pass(intervals2.copy())
    result2e = merge_recursive(intervals2.copy())
    result2f = merge_functional(intervals2.copy())
    result2g = merge_sweep_line(intervals2.copy())
    result2h = merge_union_find(intervals2.copy())
    print(f"Test 2 - Input: {intervals2}, Expected: [[1,5]]")
    print(f"Sort: {result2a}, Stack: {result2b}, InPlace: {result2c}, TwoPass: {result2d}, Recursive: {result2e}, Functional: {result2f}, SweepLine: {result2g}, UnionFind: {result2h}")
    print()
    
    # Test case 3 - No overlaps
    intervals3 = [[1,2],[3,4],[5,6]]
    result3a = merge(intervals3.copy())
    result3b = merge_stack(intervals3.copy())
    result3c = merge_in_place(intervals3.copy())
    result3d = merge_two_pass(intervals3.copy())
    result3e = merge_recursive(intervals3.copy())
    result3f = merge_functional(intervals3.copy())
    result3g = merge_sweep_line(intervals3.copy())
    result3h = merge_union_find(intervals3.copy())
    print(f"Test 3 - Input: {intervals3}, Expected: [[1,2],[3,4],[5,6]]")
    print(f"Sort: {result3a}, Stack: {result3b}, InPlace: {result3c}, TwoPass: {result3d}, Recursive: {result3e}, Functional: {result3f}, SweepLine: {result3g}, UnionFind: {result3h}")
    print()
    
    # Test case 4 - All overlap
    intervals4 = [[1,4],[2,3]]
    result4a = merge(intervals4.copy())
    result4b = merge_stack(intervals4.copy())
    result4c = merge_in_place(intervals4.copy())
    result4d = merge_two_pass(intervals4.copy())
    result4e = merge_recursive(intervals4.copy())
    result4f = merge_functional(intervals4.copy())
    result4g = merge_sweep_line(intervals4.copy())
    result4h = merge_union_find(intervals4.copy())
    print(f"Test 4 - Input: {intervals4}, Expected: [[1,4]]")
    print(f"Sort: {result4a}, Stack: {result4b}, InPlace: {result4c}, TwoPass: {result4d}, Recursive: {result4e}, Functional: {result4f}, SweepLine: {result4g}, UnionFind: {result4h}")
    print()
    
    # Test case 5 - Single interval
    intervals5 = [[1,4]]
    result5a = merge(intervals5.copy())
    result5b = merge_stack(intervals5.copy())
    result5c = merge_in_place(intervals5.copy())
    result5d = merge_two_pass(intervals5.copy())
    result5e = merge_recursive(intervals5.copy())
    result5f = merge_functional(intervals5.copy())
    result5g = merge_sweep_line(intervals5.copy())
    result5h = merge_union_find(intervals5.copy())
    print(f"Test 5 - Input: {intervals5}, Expected: [[1,4]]")
    print(f"Sort: {result5a}, Stack: {result5b}, InPlace: {result5c}, TwoPass: {result5d}, Recursive: {result5e}, Functional: {result5f}, SweepLine: {result5g}, UnionFind: {result5h}")
    print()
    
    # Test case 6 - Empty input
    intervals6 = []
    result6a = merge(intervals6.copy())
    result6b = merge_stack(intervals6.copy())
    result6c = merge_in_place(intervals6.copy())
    result6d = merge_two_pass(intervals6.copy())
    result6e = merge_recursive(intervals6.copy())
    result6f = merge_functional(intervals6.copy())
    result6g = merge_sweep_line(intervals6.copy())
    result6h = merge_union_find(intervals6.copy())
    print(f"Test 6 - Input: {intervals6}, Expected: []")
    print(f"Sort: {result6a}, Stack: {result6b}, InPlace: {result6c}, TwoPass: {result6d}, Recursive: {result6e}, Functional: {result6f}, SweepLine: {result6g}, UnionFind: {result6h}")
    print()
    
    # Test case 7 - Complex overlapping
    intervals7 = [[1,3],[2,6],[8,10],[9,12],[15,18]]
    result7a = merge(intervals7.copy())
    result7b = merge_stack(intervals7.copy())
    result7c = merge_in_place(intervals7.copy())
    result7d = merge_two_pass(intervals7.copy())
    result7e = merge_recursive(intervals7.copy())
    result7f = merge_functional(intervals7.copy())
    result7g = merge_sweep_line(intervals7.copy())
    result7h = merge_union_find(intervals7.copy())
    print(f"Test 7 - Input: {intervals7}, Expected: [[1,6],[8,12],[15,18]]")
    print(f"Sort: {result7a}, Stack: {result7b}, InPlace: {result7c}, TwoPass: {result7d}, Recursive: {result7e}, Functional: {result7f}, SweepLine: {result7g}, UnionFind: {result7h}")
    print()
    
    # Test case 8 - Touching intervals
    intervals8 = [[1,3],[4,6]]
    result8a = merge(intervals8.copy())
    result8b = merge_stack(intervals8.copy())
    result8c = merge_in_place(intervals8.copy())
    result8d = merge_two_pass(intervals8.copy())
    result8e = merge_recursive(intervals8.copy())
    result8f = merge_functional(intervals8.copy())
    result8g = merge_sweep_line(intervals8.copy())
    result8h = merge_union_find(intervals8.copy())
    print(f"Test 8 - Input: {intervals8}, Expected: [[1,3],[4,6]]")
    print(f"Sort: {result8a}, Stack: {result8b}, InPlace: {result8c}, TwoPass: {result8d}, Recursive: {result8e}, Functional: {result8f}, SweepLine: {result8g}, UnionFind: {result8h}")
    print()
    
    # Test case 9 - Same start times
    intervals9 = [[1,4],[1,5]]
    result9a = merge(intervals9.copy())
    result9b = merge_stack(intervals9.copy())
    result9c = merge_in_place(intervals9.copy())
    result9d = merge_two_pass(intervals9.copy())
    result9e = merge_recursive(intervals9.copy())
    result9f = merge_functional(intervals9.copy())
    result9g = merge_sweep_line(intervals9.copy())
    result9h = merge_union_find(intervals9.copy())
    print(f"Test 9 - Input: {intervals9}, Expected: [[1,5]]")
    print(f"Sort: {result9a}, Stack: {result9b}, InPlace: {result9c}, TwoPass: {result9d}, Recursive: {result9e}, Functional: {result9f}, SweepLine: {result9g}, UnionFind: {result9h}")
    print()
    
    # Test case 10 - Nested intervals
    intervals10 = [[1,10],[2,3],[4,5],[6,7],[8,9]]
    result10a = merge(intervals10.copy())
    result10b = merge_stack(intervals10.copy())
    result10c = merge_in_place(intervals10.copy())
    result10d = merge_two_pass(intervals10.copy())
    result10e = merge_recursive(intervals10.copy())
    result10f = merge_functional(intervals10.copy())
    result10g = merge_sweep_line(intervals10.copy())
    result10h = merge_union_find(intervals10.copy())
    print(f"Test 10 - Input: {intervals10}, Expected: [[1,10]]")
    print(f"Sort: {result10a}, Stack: {result10b}, InPlace: {result10c}, TwoPass: {result10d}, Recursive: {result10e}, Functional: {result10f}, SweepLine: {result10g}, UnionFind: {result10h}") 