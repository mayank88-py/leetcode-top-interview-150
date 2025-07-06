"""
435. Non-overlapping Intervals

Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Example 1:
Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.

Example 2:
Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.

Example 3:
Input: intervals = [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of the intervals since they're already non-overlapping.

Constraints:
- 1 <= intervals.length <= 10^5
- intervals[i].length == 2
- -5 * 10^4 <= starti < endi <= 5 * 10^4
"""

from typing import List


def erase_overlap_intervals_greedy(intervals: List[List[int]]) -> int:
    """
    Greedy approach: Sort by end time and keep intervals that end earliest.
    
    Time Complexity: O(n log n) - sorting
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Sort intervals by end time
    2. Keep track of the last kept interval's end time
    3. For each interval, if it doesn't overlap with the last kept interval, keep it
    4. Otherwise, count it as removed
    """
    if not intervals:
        return 0
    
    # Sort by end time
    intervals.sort(key=lambda x: x[1])
    
    removed = 0
    last_end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        start, end = intervals[i]
        
        # If current interval overlaps with the last kept interval
        if start < last_end:
            removed += 1
        else:
            # No overlap, keep this interval
            last_end = end
    
    return removed


def erase_overlap_intervals_dp(intervals: List[List[int]]) -> int:
    """
    Dynamic Programming approach: Find maximum number of non-overlapping intervals.
    
    Time Complexity: O(n^2) - nested loops
    Space Complexity: O(n) - dp array
    
    Algorithm:
    1. Sort intervals by start time
    2. Use DP where dp[i] = max non-overlapping intervals ending at or before i
    3. For each interval, find the best previous non-overlapping interval
    4. Return total - max non-overlapping
    """
    if not intervals:
        return 0
    
    n = len(intervals)
    intervals.sort()
    
    # dp[i] = maximum number of non-overlapping intervals ending at or before i
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            # If intervals[j] and intervals[i] don't overlap
            if intervals[j][1] <= intervals[i][0]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    # Maximum non-overlapping intervals
    max_non_overlapping = max(dp)
    
    # Return number of intervals to remove
    return n - max_non_overlapping


def erase_overlap_intervals_binary_search(intervals: List[List[int]]) -> int:
    """
    Binary search approach: Optimized DP using binary search.
    
    Time Complexity: O(n log n) - sorting + binary search
    Space Complexity: O(n) - dp array
    
    Algorithm:
    1. Sort intervals by start time
    2. Use binary search to find the rightmost non-overlapping interval
    3. Build dp array efficiently
    """
    if not intervals:
        return 0
    
    n = len(intervals)
    intervals.sort()
    
    # dp[i] = maximum number of non-overlapping intervals ending at or before i
    dp = [1] * n
    
    def binary_search(i):
        """Find the rightmost interval j where intervals[j][1] <= intervals[i][0]"""
        left, right = 0, i - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if intervals[mid][1] <= intervals[i][0]:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    for i in range(1, n):
        # Find the rightmost non-overlapping interval before i
        j = binary_search(i)
        
        if j != -1:
            dp[i] = max(dp[i], dp[j] + 1)
        
        # Also consider the best result from previous interval
        dp[i] = max(dp[i], dp[i-1])
    
    return n - dp[n-1]


def erase_overlap_intervals_activity_selection(intervals: List[List[int]]) -> int:
    """
    Activity Selection approach: Select maximum number of non-overlapping activities.
    
    Time Complexity: O(n log n) - sorting
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. Sort by end time
    2. Greedily select intervals that don't overlap
    3. Count selected intervals and subtract from total
    """
    if not intervals:
        return 0
    
    # Sort by end time
    intervals.sort(key=lambda x: x[1])
    
    selected = 1  # Always select the first interval
    last_end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        start, end = intervals[i]
        
        # If current interval doesn't overlap with last selected
        if start >= last_end:
            selected += 1
            last_end = end
    
    return len(intervals) - selected


def erase_overlap_intervals_sweep_line(intervals: List[List[int]]) -> int:
    """
    Sweep line approach: Process events in chronological order.
    
    Time Complexity: O(n log n) - sorting events
    Space Complexity: O(n) - events array
    
    Algorithm:
    1. Create events for start and end of each interval
    2. Sort events by time
    3. Process events to track overlaps
    """
    if not intervals:
        return 0
    
    events = []
    
    # Create events: (time, type, index)
    # type: 0 = start, 1 = end
    for i, (start, end) in enumerate(intervals):
        events.append((start, 0, i))  # Start event
        events.append((end, 1, i))    # End event
    
    # Sort events by time, with end events before start events at same time
    events.sort()
    
    active_intervals = set()
    removed = 0
    
    for time, event_type, idx in events:
        if event_type == 0:  # Start event
            # Check if this interval overlaps with any active interval
            if active_intervals:
                # Remove the interval that ends latest
                if intervals[idx][1] > max(intervals[i][1] for i in active_intervals):
                    removed += 1
                    continue
                else:
                    # Remove one of the active intervals
                    max_end_idx = max(active_intervals, key=lambda i: intervals[i][1])
                    active_intervals.remove(max_end_idx)
                    removed += 1
            
            active_intervals.add(idx)
        else:  # End event
            active_intervals.discard(idx)
    
    return removed


def erase_overlap_intervals_recursive(intervals: List[List[int]]) -> int:
    """
    Recursive approach with memoization: Try all possibilities.
    
    Time Complexity: O(n^2) - with memoization
    Space Complexity: O(n^2) - memoization table
    
    Algorithm:
    1. Sort intervals by start time
    2. For each interval, decide whether to include it or not
    3. Use memoization to avoid recalculating subproblems
    """
    if not intervals:
        return 0
    
    intervals.sort()
    n = len(intervals)
    memo = {}
    
    def can_include(i, last_end):
        """Check if interval i can be included after an interval ending at last_end"""
        return intervals[i][0] >= last_end
    
    def solve(i, last_end):
        """Find maximum non-overlapping intervals starting from index i"""
        if i >= n:
            return 0
        
        if (i, last_end) in memo:
            return memo[(i, last_end)]
        
        # Option 1: Skip current interval
        result = solve(i + 1, last_end)
        
        # Option 2: Include current interval (if possible)
        if can_include(i, last_end):
            result = max(result, 1 + solve(i + 1, intervals[i][1]))
        
        memo[(i, last_end)] = result
        return result
    
    max_non_overlapping = solve(0, float('-inf'))
    return n - max_non_overlapping


def erase_overlap_intervals_stack(intervals: List[List[int]]) -> int:
    """
    Stack-based approach: Use stack to track non-overlapping intervals.
    
    Time Complexity: O(n log n) - sorting
    Space Complexity: O(n) - stack
    
    Algorithm:
    1. Sort intervals by start time
    2. Use stack to maintain non-overlapping intervals
    3. When overlap occurs, keep the interval with smaller end time
    """
    if not intervals:
        return 0
    
    intervals.sort()
    stack = []
    removed = 0
    
    for interval in intervals:
        start, end = interval
        
        # Remove overlapping intervals from stack
        while stack and stack[-1][1] > start:
            # Remove the interval with larger end time
            if stack[-1][1] > end:
                stack.pop()
                removed += 1
            else:
                removed += 1
                break
        else:
            stack.append(interval)
    
    return removed


def erase_overlap_intervals_interval_graph(intervals: List[List[int]]) -> int:
    """
    Interval graph approach: Model as graph and find minimum vertex cover.
    
    Time Complexity: O(n^2) - building graph
    Space Complexity: O(n^2) - adjacency matrix
    
    Algorithm:
    1. Build a graph where each interval is a vertex
    2. Add edges between overlapping intervals
    3. Find minimum vertex cover (equivalent to minimum intervals to remove)
    """
    if not intervals:
        return 0
    
    n = len(intervals)
    
    # Build adjacency list for overlap graph
    overlaps = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if intervals i and j overlap
            if intervals[i][1] > intervals[j][0] and intervals[j][1] > intervals[i][0]:
                overlaps[i].append(j)
                overlaps[j].append(i)
    
    # Greedy approach to find minimum vertex cover
    removed = [False] * n
    count = 0
    
    # Sort intervals by degree (number of overlaps) in descending order
    interval_indices = list(range(n))
    interval_indices.sort(key=lambda i: len(overlaps[i]), reverse=True)
    
    for i in interval_indices:
        if removed[i]:
            continue
        
        # If this interval has overlaps, remove it
        if overlaps[i]:
            # Check if any neighbor is not removed
            has_unremoved_neighbor = False
            for j in overlaps[i]:
                if not removed[j]:
                    has_unremoved_neighbor = True
                    break
            
            if has_unremoved_neighbor:
                removed[i] = True
                count += 1
    
    return count


def test_erase_overlap_intervals():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([[1,2],[2,3],[3,4],[1,3]], 1),
        ([[1,2],[1,2],[1,2]], 2),
        ([[1,2],[2,3]], 0),
        
        # Edge cases
        ([[1,2]], 0),
        ([[1,2],[3,4]], 0),
        ([[1,4],[2,3]], 1),
        ([[1,4],[2,3],[3,4]], 1),
        
        # Complex cases
        ([[1,100],[11,22],[1,11],[2,12]], 2),
        ([[-1,1],[0,2],[1,3],[2,4]], 2),
        ([[0,1],[1,2],[1,3],[2,3],[3,4]], 2),
        
        # Large overlaps
        ([[1,10],[2,3],[3,4],[4,5],[5,6]], 0),
        ([[1,10],[2,9],[3,8],[4,7],[5,6]], 4),
        
        # Same start times
        ([[1,2],[1,3],[1,4]], 2),
        ([[1,4],[1,3],[1,2]], 2),
        
        # Same end times
        ([[1,3],[2,3],[3,3]], 1),
        
        # Nested intervals
        ([[1,10],[2,9],[3,8],[4,7]], 3),
        
        # Mixed cases
        ([[-25,-14],[-21,-16],[-20,-13],[-19,-12],[-16,-11],[-14,-3],[-8,-1]], 2),
        ([[1,2],[2,4],[1,3],[3,5],[2,6]], 3),
    ]
    
    # Test all implementations
    implementations = [
        ("Greedy", erase_overlap_intervals_greedy),
        ("DP", erase_overlap_intervals_dp),
        ("Binary Search", erase_overlap_intervals_binary_search),
        ("Activity Selection", erase_overlap_intervals_activity_selection),
        ("Sweep Line", erase_overlap_intervals_sweep_line),
        ("Recursive", erase_overlap_intervals_recursive),
        ("Stack", erase_overlap_intervals_stack),
        ("Interval Graph", erase_overlap_intervals_interval_graph),
    ]
    
    print("Testing Non-overlapping Intervals implementations:")
    print("=" * 60)
    
    for i, (intervals, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {intervals}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                result = func(intervals.copy())  # Use copy to avoid modifying original
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Analysis:")
    print("=" * 60)
    
    import time
    
    # Generate large test case
    large_intervals = [[i, i+2] for i in range(0, 1000, 1)]  # Many overlapping intervals
    
    for name, func in implementations:
        try:
            start_time = time.time()
            result = func(large_intervals.copy())
            end_time = time.time()
            
            print(f"{name}: {result} intervals removed in {end_time - start_time:.4f} seconds")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_erase_overlap_intervals() 