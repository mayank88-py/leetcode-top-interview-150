"""
46. Permutations

Problem:
Given an array nums of distinct integers, return all the possible permutations. 
You can return the answer in any order.

Example 1:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Example 2:
Input: nums = [0,1]
Output: [[0,1],[1,0]]

Example 3:
Input: nums = [1]
Output: [[1]]

Time Complexity: O(N × N!) where N is the length of nums
Space Complexity: O(N!) for storing all permutations
"""


def permute_backtrack(nums):
    """
    Backtracking approach - most intuitive.
    
    Time Complexity: O(N × N!) where N is the length of nums
    Space Complexity: O(N!) for result + O(N) for recursion stack
    
    Algorithm:
    1. Use backtracking to build permutations
    2. Track used elements with boolean array
    3. Add element to path, recurse, then remove (backtrack)
    4. When path length equals nums length, add to result
    """
    result = []
    
    def backtrack(path):
        # Base case: permutation is complete
        if len(path) == len(nums):
            result.append(path[:])  # Make a copy
            return
        
        # Try each number
        for num in nums:
            if num not in path:  # Skip if already used
                path.append(num)
                backtrack(path)
                path.pop()  # Backtrack
    
    backtrack([])
    return result


def permute_backtrack_optimized(nums):
    """
    Optimized backtracking with used array.
    
    Time Complexity: O(N × N!)
    Space Complexity: O(N!) for result + O(N) for recursion stack
    
    Algorithm:
    1. Use boolean array to track used elements (faster than 'in' check)
    2. Same backtracking logic but with O(1) used element check
    """
    result = []
    used = [False] * len(nums)
    
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if not used[i]:
                path.append(nums[i])
                used[i] = True
                backtrack(path)
                path.pop()
                used[i] = False
    
    backtrack([])
    return result


def permute_swap(nums):
    """
    Backtracking with swapping approach.
    
    Time Complexity: O(N × N!)
    Space Complexity: O(N!) for result + O(N) for recursion stack
    
    Algorithm:
    1. Swap elements to generate permutations
    2. At each level, swap current element with all elements after it
    3. Recursively permute the rest
    4. Swap back to restore original array (backtrack)
    """
    result = []
    
    def backtrack(start):
        # Base case: reached end of array
        if start == len(nums):
            result.append(nums[:])  # Make a copy
            return
        
        for i in range(start, len(nums)):
            # Swap current element with element at start position
            nums[start], nums[i] = nums[i], nums[start]
            
            # Recursively permute the rest
            backtrack(start + 1)
            
            # Swap back (backtrack)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result


def permute_iterative(nums):
    """
    Iterative approach using queue/list.
    
    Time Complexity: O(N × N!)
    Space Complexity: O(N × N!) for storing intermediate permutations
    
    Algorithm:
    1. Start with empty permutation
    2. For each number, insert it at all possible positions in existing permutations
    3. Continue until all numbers are used
    """
    permutations = [[]]
    
    for num in nums:
        new_permutations = []
        
        for perm in permutations:
            # Insert num at each possible position
            for i in range(len(perm) + 1):
                new_perm = perm[:i] + [num] + perm[i:]
                new_permutations.append(new_perm)
        
        permutations = new_permutations
    
    return permutations


def permute_recursive(nums):
    """
    Pure recursive approach without explicit backtracking.
    
    Time Complexity: O(N × N!)
    Space Complexity: O(N × N!)
    
    Algorithm:
    1. For each element, make it the first element
    2. Recursively get permutations of remaining elements
    3. Prepend the first element to each permutation
    """
    if len(nums) <= 1:
        return [nums]
    
    result = []
    
    for i in range(len(nums)):
        # Choose current element as first
        first = nums[i]
        remaining = nums[:i] + nums[i+1:]
        
        # Get all permutations of remaining elements
        for perm in permute_recursive(remaining):
            result.append([first] + perm)
    
    return result


def permute_heap_algorithm(nums):
    """
    Heap's algorithm for generating permutations.
    
    Time Complexity: O(N!)
    Space Complexity: O(N!) for result + O(N) for recursion stack
    
    Algorithm:
    1. Heap's algorithm generates all permutations with minimal changes
    2. For odd n, swap first and last elements
    3. For even n, swap ith and last elements
    4. Recursively apply to n-1 elements
    """
    result = []
    
    def heap_permute(k):
        if k == 1:
            result.append(nums[:])
            return
        
        heap_permute(k - 1)
        
        for i in range(k - 1):
            if k % 2 == 0:
                # k is even: swap ith element with last
                nums[i], nums[k - 1] = nums[k - 1], nums[i]
            else:
                # k is odd: swap first element with last
                nums[0], nums[k - 1] = nums[k - 1], nums[0]
            
            heap_permute(k - 1)
    
    heap_permute(len(nums))
    return result


def permute_lexicographic(nums):
    """
    Generate permutations in lexicographic order.
    
    Time Complexity: O(N × N!)
    Space Complexity: O(N!)
    
    Algorithm:
    1. Start with sorted array
    2. Find next lexicographic permutation until no more exist
    3. Use standard next permutation algorithm
    """
    def next_permutation(arr):
        # Find the largest index i such that arr[i] < arr[i + 1]
        i = len(arr) - 2
        while i >= 0 and arr[i] >= arr[i + 1]:
            i -= 1
        
        if i == -1:
            return False  # No next permutation
        
        # Find the largest index j such that arr[i] < arr[j]
        j = len(arr) - 1
        while arr[j] <= arr[i]:
            j -= 1
        
        # Swap arr[i] and arr[j]
        arr[i], arr[j] = arr[j], arr[i]
        
        # Reverse the suffix starting at arr[i + 1]
        arr[i + 1:] = reversed(arr[i + 1:])
        return True
    
    # Start with sorted array
    nums.sort()
    result = [nums[:]]
    
    # Generate all permutations
    while next_permutation(nums):
        result.append(nums[:])
    
    return result


def test_permutations():
    """Test all implementations with various test cases."""
    
    test_cases = [
        ([1,2,3], [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]),
        ([0,1], [[0,1],[1,0]]),
        ([1], [[1]]),
        ([1,2], [[1,2],[2,1]]),
        ([1,2,3,4], None)  # Too many to list, just check count
    ]
    
    implementations = [
        ("Backtracking", permute_backtrack),
        ("Optimized Backtrack", permute_backtrack_optimized),
        ("Swap Method", permute_swap),
        ("Iterative", permute_iterative),
        ("Pure Recursive", permute_recursive),
        ("Heap Algorithm", permute_heap_algorithm),
        ("Lexicographic", permute_lexicographic)
    ]
    
    print("Testing Permutations...")
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        
        if expected:
            print(f"Expected count: {len(expected)}")
        else:
            # Calculate expected count using factorial
            import math
            expected_count = math.factorial(len(nums))
            print(f"Expected count: {expected_count}")
        
        for impl_name, impl_func in implementations:
            # Make a copy since some implementations modify the input
            nums_copy = nums[:]
            result = impl_func(nums_copy)
            
            if expected:
                # Sort both for comparison since order may vary
                result_sorted = [sorted(perm) for perm in result]
                result_sorted.sort()
                expected_sorted = [sorted(perm) for perm in expected]
                expected_sorted.sort()
                
                # Check if permutations match (regardless of order)
                is_correct = len(result) == len(expected) and all(
                    sorted(result) == sorted(expected) or
                    set(tuple(p) for p in result) == set(tuple(p) for p in expected)
                )
                
                print(f"{impl_name:20} | Count: {len(result):2} | {'✓' if is_correct else '✗'}")
            else:
                import math
                expected_count = math.factorial(len(nums))
                is_correct = len(result) == expected_count
                print(f"{impl_name:20} | Count: {len(result):2} | {'✓' if is_correct else '✗'}")


if __name__ == "__main__":
    test_permutations() 