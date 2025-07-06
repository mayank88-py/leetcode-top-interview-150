"""
39. Combination Sum

Problem:
Given an array of distinct integers candidates and a target integer target, 
return a list of all unique combinations of candidates where the chosen numbers sum to target. 
You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. 
Two combinations are unique if the frequency of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to target 
is less than 150 combinations for the given input.

Example 1:
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]

Example 2:
Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]

Example 3:
Input: candidates = [2], target = 1
Output: []

Time Complexity: O(N^(T/M)) where N = len(candidates), T = target, M = minimal value in candidates
Space Complexity: O(T/M) for recursion depth
"""


def combination_sum_backtrack(candidates, target):
    """
    Backtracking approach - optimal solution.
    
    Time Complexity: O(N^(T/M)) in worst case
    Space Complexity: O(T/M) for recursion stack
    
    Algorithm:
    1. Use backtracking to explore all possible combinations
    2. At each step, try each candidate from current index onwards
    3. Allow reusing same candidate by not advancing index
    4. Prune when current sum exceeds target
    """
    result = []
    
    def backtrack(remaining, combination, start):
        # Base case: found valid combination
        if remaining == 0:
            result.append(combination[:])  # Make a copy
            return
        
        # Prune: if remaining is negative, stop exploring
        if remaining < 0:
            return
        
        # Try each candidate starting from 'start' index
        for i in range(start, len(candidates)):
            candidate = candidates[i]
            
            # Include current candidate
            combination.append(candidate)
            
            # Recurse with same start index (allows reuse)
            backtrack(remaining - candidate, combination, i)
            
            # Backtrack
            combination.pop()
    
    backtrack(target, [], 0)
    return result


def combination_sum_optimized(candidates, target):
    """
    Optimized backtracking with sorting.
    
    Time Complexity: O(N^(T/M)) but with better pruning
    Space Complexity: O(T/M)
    
    Algorithm:
    1. Sort candidates for better pruning
    2. Skip candidates that are too large early
    3. Use same backtracking approach but with optimization
    """
    candidates.sort()  # Sort for better pruning
    result = []
    
    def backtrack(remaining, combination, start):
        if remaining == 0:
            result.append(combination[:])
            return
        
        for i in range(start, len(candidates)):
            candidate = candidates[i]
            
            # Early termination: if current candidate > remaining, 
            # all subsequent candidates will also be > remaining
            if candidate > remaining:
                break
            
            combination.append(candidate)
            backtrack(remaining - candidate, combination, i)
            combination.pop()
    
    backtrack(target, [], 0)
    return result


def combination_sum_iterative(candidates, target):
    """
    Iterative approach using stack/queue.
    
    Time Complexity: O(N^(T/M))
    Space Complexity: O(N^(T/M)) for storing all states
    
    Algorithm:
    1. Use stack to store (current_sum, combination, start_index)
    2. Process each state and generate next valid states
    3. Add to result when target is reached
    """
    result = []
    # Stack stores (remaining, combination, start_index)
    stack = [(target, [], 0)]
    
    while stack:
        remaining, combination, start = stack.pop()
        
        if remaining == 0:
            result.append(combination)
            continue
        
        if remaining < 0:
            continue
        
        for i in range(start, len(candidates)):
            candidate = candidates[i]
            if candidate > remaining:
                continue
            
            # Create new state
            new_combination = combination + [candidate]
            stack.append((remaining - candidate, new_combination, i))
    
    return result


def combination_sum_dp(candidates, target):
    """
    Dynamic programming approach.
    
    Time Complexity: O(T * N * average_combination_length)
    Space Complexity: O(T * total_combinations)
    
    Algorithm:
    1. dp[i] stores all combinations that sum to i
    2. For each target value, try adding each candidate
    3. Build combinations incrementally
    """
    # dp[i] will store all combinations that sum to i
    dp = [[] for _ in range(target + 1)]
    dp[0] = [[]]  # One way to make 0: empty combination
    
    for i in range(1, target + 1):
        for candidate in candidates:
            if candidate <= i:
                # For each combination that sums to (i - candidate)
                for combination in dp[i - candidate]:
                    # Add current candidate to form new combination
                    new_combination = combination + [candidate]
                    # Sort to avoid duplicates (since we want unique combinations)
                    new_combination.sort()
                    if new_combination not in dp[i]:
                        dp[i].append(new_combination)
    
    return dp[target]


def combination_sum_memoized(candidates, target):
    """
    Recursive approach with memoization.
    
    Time Complexity: O(N^(T/M)) but with memoization benefits
    Space Complexity: O(T * combinations_count)
    
    Algorithm:
    1. Use recursion with memoization on (remaining, start_index)
    2. Cache results to avoid recomputation
    3. Build combinations recursively
    """
    memo = {}
    
    def helper(remaining, start):
        if (remaining, start) in memo:
            return memo[(remaining, start)]
        
        if remaining == 0:
            return [[]]
        
        if remaining < 0 or start >= len(candidates):
            return []
        
        result = []
        
        # Try each candidate from start index onwards
        for i in range(start, len(candidates)):
            candidate = candidates[i]
            if candidate > remaining:
                continue
            
            # Get all combinations for remaining - candidate
            sub_combinations = helper(remaining - candidate, i)
            
            # Add current candidate to each sub-combination
            for sub_combo in sub_combinations:
                result.append([candidate] + sub_combo)
        
        memo[(remaining, start)] = result
        return result
    
    return helper(target, 0)


def test_combination_sum():
    """Test all implementations with various test cases."""
    
    test_cases = [
        ([2,3,6,7], 7, [[2,2,3],[7]]),
        ([2,3,5], 8, [[2,2,2,2],[2,3,3],[3,5]]),
        ([2], 1, []),
        ([1], 1, [[1]]),
        ([1], 2, [[1,1]]),
        ([2,3,4], 6, [[2,2,2],[2,4],[3,3]]),
        ([7,3,2], 18, [[2,2,2,2,2,2,2,2,2],[2,2,2,2,2,2,3,3],[2,2,2,2,3,7],[2,2,2,3,3,3,3],[2,2,3,3,3,7],[2,3,3,3,7],[3,3,3,3,3,3]])
    ]
    
    implementations = [
        ("Backtracking", combination_sum_backtrack),
        ("Optimized Backtrack", combination_sum_optimized),
        ("Iterative", combination_sum_iterative),
        ("Dynamic Programming", combination_sum_dp),
        ("Memoized Recursive", combination_sum_memoized)
    ]
    
    print("Testing Combination Sum...")
    
    for i, (candidates, target, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: candidates = {candidates}, target = {target}")
        print(f"Expected: {expected}")
        
        for impl_name, impl_func in implementations:
            # Make a copy since some implementations might modify the input
            candidates_copy = candidates[:]
            result = impl_func(candidates_copy, target)
            
            # Sort each combination and then sort the list for comparison
            result_sorted = [sorted(combo) for combo in result]
            result_sorted.sort()
            expected_sorted = [sorted(combo) for combo in expected]
            expected_sorted.sort()
            
            is_correct = result_sorted == expected_sorted
            print(f"{impl_name:20} | Count: {len(result):2} | {'✓' if is_correct else '✗'}")
            
            if not is_correct and len(result) <= 20:
                print(f"                       Got: {result_sorted}")


if __name__ == "__main__":
    test_combination_sum() 