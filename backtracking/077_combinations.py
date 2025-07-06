"""
77. Combinations

Problem:
Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n].

You may return the answer in any order.

Example 1:
Input: n = 4, k = 2
Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]

Example 2:
Input: n = 1, k = 1
Output: [[1]]

Time Complexity: O(C(n,k) * k) where C(n,k) is binomial coefficient
Space Complexity: O(C(n,k) * k) for storing all combinations
"""


def combine_backtrack(n, k):
    """
    Backtracking approach - most intuitive.
    
    Time Complexity: O(C(n,k) * k)
    Space Complexity: O(C(n,k) * k) for result + O(k) for recursion stack
    
    Algorithm:
    1. Use backtracking to build combinations of size k
    2. Start from number 1 and try each subsequent number
    3. Maintain start index to avoid duplicates
    4. When combination size reaches k, add to result
    """
    result = []
    
    def backtrack(start, current_combination):
        # Base case: combination is complete
        if len(current_combination) == k:
            result.append(current_combination[:])  # Make a copy
            return
        
        # Try each number from start to n
        for i in range(start, n + 1):
            current_combination.append(i)
            backtrack(i + 1, current_combination)
            current_combination.pop()  # Backtrack
    
    backtrack(1, [])
    return result


def combine_optimized(n, k):
    """
    Optimized backtracking with pruning.
    
    Time Complexity: O(C(n,k) * k)
    Space Complexity: O(C(n,k) * k)
    
    Algorithm:
    1. Add pruning to skip impossible branches
    2. If remaining numbers < needed numbers, skip
    3. needed = k - len(current_combination)
    4. remaining = n - start + 1
    """
    result = []
    
    def backtrack(start, current_combination):
        if len(current_combination) == k:
            result.append(current_combination[:])
            return
        
        # Pruning: if remaining numbers < needed numbers, skip
        needed = k - len(current_combination)
        remaining = n - start + 1
        
        if remaining < needed:
            return
        
        for i in range(start, n + 1):
            current_combination.append(i)
            backtrack(i + 1, current_combination)
            current_combination.pop()
    
    backtrack(1, [])
    return result


def combine_iterative(n, k):
    """
    Iterative approach using queue.
    
    Time Complexity: O(C(n,k) * k)
    Space Complexity: O(C(n,k) * k)
    
    Algorithm:
    1. Use queue to store partial combinations
    2. For each partial combination, extend with next possible numbers
    3. Continue until all combinations have size k
    """
    from collections import deque
    
    if k == 0:
        return [[]]
    
    # Queue stores (current_combination, next_start)
    queue = deque([([], 1)])
    result = []
    
    while queue:
        current_combination, start = queue.popleft()
        
        if len(current_combination) == k:
            result.append(current_combination)
            continue
        
        # Add all possible next numbers
        for i in range(start, n + 1):
            new_combination = current_combination + [i]
            
            # Pruning: check if we can still complete the combination
            needed = k - len(new_combination)
            remaining = n - i
            
            if remaining >= needed - 1:  # -1 because we already added i
                queue.append((new_combination, i + 1))
    
    return result


def combine_recursive(n, k):
    """
    Pure recursive approach.
    
    Time Complexity: O(C(n,k) * k)
    Space Complexity: O(C(n,k) * k)
    
    Algorithm:
    1. For each number, decide to include it or not
    2. If included, recursively solve for (n-1, k-1)
    3. If not included, recursively solve for (n-1, k)
    4. Combine results from both choices
    """
    if k == 0:
        return [[]]
    if k > n:
        return []
    if k == n:
        return [list(range(1, n + 1))]
    
    # Include current number n
    with_n = []
    for combo in combine_recursive(n - 1, k - 1):
        with_n.append(combo + [n])
    
    # Exclude current number n
    without_n = combine_recursive(n - 1, k)
    
    return with_n + without_n


def combine_itertools(n, k):
    """
    Using itertools.combinations (Pythonic approach).
    
    Time Complexity: O(C(n,k) * k)
    Space Complexity: O(C(n,k) * k)
    
    Algorithm:
    1. Use built-in itertools.combinations
    2. Convert tuples to lists for output format
    """
    from itertools import combinations
    
    return [list(combo) for combo in combinations(range(1, n + 1), k)]


def combine_lexicographic(n, k):
    """
    Generate combinations in lexicographic order.
    
    Time Complexity: O(C(n,k) * k)
    Space Complexity: O(C(n,k) * k)
    
    Algorithm:
    1. Start with first lexicographic combination [1,2,...,k]
    2. Generate next combination using standard algorithm
    3. Continue until no more combinations exist
    """
    if k == 0:
        return [[]]
    
    def next_combination(combo):
        # Find rightmost element that can be incremented
        i = k - 1
        while i >= 0 and combo[i] == n - k + i + 1:
            i -= 1
        
        if i < 0:
            return None  # No more combinations
        
        # Create next combination
        next_combo = combo[:]
        next_combo[i] += 1
        
        # Reset all elements to the right
        for j in range(i + 1, k):
            next_combo[j] = next_combo[i] + (j - i)
        
        return next_combo
    
    result = []
    current = list(range(1, k + 1))  # First combination [1,2,...,k]
    
    while current:
        result.append(current[:])
        current = next_combination(current)
    
    return result


def combine_bit_manipulation(n, k):
    """
    Generate combinations using bit manipulation.
    
    Time Complexity: O(2^n * k) but only valid combinations processed
    Space Complexity: O(C(n,k) * k)
    
    Algorithm:
    1. Iterate through all possible bit patterns
    2. Count set bits - if equals k, it's a valid combination
    3. Extract numbers corresponding to set bits
    """
    result = []
    
    # Iterate through all possible subsets
    for mask in range(1 << n):
        if bin(mask).count('1') == k:
            combination = []
            for i in range(n):
                if mask & (1 << i):
                    combination.append(i + 1)
            result.append(combination)
    
    return result


def test_combinations():
    """Test all implementations with various test cases."""
    
    test_cases = [
        (4, 2, [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]),
        (1, 1, [[1]]),
        (3, 3, [[1,2,3]]),
        (5, 1, [[1],[2],[3],[4],[5]]),
        (4, 0, [[]]),
        (5, 3, [[1,2,3],[1,2,4],[1,2,5],[1,3,4],[1,3,5],[1,4,5],[2,3,4],[2,3,5],[2,4,5],[3,4,5]])
    ]
    
    implementations = [
        ("Backtracking", combine_backtrack),
        ("Optimized", combine_optimized),
        ("Iterative", combine_iterative),
        ("Recursive", combine_recursive),
        ("Itertools", combine_itertools),
        ("Lexicographic", combine_lexicographic),
        ("Bit Manipulation", combine_bit_manipulation)
    ]
    
    print("Testing Combinations...")
    
    for i, (n, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: n = {n}, k = {k}")
        print(f"Expected: {expected}")
        
        for impl_name, impl_func in implementations:
            result = impl_func(n, k)
            
            # Sort for comparison since order may vary
            result_sorted = [sorted(combo) for combo in result]
            result_sorted.sort()
            expected_sorted = [sorted(combo) for combo in expected]
            expected_sorted.sort()
            
            is_correct = result_sorted == expected_sorted
            print(f"{impl_name:17} | Count: {len(result):2} | {'✓' if is_correct else '✗'}")
            
            if not is_correct and len(result) <= 20:
                print(f"                    Got: {result_sorted}")


if __name__ == "__main__":
    test_combinations() 