"""
135. Candy

Problem:
There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.

You are giving candies to these children subjected to the following requirements:
- Each child must receive at least one candy.
- Children with a higher rating get more candies than their neighbors with lower ratings.

Return the minimum number of candies you need to have to distribute the candies to the children.

Example 1:
Input: ratings = [1,0,2]
Output: 5
Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.

Example 2:
Input: ratings = [1,2,2]
Output: 4
Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.
The third child gets 1 candy because it satisfies the above two conditions.

Time Complexity: O(n) for optimal solution
Space Complexity: O(n) for the candies array
"""


def candy(ratings):
    """
    Calculate minimum candies using two-pass approach.
    
    Args:
        ratings: List of ratings for each child
    
    Returns:
        Minimum number of candies needed
    """
    n = len(ratings)
    if n == 0:
        return 0
    
    candies = [1] * n
    
    # Left to right pass - ensure right neighbor with higher rating gets more candy
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    
    # Right to left pass - ensure left neighbor with higher rating gets more candy
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    
    return sum(candies)


def candy_single_pass(ratings):
    """
    Calculate minimum candies using single pass with counting.
    
    Args:
        ratings: List of ratings for each child
    
    Returns:
        Minimum number of candies needed
    """
    n = len(ratings)
    if n <= 1:
        return n
    
    total = 1  # First child gets 1 candy
    up = 0     # Length of increasing sequence
    down = 0   # Length of decreasing sequence
    peak = 0   # Height of the peak
    
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            # Increasing sequence
            down = 0
            up += 1
            peak = up
            total += 1 + up
        elif ratings[i] == ratings[i-1]:
            # Equal ratings
            up = down = peak = 0
            total += 1
        else:
            # Decreasing sequence
            up = 0
            down += 1
            total += 1 + down
            # If peak is not tall enough, we need to add extra candy
            if peak >= down:
                total -= 1
    
    return total


def candy_brute_force(ratings):
    """
    Calculate minimum candies using brute force approach.
    
    Args:
        ratings: List of ratings for each child
    
    Returns:
        Minimum number of candies needed
    """
    n = len(ratings)
    if n == 0:
        return 0
    
    candies = [1] * n
    
    # Keep adjusting until all conditions are satisfied
    changed = True
    while changed:
        changed = False
        
        for i in range(n):
            # Check left neighbor
            if i > 0 and ratings[i] > ratings[i-1] and candies[i] <= candies[i-1]:
                candies[i] = candies[i-1] + 1
                changed = True
            
            # Check right neighbor
            if i < n-1 and ratings[i] > ratings[i+1] and candies[i] <= candies[i+1]:
                candies[i] = candies[i+1] + 1
                changed = True
    
    return sum(candies)


def candy_recursion(ratings):
    """
    Calculate minimum candies using recursion with memoization.
    
    Args:
        ratings: List of ratings for each child
    
    Returns:
        Minimum number of candies needed
    """
    n = len(ratings)
    if n == 0:
        return 0
    
    memo = {}
    
    def get_candy(i):
        """Get minimum candies for child i"""
        if i in memo:
            return memo[i]
        
        candies = 1  # At least 1 candy
        
        # Check left neighbor
        if i > 0 and ratings[i] > ratings[i-1]:
            candies = max(candies, get_candy(i-1) + 1)
        
        # Check right neighbor
        if i < n-1 and ratings[i] > ratings[i+1]:
            candies = max(candies, get_candy(i+1) + 1)
        
        memo[i] = candies
        return candies
    
    total = 0
    for i in range(n):
        total += get_candy(i)
    
    return total


def candy_slope_counting(ratings):
    """
    Calculate minimum candies by counting slopes.
    
    Args:
        ratings: List of ratings for each child
    
    Returns:
        Minimum number of candies needed
    """
    n = len(ratings)
    if n <= 1:
        return n
    
    def count_candies(length):
        """Calculate sum 1+2+...+length"""
        return length * (length + 1) // 2
    
    total = 1
    i = 1
    
    while i < n:
        if ratings[i] == ratings[i-1]:
            total += 1
            i += 1
        else:
            # Count increasing slope
            peak = 0
            while i < n and ratings[i] > ratings[i-1]:
                peak += 1
                total += 1 + peak
                i += 1
            
            # Count decreasing slope
            valley = 0
            while i < n and ratings[i] < ratings[i-1]:
                valley += 1
                total += 1 + valley
                i += 1
            
            # Adjust for peak if necessary
            if valley > peak:
                total += valley - peak
    
    return total


def candy_stack_based(ratings):
    """
    Calculate minimum candies using stack-based approach.
    
    Args:
        ratings: List of ratings for each child
    
    Returns:
        Minimum number of candies needed
    """
    n = len(ratings)
    if n <= 1:
        return n
    
    candies = [1] * n
    stack = []
    
    for i in range(n):
        # Process decreasing sequence
        while stack and ratings[stack[-1]] > ratings[i]:
            j = stack.pop()
            # Calculate candies based on position in decreasing sequence
            rank = len(stack) + 1
            candies[j] = max(candies[j], rank)
        
        stack.append(i)
    
    # Process remaining elements in stack (increasing sequence)
    while stack:
        j = stack.pop()
        rank = len(stack) + 1
        candies[j] = max(candies[j], rank)
    
    return sum(candies)


# Test cases
if __name__ == "__main__":
    # Test case 1
    ratings1 = [1,0,2]
    result1a = candy(ratings1)
    result1b = candy_single_pass(ratings1)
    result1c = candy_brute_force(ratings1)
    result1d = candy_recursion(ratings1)
    result1e = candy_slope_counting(ratings1)
    result1f = candy_stack_based(ratings1)
    print(f"Test 1 - Ratings: {ratings1}, Expected: 5")
    print(f"TwoPass: {result1a}, SinglePass: {result1b}, BruteForce: {result1c}, Recursion: {result1d}, SlopeCounting: {result1e}, StackBased: {result1f}")
    print()
    
    # Test case 2
    ratings2 = [1,2,2]
    result2a = candy(ratings2)
    result2b = candy_single_pass(ratings2)
    result2c = candy_brute_force(ratings2)
    result2d = candy_recursion(ratings2)
    result2e = candy_slope_counting(ratings2)
    result2f = candy_stack_based(ratings2)
    print(f"Test 2 - Ratings: {ratings2}, Expected: 4")
    print(f"TwoPass: {result2a}, SinglePass: {result2b}, BruteForce: {result2c}, Recursion: {result2d}, SlopeCounting: {result2e}, StackBased: {result2f}")
    print()
    
    # Test case 3 - Single child
    ratings3 = [1]
    result3a = candy(ratings3)
    result3b = candy_single_pass(ratings3)
    result3c = candy_brute_force(ratings3)
    result3d = candy_recursion(ratings3)
    result3e = candy_slope_counting(ratings3)
    result3f = candy_stack_based(ratings3)
    print(f"Test 3 - Ratings: {ratings3}, Expected: 1")
    print(f"TwoPass: {result3a}, SinglePass: {result3b}, BruteForce: {result3c}, Recursion: {result3d}, SlopeCounting: {result3e}, StackBased: {result3f}")
    print()
    
    # Test case 4 - Strictly increasing
    ratings4 = [1,2,3,4,5]
    result4a = candy(ratings4)
    result4b = candy_single_pass(ratings4)
    result4c = candy_brute_force(ratings4)
    result4d = candy_recursion(ratings4)
    result4e = candy_slope_counting(ratings4)
    result4f = candy_stack_based(ratings4)
    print(f"Test 4 - Ratings: {ratings4}, Expected: 15")
    print(f"TwoPass: {result4a}, SinglePass: {result4b}, BruteForce: {result4c}, Recursion: {result4d}, SlopeCounting: {result4e}, StackBased: {result4f}")
    print()
    
    # Test case 5 - Strictly decreasing
    ratings5 = [5,4,3,2,1]
    result5a = candy(ratings5)
    result5b = candy_single_pass(ratings5)
    result5c = candy_brute_force(ratings5)
    result5d = candy_recursion(ratings5)
    result5e = candy_slope_counting(ratings5)
    result5f = candy_stack_based(ratings5)
    print(f"Test 5 - Ratings: {ratings5}, Expected: 15")
    print(f"TwoPass: {result5a}, SinglePass: {result5b}, BruteForce: {result5c}, Recursion: {result5d}, SlopeCounting: {result5e}, StackBased: {result5f}")
    print()
    
    # Test case 6 - All same
    ratings6 = [1,1,1,1]
    result6a = candy(ratings6)
    result6b = candy_single_pass(ratings6)
    result6c = candy_brute_force(ratings6)
    result6d = candy_recursion(ratings6)
    result6e = candy_slope_counting(ratings6)
    result6f = candy_stack_based(ratings6)
    print(f"Test 6 - Ratings: {ratings6}, Expected: 4")
    print(f"TwoPass: {result6a}, SinglePass: {result6b}, BruteForce: {result6c}, Recursion: {result6d}, SlopeCounting: {result6e}, StackBased: {result6f}")
    print()
    
    # Test case 7 - Peak in middle
    ratings7 = [1,3,2,2,1]
    result7a = candy(ratings7)
    result7b = candy_single_pass(ratings7)
    result7c = candy_brute_force(ratings7)
    result7d = candy_recursion(ratings7)
    result7e = candy_slope_counting(ratings7)
    result7f = candy_stack_based(ratings7)
    print(f"Test 7 - Ratings: {ratings7}, Expected: 7")
    print(f"TwoPass: {result7a}, SinglePass: {result7b}, BruteForce: {result7c}, Recursion: {result7d}, SlopeCounting: {result7e}, StackBased: {result7f}")
    print()
    
    # Test case 8 - Valley in middle
    ratings8 = [3,1,3]
    result8a = candy(ratings8)
    result8b = candy_single_pass(ratings8)
    result8c = candy_brute_force(ratings8)
    result8d = candy_recursion(ratings8)
    result8e = candy_slope_counting(ratings8)
    result8f = candy_stack_based(ratings8)
    print(f"Test 8 - Ratings: {ratings8}, Expected: 5")
    print(f"TwoPass: {result8a}, SinglePass: {result8b}, BruteForce: {result8c}, Recursion: {result8d}, SlopeCounting: {result8e}, StackBased: {result8f}")
    print()
    
    # Test case 9 - Complex pattern
    ratings9 = [1,3,4,5,2]
    result9a = candy(ratings9)
    result9b = candy_single_pass(ratings9)
    result9c = candy_brute_force(ratings9)
    result9d = candy_recursion(ratings9)
    result9e = candy_slope_counting(ratings9)
    result9f = candy_stack_based(ratings9)
    print(f"Test 9 - Ratings: {ratings9}, Expected: 11")
    print(f"TwoPass: {result9a}, SinglePass: {result9b}, BruteForce: {result9c}, Recursion: {result9d}, SlopeCounting: {result9e}, StackBased: {result9f}")
    print()
    
    # Test case 10 - Two children
    ratings10 = [1,2]
    result10a = candy(ratings10)
    result10b = candy_single_pass(ratings10)
    result10c = candy_brute_force(ratings10)
    result10d = candy_recursion(ratings10)
    result10e = candy_slope_counting(ratings10)
    result10f = candy_stack_based(ratings10)
    print(f"Test 10 - Ratings: {ratings10}, Expected: 3")
    print(f"TwoPass: {result10a}, SinglePass: {result10b}, BruteForce: {result10c}, Recursion: {result10d}, SlopeCounting: {result10e}, StackBased: {result10f}") 