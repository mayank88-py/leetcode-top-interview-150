"""
322. Coin Change

You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

Example 1:
Input: coins = [1,3,4], amount = 6
Output: 2
Explanation: 6 = 3 + 3

Example 2:
Input: coins = [2], amount = 3
Output: -1

Example 3:
Input: coins = [1], amount = 0
Output: 0

Constraints:
- 1 <= coins.length <= 12
- 1 <= coins[i] <= 2^31 - 1
- 0 <= amount <= 10^4
"""

def coin_change_dp_bottom_up(coins, amount):
    """
    Approach 1: Dynamic Programming (Bottom-up)
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    
    dp[i] = minimum coins needed to make amount i
    """
    if amount == 0:
        return 0
    
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_memoization(coins, amount):
    """
    Approach 2: Memoization (Top-down DP)
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    
    Use memoization to avoid recalculating subproblems.
    """
    memo = {}
    
    def min_coins(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        
        if remaining in memo:
            return memo[remaining]
        
        min_count = float('inf')
        for coin in coins:
            count = min_coins(remaining - coin) + 1
            min_count = min(min_count, count)
        
        memo[remaining] = min_count
        return min_count
    
    result = min_coins(amount)
    return result if result != float('inf') else -1


def coin_change_bfs(coins, amount):
    """
    Approach 3: BFS (Level-order traversal)
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    
    Use BFS to find shortest path (minimum coins).
    """
    if amount == 0:
        return 0
    
    from collections import deque
    
    queue = deque([amount])
    visited = set([amount])
    level = 0
    
    while queue:
        level += 1
        size = len(queue)
        
        for _ in range(size):
            current = queue.popleft()
            
            for coin in coins:
                remaining = current - coin
                
                if remaining == 0:
                    return level
                
                if remaining > 0 and remaining not in visited:
                    visited.add(remaining)
                    queue.append(remaining)
    
    return -1


def coin_change_optimized_dp(coins, amount):
    """
    Approach 4: Optimized DP with Early Termination
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    
    Sort coins in descending order for potential early termination.
    """
    if amount == 0:
        return 0
    
    coins.sort(reverse=True)  # Sort coins in descending order
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            if dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_greedy_with_backtrack(coins, amount):
    """
    Approach 5: Greedy with Backtracking
    Time Complexity: O(coins^(amount/min_coin)) worst case
    Space Complexity: O(amount/min_coin)
    
    Try greedy approach with backtracking for optimization.
    """
    coins.sort(reverse=True)
    min_coins = float('inf')
    
    def backtrack(remaining, coin_count, coin_index):
        nonlocal min_coins
        
        if remaining == 0:
            min_coins = min(min_coins, coin_count)
            return
        
        if coin_index >= len(coins) or coin_count >= min_coins:
            return
        
        coin = coins[coin_index]
        # Try different numbers of current coin
        for count in range(remaining // coin, -1, -1):
            if coin_count + count < min_coins:
                backtrack(remaining - coin * count, coin_count + count, coin_index + 1)
    
    backtrack(amount, 0, 0)
    return min_coins if min_coins != float('inf') else -1


def coin_change_space_optimized(coins, amount):
    """
    Approach 6: Space-Optimized DP
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    
    More space-efficient implementation.
    """
    if amount == 0:
        return 0
    
    prev = [float('inf')] * (amount + 1)
    prev[0] = 0
    
    for coin in coins:
        curr = prev[:]
        for i in range(coin, amount + 1):
            if prev[i - coin] != float('inf'):
                curr[i] = min(curr[i], prev[i - coin] + 1)
        prev = curr
    
    return prev[amount] if prev[amount] != float('inf') else -1


def coin_change_iterative_improvement(coins, amount):
    """
    Approach 7: Iterative Improvement
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    
    Keep improving the solution iteratively.
    """
    if amount == 0:
        return 0
    
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    # Process each amount from 1 to target amount
    for target in range(1, amount + 1):
        for coin in coins:
            if coin <= target and dp[target - coin] != float('inf'):
                dp[target] = min(dp[target], dp[target - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_bfs_optimized(coins, amount):
    """
    Approach 8: Optimized BFS
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    
    BFS with optimizations to avoid unnecessary computations.
    """
    if amount == 0:
        return 0
    
    from collections import deque
    
    coins.sort(reverse=True)  # Start with larger coins
    queue = deque([(amount, 0)])  # (remaining_amount, coins_used)
    visited = set([amount])
    
    while queue:
        remaining, coins_used = queue.popleft()
        
        for coin in coins:
            new_remaining = remaining - coin
            
            if new_remaining == 0:
                return coins_used + 1
            
            if new_remaining > 0 and new_remaining not in visited:
                visited.add(new_remaining)
                queue.append((new_remaining, coins_used + 1))
    
    return -1


def coin_change_mathematical_bound(coins, amount):
    """
    Approach 9: DP with Mathematical Bounds
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    
    Use mathematical bounds to optimize the search.
    """
    if amount == 0:
        return 0
    
    max_coin = max(coins)
    min_coin = min(coins)
    
    # Early return if impossible
    if amount < min_coin:
        return -1
    
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(min_coin, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def test_coin_change():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([1, 3, 4], 6, 2),
        ([2], 3, -1),
        ([1], 0, 0),
        ([1], 1, 1),
        ([1], 2, 2),
        ([2, 5, 10, 1], 27, 4),
        ([2, 5, 10], 27, -1),
        ([1, 2, 5], 11, 3),
        ([1, 3, 4], 6, 2),
        ([5], 3, -1),
        ([1, 2, 5], 7, 2),
        ([1, 4, 5], 8, 2),
        ([2, 3, 5], 9, 2),
        ([1, 5, 10, 25], 30, 2),
        ([2, 5], 3, -1),
    ]
    
    approaches = [
        ("DP Bottom-up", coin_change_dp_bottom_up),
        ("Memoization", coin_change_memoization),
        ("BFS", coin_change_bfs),
        ("Optimized DP", coin_change_optimized_dp),
        ("Greedy with Backtrack", coin_change_greedy_with_backtrack),
        ("Space Optimized", coin_change_space_optimized),
        ("Iterative Improvement", coin_change_iterative_improvement),
        ("BFS Optimized", coin_change_bfs_optimized),
        ("Mathematical Bound", coin_change_mathematical_bound),
    ]
    
    for i, (coins, amount, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: coins = {coins}, amount = {amount}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            # Create copy to avoid modifying original
            coins_copy = coins.copy()
            result = func(coins_copy, amount)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_coin_change() 