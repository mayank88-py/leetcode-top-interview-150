"""
123. Best Time to Buy and Sell Stock III

You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete at most two transactions.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

Example 1:
Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.

Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.

Example 3:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.

Constraints:
- 1 <= prices.length <= 10^5
- 0 <= prices[i] <= 10^5
"""

def max_profit_2d_dp(prices):
    """
    Approach 1: 2D Dynamic Programming
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    dp[i][j] = maximum profit on day i with at most j transactions
    """
    if not prices or len(prices) < 2:
        return 0
    
    n = len(prices)
    k = 2  # At most 2 transactions
    
    # dp[i][t] = max profit up to day i with at most t transactions
    dp = [[0] * (k + 1) for _ in range(n)]
    
    for t in range(1, k + 1):
        max_diff = -prices[0]  # Maximum difference of buying stock
        for i in range(1, n):
            dp[i][t] = max(dp[i-1][t], prices[i] + max_diff)
            max_diff = max(max_diff, dp[i-1][t-1] - prices[i])
    
    return dp[n-1][k]


def max_profit_state_machine(prices):
    """
    Approach 2: State Machine DP
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Track four states: first buy, first sell, second buy, second sell
    """
    if not prices or len(prices) < 2:
        return 0
    
    # Initialize states
    first_buy = -prices[0]   # After first purchase
    first_sell = 0           # After first sale
    second_buy = -prices[0]  # After second purchase
    second_sell = 0          # After second sale
    
    for i in range(1, len(prices)):
        price = prices[i]
        
        # Update states in reverse order to avoid using updated values
        second_sell = max(second_sell, second_buy + price)
        second_buy = max(second_buy, first_sell - price)
        first_sell = max(first_sell, first_buy + price)
        first_buy = max(first_buy, -price)
    
    return second_sell


def max_profit_forward_backward(prices):
    """
    Approach 3: Forward-Backward DP
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Split into two parts: max profit from left and max profit from right
    """
    if not prices or len(prices) < 2:
        return 0
    
    n = len(prices)
    
    # Forward pass: max profit with one transaction up to day i
    left_profits = [0] * n
    min_price = prices[0]
    
    for i in range(1, n):
        min_price = min(min_price, prices[i])
        left_profits[i] = max(left_profits[i-1], prices[i] - min_price)
    
    # Backward pass: max profit with one transaction from day i
    right_profits = [0] * n
    max_price = prices[n-1]
    
    for i in range(n-2, -1, -1):
        max_price = max(max_price, prices[i])
        right_profits[i] = max(right_profits[i+1], max_price - prices[i])
    
    # Combine results
    max_profit = 0
    for i in range(n):
        max_profit = max(max_profit, left_profits[i] + right_profits[i])
    
    return max_profit


def max_profit_general_k_transactions(prices):
    """
    Approach 4: General K Transactions (K=2)
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    General solution for at most k transactions, specialized for k=2
    """
    if not prices or len(prices) < 2:
        return 0
    
    k = 2
    n = len(prices)
    
    # If k >= n//2, we can do as many transactions as we want
    if k >= n // 2:
        profit = 0
        for i in range(1, n):
            if prices[i] > prices[i-1]:
                profit += prices[i] - prices[i-1]
        return profit
    
    # buy[i] = max profit after buying for transaction i
    # sell[i] = max profit after selling for transaction i
    buy = [-prices[0]] * k
    sell = [0] * k
    
    for i in range(1, n):
        for j in range(k-1, -1, -1):
            sell[j] = max(sell[j], buy[j] + prices[i])
            buy[j] = max(buy[j], (sell[j-1] if j > 0 else 0) - prices[i])
    
    return sell[k-1]


def max_profit_optimized_space(prices):
    """
    Approach 5: Space Optimized
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Optimized version using only necessary variables
    """
    if not prices or len(prices) < 2:
        return 0
    
    # Four variables for the four states
    buy1 = buy2 = -prices[0]
    sell1 = sell2 = 0
    
    for price in prices[1:]:
        sell2 = max(sell2, buy2 + price)
        buy2 = max(buy2, sell1 - price)
        sell1 = max(sell1, buy1 + price)
        buy1 = max(buy1, -price)
    
    return sell2


def max_profit_memoization(prices):
    """
    Approach 6: Top-down DP with Memoization
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Use memoization for recursive solution
    """
    if not prices or len(prices) < 2:
        return 0
    
    memo = {}
    
    def max_profit_helper(day, transactions_left, holding):
        if day >= len(prices) or transactions_left == 0:
            return 0
        
        if (day, transactions_left, holding) in memo:
            return memo[(day, transactions_left, holding)]
        
        # Option 1: Do nothing
        result = max_profit_helper(day + 1, transactions_left, holding)
        
        if holding:
            # Option 2: Sell (complete a transaction)
            result = max(result, prices[day] + max_profit_helper(day + 1, transactions_left - 1, False))
        else:
            # Option 2: Buy
            result = max(result, -prices[day] + max_profit_helper(day + 1, transactions_left, True))
        
        memo[(day, transactions_left, holding)] = result
        return result
    
    return max_profit_helper(0, 2, False)


def max_profit_rolling_array(prices):
    """
    Approach 7: Rolling Array
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Use rolling array technique
    """
    if not prices or len(prices) < 2:
        return 0
    
    n = len(prices)
    
    # Use arrays of size 3 (for 0, 1, 2 transactions)
    buy = [-float('inf')] * 3
    sell = [0] * 3
    
    buy[0] = buy[1] = buy[2] = -prices[0]
    
    for i in range(1, n):
        for k in range(2, 0, -1):
            sell[k] = max(sell[k], buy[k] + prices[i])
            buy[k] = max(buy[k], sell[k-1] - prices[i])
    
    return sell[2]


def max_profit_finite_state_machine(prices):
    """
    Approach 8: Finite State Machine
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Model as a finite state machine with explicit states
    """
    if not prices or len(prices) < 2:
        return 0
    
    # States: S0 (initial), S1 (bought once), S2 (sold once), S3 (bought twice), S4 (sold twice)
    s0 = 0
    s1 = -prices[0]
    s2 = 0
    s3 = -prices[0]
    s4 = 0
    
    for i in range(1, len(prices)):
        price = prices[i]
        
        # Transition in reverse order
        s4 = max(s4, s3 + price)  # Sell second stock
        s3 = max(s3, s2 - price)  # Buy second stock
        s2 = max(s2, s1 + price)  # Sell first stock
        s1 = max(s1, s0 - price)  # Buy first stock
        # s0 remains 0
    
    return s4


def max_profit_segment_approach(prices):
    """
    Approach 9: Segment-based Approach
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Find optimal split point for two transactions
    """
    if not prices or len(prices) < 2:
        return 0
    
    n = len(prices)
    
    # Calculate max profit for single transaction ending at or before each day
    max_profit_before = [0] * n
    min_price = prices[0]
    
    for i in range(1, n):
        min_price = min(min_price, prices[i])
        max_profit_before[i] = max(max_profit_before[i-1], prices[i] - min_price)
    
    # Calculate max profit for single transaction starting at or after each day
    max_profit_after = [0] * n
    max_price = prices[n-1]
    
    for i in range(n-2, -1, -1):
        max_price = max(max_price, prices[i])
        max_profit_after[i] = max(max_profit_after[i+1], max_price - prices[i])
    
    # Find the best split point
    max_total_profit = 0
    for i in range(n):
        total_profit = max_profit_before[i] + max_profit_after[i]
        max_total_profit = max(max_total_profit, total_profit)
    
    return max_total_profit


def test_max_profit():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([3, 3, 5, 0, 0, 3, 1, 4], 6),
        ([1, 2, 3, 4, 5], 4),
        ([7, 6, 4, 3, 1], 0),
        ([1], 0),
        ([1, 2], 1),
        ([2, 1], 0),
        ([1, 4, 2], 3),
        ([2, 4, 1], 2),
        ([3, 2, 6, 5, 0, 3], 7),
        ([1, 2, 4, 2, 5, 7, 2, 4, 9, 0], 13),
        ([6, 1, 3, 2, 4, 7], 7),
    ]
    
    approaches = [
        ("2D DP", max_profit_2d_dp),
        ("State Machine", max_profit_state_machine),
        ("Forward-Backward", max_profit_forward_backward),
        ("General K", max_profit_general_k_transactions),
        ("Optimized Space", max_profit_optimized_space),
        ("Memoization", max_profit_memoization),
        ("Rolling Array", max_profit_rolling_array),
        ("FSM", max_profit_finite_state_machine),
        ("Segment", max_profit_segment_approach),
    ]
    
    for i, (prices, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {prices}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(prices)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_max_profit() 