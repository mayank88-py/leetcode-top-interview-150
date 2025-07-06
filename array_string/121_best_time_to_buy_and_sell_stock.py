"""
121. Best Time to Buy and Sell Stock

Problem:
You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing
a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.

Example 2:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.

Time Complexity: O(n)
Space Complexity: O(1)
"""


def max_profit(prices):
    """
    Find maximum profit from buying and selling stock once.
    
    Args:
        prices: List of stock prices
    
    Returns:
        Maximum profit possible
    """
    if not prices or len(prices) < 2:
        return 0
    
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        # Update minimum price seen so far
        min_price = min(min_price, price)
        # Calculate profit if we sell at current price
        profit = price - min_price
        # Update maximum profit
        max_profit = max(max_profit, profit)
    
    return max_profit


def max_profit_brute_force(prices):
    """
    Brute force approach to find maximum profit.
    
    Args:
        prices: List of stock prices
    
    Returns:
        Maximum profit possible
    """
    if not prices or len(prices) < 2:
        return 0
    
    max_profit = 0
    
    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            profit = prices[j] - prices[i]
            max_profit = max(max_profit, profit)
    
    return max_profit


# Test cases
if __name__ == "__main__":
    # Test case 1
    prices1 = [7, 1, 5, 3, 6, 4]
    result1 = max_profit(prices1)
    result1_bf = max_profit_brute_force(prices1)
    print(f"Test 1 - Expected: 5, Optimal: {result1}, Brute Force: {result1_bf}")
    
    # Test case 2
    prices2 = [7, 6, 4, 3, 1]
    result2 = max_profit(prices2)
    result2_bf = max_profit_brute_force(prices2)
    print(f"Test 2 - Expected: 0, Optimal: {result2}, Brute Force: {result2_bf}")
    
    # Test case 3 - Single element
    prices3 = [1]
    result3 = max_profit(prices3)
    result3_bf = max_profit_brute_force(prices3)
    print(f"Test 3 - Expected: 0, Optimal: {result3}, Brute Force: {result3_bf}")
    
    # Test case 4 - Empty array
    prices4 = []
    result4 = max_profit(prices4)
    result4_bf = max_profit_brute_force(prices4)
    print(f"Test 4 - Expected: 0, Optimal: {result4}, Brute Force: {result4_bf}")
    
    # Test case 5 - Two elements
    prices5 = [1, 5]
    result5 = max_profit(prices5)
    result5_bf = max_profit_brute_force(prices5)
    print(f"Test 5 - Expected: 4, Optimal: {result5}, Brute Force: {result5_bf}")
    
    # Test case 6 - All prices increasing
    prices6 = [1, 2, 3, 4, 5]
    result6 = max_profit(prices6)
    result6_bf = max_profit_brute_force(prices6)
    print(f"Test 6 - Expected: 4, Optimal: {result6}, Brute Force: {result6_bf}")
    
    # Test case 7 - All prices decreasing
    prices7 = [5, 4, 3, 2, 1]
    result7 = max_profit(prices7)
    result7_bf = max_profit_brute_force(prices7)
    print(f"Test 7 - Expected: 0, Optimal: {result7}, Brute Force: {result7_bf}") 