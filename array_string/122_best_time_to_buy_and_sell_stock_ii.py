"""
122. Best Time to Buy and Sell Stock II

Problem:
You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time.
However, you can buy it then immediately sell it on the same day.

Find and return the maximum profit you can achieve.

Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.

Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Total profit is 4.

Example 3:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: There is no way to make a positive profit, so we never buy the stock to achieve the maximum profit of 0.

Time Complexity: O(n)
Space Complexity: O(1)
"""


def max_profit(prices):
    """
    Find maximum profit with multiple transactions allowed.
    
    Args:
        prices: List of stock prices
    
    Returns:
        Maximum profit possible
    """
    if not prices or len(prices) < 2:
        return 0
    
    max_profit = 0
    
    # Add profit for every consecutive increasing pair
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            max_profit += prices[i] - prices[i - 1]
    
    return max_profit


def max_profit_peak_valley(prices):
    """
    Find maximum profit using peak and valley approach.
    
    Args:
        prices: List of stock prices
    
    Returns:
        Maximum profit possible
    """
    if not prices or len(prices) < 2:
        return 0
    
    i = 0
    max_profit = 0
    
    while i < len(prices) - 1:
        # Find valley (local minimum)
        while i < len(prices) - 1 and prices[i + 1] <= prices[i]:
            i += 1
        
        if i == len(prices) - 1:
            break
        
        valley = prices[i]
        
        # Find peak (local maximum)
        while i < len(prices) - 1 and prices[i + 1] >= prices[i]:
            i += 1
        
        peak = prices[i]
        max_profit += peak - valley
    
    return max_profit


# Test cases
if __name__ == "__main__":
    # Test case 1
    prices1 = [7, 1, 5, 3, 6, 4]
    result1a = max_profit(prices1)
    result1b = max_profit_peak_valley(prices1)
    print(f"Test 1 - Expected: 7, Greedy: {result1a}, Peak-Valley: {result1b}")
    
    # Test case 2
    prices2 = [1, 2, 3, 4, 5]
    result2a = max_profit(prices2)
    result2b = max_profit_peak_valley(prices2)
    print(f"Test 2 - Expected: 4, Greedy: {result2a}, Peak-Valley: {result2b}")
    
    # Test case 3
    prices3 = [7, 6, 4, 3, 1]
    result3a = max_profit(prices3)
    result3b = max_profit_peak_valley(prices3)
    print(f"Test 3 - Expected: 0, Greedy: {result3a}, Peak-Valley: {result3b}")
    
    # Test case 4 - Single element
    prices4 = [1]
    result4a = max_profit(prices4)
    result4b = max_profit_peak_valley(prices4)
    print(f"Test 4 - Expected: 0, Greedy: {result4a}, Peak-Valley: {result4b}")
    
    # Test case 5 - Two elements increasing
    prices5 = [1, 5]
    result5a = max_profit(prices5)
    result5b = max_profit_peak_valley(prices5)
    print(f"Test 5 - Expected: 4, Greedy: {result5a}, Peak-Valley: {result5b}")
    
    # Test case 6 - Two elements decreasing
    prices6 = [5, 1]
    result6a = max_profit(prices6)
    result6b = max_profit_peak_valley(prices6)
    print(f"Test 6 - Expected: 0, Greedy: {result6a}, Peak-Valley: {result6b}")
    
    # Test case 7 - Up and down pattern
    prices7 = [1, 3, 2, 4, 1, 5]
    result7a = max_profit(prices7)
    result7b = max_profit_peak_valley(prices7)
    print(f"Test 7 - Expected: 8, Greedy: {result7a}, Peak-Valley: {result7b}") 