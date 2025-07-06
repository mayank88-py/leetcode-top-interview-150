"""
134. Gas Station

Problem:
There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.

If there exists a solution, it is guaranteed to be unique.

Example 1:
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
Explanation:
Start at station 3 (index 3) and fill up with 4 units of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.

Example 2:
Input: gas = [2,3,4], cost = [3,4,3]
Output: -1
Explanation:
You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
Let's start at station 2 and fill up with 4 units of gas. Your tank = 0 + 4 = 4
Travel to station 0. Your tank = 4 - 3 + 2 = 3
Travel to station 1. Your tank = 3 - 3 + 3 = 3
Travel to station 2. Your tank = 3 - 4 + 4 = 3
Your tank = 3 < 4, so you can't travel back to station 2.
Therefore, return -1.

Time Complexity: O(n) for optimal solution
Space Complexity: O(1)
"""


def can_complete_circuit(gas, cost):
    """
    Find starting gas station index using one-pass greedy approach.
    
    Args:
        gas: List of gas amounts at each station
        cost: List of costs to travel to next station
    
    Returns:
        Starting station index or -1 if impossible
    """
    n = len(gas)
    total_gas = sum(gas)
    total_cost = sum(cost)
    
    # If total gas is less than total cost, impossible to complete circuit
    if total_gas < total_cost:
        return -1
    
    # Find the starting point
    current_gas = 0
    start = 0
    
    for i in range(n):
        current_gas += gas[i] - cost[i]
        
        # If we can't reach the next station, start from next station
        if current_gas < 0:
            start = i + 1
            current_gas = 0
    
    return start


def can_complete_circuit_brute_force(gas, cost):
    """
    Find starting gas station index using brute force approach.
    
    Args:
        gas: List of gas amounts at each station
        cost: List of costs to travel to next station
    
    Returns:
        Starting station index or -1 if impossible
    """
    n = len(gas)
    
    # Try each starting position
    for start in range(n):
        current_gas = 0
        position = start
        
        # Try to complete the circuit
        for _ in range(n):
            current_gas += gas[position] - cost[position]
            
            # If we can't reach the next station
            if current_gas < 0:
                break
            
            position = (position + 1) % n
        
        # If we completed the circuit successfully
        if current_gas >= 0:
            return start
    
    return -1


def can_complete_circuit_two_pass(gas, cost):
    """
    Find starting gas station index using two-pass approach.
    
    Args:
        gas: List of gas amounts at each station
        cost: List of costs to travel to next station
    
    Returns:
        Starting station index or -1 if impossible
    """
    n = len(gas)
    
    # First pass: check if solution exists
    total_gas = sum(gas)
    total_cost = sum(cost)
    
    if total_gas < total_cost:
        return -1
    
    # Second pass: find the starting position
    current_gas = 0
    start = 0
    
    for i in range(n):
        current_gas += gas[i] - cost[i]
        
        if current_gas < 0:
            start = i + 1
            current_gas = 0
    
    return start


def can_complete_circuit_prefix_sum(gas, cost):
    """
    Find starting gas station index using prefix sum approach.
    
    Args:
        gas: List of gas amounts at each station
        cost: List of costs to travel to next station
    
    Returns:
        Starting station index or -1 if impossible
    """
    n = len(gas)
    
    # Calculate net gas at each station
    net_gas = [gas[i] - cost[i] for i in range(n)]
    
    # Check if total net gas is non-negative
    if sum(net_gas) < 0:
        return -1
    
    # Find starting position where we never go negative
    current_sum = 0
    start = 0
    
    for i in range(n):
        current_sum += net_gas[i]
        
        if current_sum < 0:
            start = i + 1
            current_sum = 0
    
    return start


def can_complete_circuit_simulation(gas, cost):
    """
    Find starting gas station index using simulation approach.
    
    Args:
        gas: List of gas amounts at each station
        cost: List of costs to travel to next station
    
    Returns:
        Starting station index or -1 if impossible
    """
    n = len(gas)
    
    def can_start_from(start):
        """Check if we can complete circuit starting from given position"""
        current_gas = 0
        
        for i in range(n):
            station = (start + i) % n
            current_gas += gas[station] - cost[station]
            
            if current_gas < 0:
                return False
        
        return True
    
    # Try each starting position
    for start in range(n):
        if can_start_from(start):
            return start
    
    return -1


def can_complete_circuit_mathematical(gas, cost):
    """
    Find starting gas station index using mathematical approach.
    
    Args:
        gas: List of gas amounts at each station
        cost: List of costs to travel to next station
    
    Returns:
        Starting station index or -1 if impossible
    """
    n = len(gas)
    
    # Calculate differences
    diff = [gas[i] - cost[i] for i in range(n)]
    
    # If sum of differences is negative, no solution
    if sum(diff) < 0:
        return -1
    
    # Find the position where minimum prefix sum occurs
    min_prefix = 0
    current_sum = 0
    min_index = 0
    
    for i in range(n):
        current_sum += diff[i]
        if current_sum < min_prefix:
            min_prefix = current_sum
            min_index = i
    
    # Starting position is next to the minimum prefix sum position
    return (min_index + 1) % n


# Test cases
if __name__ == "__main__":
    # Test case 1
    gas1 = [1,2,3,4,5]
    cost1 = [3,4,5,1,2]
    result1a = can_complete_circuit(gas1, cost1)
    result1b = can_complete_circuit_brute_force(gas1, cost1)
    result1c = can_complete_circuit_two_pass(gas1, cost1)
    result1d = can_complete_circuit_prefix_sum(gas1, cost1)
    result1e = can_complete_circuit_simulation(gas1, cost1)
    result1f = can_complete_circuit_mathematical(gas1, cost1)
    print(f"Test 1 - Gas: {gas1}, Cost: {cost1}, Expected: 3")
    print(f"Greedy: {result1a}, BruteForce: {result1b}, TwoPass: {result1c}, PrefixSum: {result1d}, Simulation: {result1e}, Mathematical: {result1f}")
    print()
    
    # Test case 2
    gas2 = [2,3,4]
    cost2 = [3,4,3]
    result2a = can_complete_circuit(gas2, cost2)
    result2b = can_complete_circuit_brute_force(gas2, cost2)
    result2c = can_complete_circuit_two_pass(gas2, cost2)
    result2d = can_complete_circuit_prefix_sum(gas2, cost2)
    result2e = can_complete_circuit_simulation(gas2, cost2)
    result2f = can_complete_circuit_mathematical(gas2, cost2)
    print(f"Test 2 - Gas: {gas2}, Cost: {cost2}, Expected: -1")
    print(f"Greedy: {result2a}, BruteForce: {result2b}, TwoPass: {result2c}, PrefixSum: {result2d}, Simulation: {result2e}, Mathematical: {result2f}")
    print()
    
    # Test case 3 - Single station
    gas3 = [1]
    cost3 = [2]
    result3a = can_complete_circuit(gas3, cost3)
    result3b = can_complete_circuit_brute_force(gas3, cost3)
    result3c = can_complete_circuit_two_pass(gas3, cost3)
    result3d = can_complete_circuit_prefix_sum(gas3, cost3)
    result3e = can_complete_circuit_simulation(gas3, cost3)
    result3f = can_complete_circuit_mathematical(gas3, cost3)
    print(f"Test 3 - Gas: {gas3}, Cost: {cost3}, Expected: -1")
    print(f"Greedy: {result3a}, BruteForce: {result3b}, TwoPass: {result3c}, PrefixSum: {result3d}, Simulation: {result3e}, Mathematical: {result3f}")
    print()
    
    # Test case 4 - Single station possible
    gas4 = [2]
    cost4 = [1]
    result4a = can_complete_circuit(gas4, cost4)
    result4b = can_complete_circuit_brute_force(gas4, cost4)
    result4c = can_complete_circuit_two_pass(gas4, cost4)
    result4d = can_complete_circuit_prefix_sum(gas4, cost4)
    result4e = can_complete_circuit_simulation(gas4, cost4)
    result4f = can_complete_circuit_mathematical(gas4, cost4)
    print(f"Test 4 - Gas: {gas4}, Cost: {cost4}, Expected: 0")
    print(f"Greedy: {result4a}, BruteForce: {result4b}, TwoPass: {result4c}, PrefixSum: {result4d}, Simulation: {result4e}, Mathematical: {result4f}")
    print()
    
    # Test case 5 - All equal
    gas5 = [3,3,3,3]
    cost5 = [3,3,3,3]
    result5a = can_complete_circuit(gas5, cost5)
    result5b = can_complete_circuit_brute_force(gas5, cost5)
    result5c = can_complete_circuit_two_pass(gas5, cost5)
    result5d = can_complete_circuit_prefix_sum(gas5, cost5)
    result5e = can_complete_circuit_simulation(gas5, cost5)
    result5f = can_complete_circuit_mathematical(gas5, cost5)
    print(f"Test 5 - Gas: {gas5}, Cost: {cost5}, Expected: 0")
    print(f"Greedy: {result5a}, BruteForce: {result5b}, TwoPass: {result5c}, PrefixSum: {result5d}, Simulation: {result5e}, Mathematical: {result5f}")
    print()
    
    # Test case 6 - Start from middle
    gas6 = [1,2,3,4,5]
    cost6 = [2,3,4,5,1]
    result6a = can_complete_circuit(gas6, cost6)
    result6b = can_complete_circuit_brute_force(gas6, cost6)
    result6c = can_complete_circuit_two_pass(gas6, cost6)
    result6d = can_complete_circuit_prefix_sum(gas6, cost6)
    result6e = can_complete_circuit_simulation(gas6, cost6)
    result6f = can_complete_circuit_mathematical(gas6, cost6)
    print(f"Test 6 - Gas: {gas6}, Cost: {cost6}, Expected: 4")
    print(f"Greedy: {result6a}, BruteForce: {result6b}, TwoPass: {result6c}, PrefixSum: {result6d}, Simulation: {result6e}, Mathematical: {result6f}")
    print()
    
    # Test case 7 - Large difference
    gas7 = [5,1,2,3,4]
    cost7 = [4,4,1,5,1]
    result7a = can_complete_circuit(gas7, cost7)
    result7b = can_complete_circuit_brute_force(gas7, cost7)
    result7c = can_complete_circuit_two_pass(gas7, cost7)
    result7d = can_complete_circuit_prefix_sum(gas7, cost7)
    result7e = can_complete_circuit_simulation(gas7, cost7)
    result7f = can_complete_circuit_mathematical(gas7, cost7)
    print(f"Test 7 - Gas: {gas7}, Cost: {cost7}, Expected: 4")
    print(f"Greedy: {result7a}, BruteForce: {result7b}, TwoPass: {result7c}, PrefixSum: {result7d}, Simulation: {result7e}, Mathematical: {result7f}")
    print()
    
    # Test case 8 - Two stations
    gas8 = [3,1]
    cost8 = [1,2]
    result8a = can_complete_circuit(gas8, cost8)
    result8b = can_complete_circuit_brute_force(gas8, cost8)
    result8c = can_complete_circuit_two_pass(gas8, cost8)
    result8d = can_complete_circuit_prefix_sum(gas8, cost8)
    result8e = can_complete_circuit_simulation(gas8, cost8)
    result8f = can_complete_circuit_mathematical(gas8, cost8)
    print(f"Test 8 - Gas: {gas8}, Cost: {cost8}, Expected: 0")
    print(f"Greedy: {result8a}, BruteForce: {result8b}, TwoPass: {result8c}, PrefixSum: {result8d}, Simulation: {result8e}, Mathematical: {result8f}")
    print()
    
    # Test case 9 - Barely possible
    gas9 = [1,2,3,4,5]
    cost9 = [3,4,5,1,3]
    result9a = can_complete_circuit(gas9, cost9)
    result9b = can_complete_circuit_brute_force(gas9, cost9)
    result9c = can_complete_circuit_two_pass(gas9, cost9)
    result9d = can_complete_circuit_prefix_sum(gas9, cost9)
    result9e = can_complete_circuit_simulation(gas9, cost9)
    result9f = can_complete_circuit_mathematical(gas9, cost9)
    print(f"Test 9 - Gas: {gas9}, Cost: {cost9}, Expected: -1")
    print(f"Greedy: {result9a}, BruteForce: {result9b}, TwoPass: {result9c}, PrefixSum: {result9d}, Simulation: {result9e}, Mathematical: {result9f}")
    print()
    
    # Test case 10 - Start from first
    gas10 = [2,3,4,5,1]
    cost10 = [1,2,3,4,5]
    result10a = can_complete_circuit(gas10, cost10)
    result10b = can_complete_circuit_brute_force(gas10, cost10)
    result10c = can_complete_circuit_two_pass(gas10, cost10)
    result10d = can_complete_circuit_prefix_sum(gas10, cost10)
    result10e = can_complete_circuit_simulation(gas10, cost10)
    result10f = can_complete_circuit_mathematical(gas10, cost10)
    print(f"Test 10 - Gas: {gas10}, Cost: {cost10}, Expected: 0")
    print(f"Greedy: {result10a}, BruteForce: {result10b}, TwoPass: {result10c}, PrefixSum: {result10d}, Simulation: {result10e}, Mathematical: {result10f}") 