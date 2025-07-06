"""
909. Snakes and Ladders

You are given an n x n integer matrix board where the cells are labeled from 1 to n² in a Boustrophedon style starting from the bottom left of the board (i.e. board[n - 1][0]) and alternating direction each row.

You start on square 1 of the board. In each move, you starting from square curr, do the following:

- Choose a destination square next with a label in the range [curr + 1, curr + 6].
- If next has a snake or ladder, you must move to the destination of that snake or ladder. Otherwise, you move to next.
- The game ends when you reach the square n².

Given the integer matrix board, return the least number of moves required to reach the square n² from square 1. If it is not possible to reach the destination, return -1.

Example 1:
Input: board = [[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1]]
Output: 2
Explanation: In the first move, move from square 1 to square 6.
In the second move, move from square 6 to square 36.

Example 2:
Input: board = [[-1,-1],[-1,30],[-1,-1],[-1,-1]]
Output: -1
Explanation: You cannot reach square 4 from square 1 because square 3 is the foot of a ladder that goes up to square 30.

Constraints:
- n == board.length == board[i].length
- 2 <= n <= 20
- board[i][j] is either -1 or in the range [1, n²].
- The squares labeled 1 and n² do not have any ladders or snakes.
"""

from typing import List
from collections import deque


def snakes_and_ladders_bfs(board: List[List[int]]) -> int:
    """
    Standard BFS approach (optimal solution).
    
    Time Complexity: O(n^2) where n is the board dimension
    Space Complexity: O(n^2) - queue and visited set
    
    Algorithm:
    1. Convert 2D board to 1D for easier navigation
    2. Use BFS to find shortest path from square 1 to n²
    3. Handle snakes and ladders during exploration
    """
    n = len(board)
    target = n * n
    
    def get_coordinates(square):
        """Convert square number to board coordinates."""
        square -= 1  # Convert to 0-indexed
        row = n - 1 - square // n
        col = square % n
        
        # Handle boustrophedon (zigzag) pattern
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        return row, col
    
    def get_destination(square):
        """Get final destination considering snakes/ladders."""
        row, col = get_coordinates(square)
        if board[row][col] != -1:
            return board[row][col]
        return square
    
    # BFS
    queue = deque([(1, 0)])  # (square, moves)
    visited = {1}
    
    while queue:
        current_square, moves = queue.popleft()
        
        if current_square == target:
            return moves
        
        # Try all possible dice rolls (1-6)
        for dice in range(1, 7):
            next_square = current_square + dice
            
            if next_square > target:
                break
            
            # Handle snakes and ladders
            final_square = get_destination(next_square)
            
            if final_square not in visited:
                visited.add(final_square)
                queue.append((final_square, moves + 1))
    
    return -1


def snakes_and_ladders_level_order_bfs(board: List[List[int]]) -> int:
    """
    Level-order BFS approach.
    
    Time Complexity: O(n^2) where n is the board dimension
    Space Complexity: O(n^2) - level sets and visited set
    
    Algorithm:
    1. Process squares level by level
    2. Each level represents moves with same number of dice rolls
    3. Clear separation between move counts
    """
    n = len(board)
    target = n * n
    
    def get_coordinates(square):
        """Convert square number to board coordinates."""
        square -= 1
        row = n - 1 - square // n
        col = square % n
        
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        return row, col
    
    def get_destination(square):
        """Get final destination considering snakes/ladders."""
        row, col = get_coordinates(square)
        if board[row][col] != -1:
            return board[row][col]
        return square
    
    current_level = {1}
    visited = {1}
    moves = 0
    
    while current_level:
        next_level = set()
        
        for square in current_level:
            if square == target:
                return moves
            
            # Try all dice rolls
            for dice in range(1, 7):
                next_square = square + dice
                
                if next_square > target:
                    break
                
                final_square = get_destination(next_square)
                
                if final_square not in visited:
                    visited.add(final_square)
                    next_level.add(final_square)
        
        current_level = next_level
        moves += 1
    
    return -1


def snakes_and_ladders_optimized_bfs(board: List[List[int]]) -> int:
    """
    Optimized BFS with early termination.
    
    Time Complexity: O(n^2) where n is the board dimension
    Space Complexity: O(n^2) - queue and visited set
    
    Algorithm:
    1. Use BFS with optimizations for early termination
    2. Skip impossible moves early
    3. Optimized coordinate calculation
    """
    n = len(board)
    target = n * n
    
    # Pre-compute destinations for all squares
    destinations = [0] * (target + 1)
    for square in range(1, target + 1):
        square_idx = square - 1
        row = n - 1 - square_idx // n
        col = square_idx % n
        
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        if board[row][col] != -1:
            destinations[square] = board[row][col]
        else:
            destinations[square] = square
    
    # BFS with optimizations
    queue = deque([(1, 0)])
    visited = [False] * (target + 1)
    visited[1] = True
    
    while queue:
        current_square, moves = queue.popleft()
        
        # Try all dice rolls
        for dice in range(1, 7):
            next_square = current_square + dice
            
            if next_square > target:
                break
            
            final_square = destinations[next_square]
            
            if final_square == target:
                return moves + 1
            
            if not visited[final_square]:
                visited[final_square] = True
                queue.append((final_square, moves + 1))
    
    return -1


def snakes_and_ladders_bidirectional_bfs(board: List[List[int]]) -> int:
    """
    Bidirectional BFS approach.
    
    Time Complexity: O(n^2) where n is the board dimension
    Space Complexity: O(n^2) - two queues and visited sets
    
    Algorithm:
    1. Search from start (square 1) and end (square n²) simultaneously
    2. Meet in the middle to find shortest path
    3. Can be faster for some board configurations
    """
    n = len(board)
    target = n * n
    
    def get_coordinates(square):
        square -= 1
        row = n - 1 - square // n
        col = square % n
        
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        return row, col
    
    def get_destination(square):
        row, col = get_coordinates(square)
        if board[row][col] != -1:
            return board[row][col]
        return square
    
    # Forward search from start
    forward_queue = deque([(1, 0)])
    forward_visited = {1: 0}
    
    # Backward search from end (simulate reverse moves)
    backward_queue = deque([(target, 0)])
    backward_visited = {target: 0}
    
    def get_reverse_moves(square):
        """Get squares that can reach this square in one move."""
        reverse_moves = []
        for prev_square in range(max(1, square - 6), square):
            if get_destination(prev_square) == square:
                reverse_moves.append(prev_square)
        return reverse_moves
    
    while forward_queue or backward_queue:
        # Forward search
        if forward_queue:
            current_square, moves = forward_queue.popleft()
            
            if current_square in backward_visited:
                return moves + backward_visited[current_square]
            
            for dice in range(1, 7):
                next_square = current_square + dice
                if next_square > target:
                    break
                
                final_square = get_destination(next_square)
                
                if final_square not in forward_visited:
                    forward_visited[final_square] = moves + 1
                    forward_queue.append((final_square, moves + 1))
        
        # Backward search
        if backward_queue:
            current_square, moves = backward_queue.popleft()
            
            if current_square in forward_visited:
                return moves + forward_visited[current_square]
            
            for prev_square in get_reverse_moves(current_square):
                if prev_square not in backward_visited:
                    backward_visited[prev_square] = moves + 1
                    backward_queue.append((prev_square, moves + 1))
    
    return -1


def snakes_and_ladders_dijkstra(board: List[List[int]]) -> int:
    """
    Dijkstra's algorithm approach (overkill for this problem).
    
    Time Complexity: O(n^2 log n) where n is the board dimension
    Space Complexity: O(n^2) - priority queue and distance array
    
    Algorithm:
    1. Use Dijkstra's algorithm to find shortest path
    2. All edges have weight 1, so BFS is more efficient
    3. Demonstrates alternative graph algorithm
    """
    import heapq
    
    n = len(board)
    target = n * n
    
    def get_coordinates(square):
        square -= 1
        row = n - 1 - square // n
        col = square % n
        
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        return row, col
    
    def get_destination(square):
        row, col = get_coordinates(square)
        if board[row][col] != -1:
            return board[row][col]
        return square
    
    # Dijkstra's algorithm
    dist = [float('inf')] * (target + 1)
    dist[1] = 0
    heap = [(0, 1)]
    
    while heap:
        current_dist, current_square = heapq.heappop(heap)
        
        if current_square == target:
            return current_dist
        
        if current_dist > dist[current_square]:
            continue
        
        for dice in range(1, 7):
            next_square = current_square + dice
            if next_square > target:
                break
            
            final_square = get_destination(next_square)
            new_dist = current_dist + 1
            
            if new_dist < dist[final_square]:
                dist[final_square] = new_dist
                heapq.heappush(heap, (new_dist, final_square))
    
    return -1


def snakes_and_ladders_dfs_with_memo(board: List[List[int]]) -> int:
    """
    DFS with memoization approach (not optimal for shortest path).
    
    Time Complexity: O(n^2) where n is the board dimension
    Space Complexity: O(n^2) - memoization cache + recursion stack
    
    Algorithm:
    1. Use DFS with memoization to find shortest path
    2. Not optimal for shortest path problems but shows alternative
    3. Can lead to stack overflow for large boards
    """
    n = len(board)
    target = n * n
    
    def get_coordinates(square):
        square -= 1
        row = n - 1 - square // n
        col = square % n
        
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        return row, col
    
    def get_destination(square):
        row, col = get_coordinates(square)
        if board[row][col] != -1:
            return board[row][col]
        return square
    
    memo = {}
    
    def dfs(square):
        if square == target:
            return 0
        
        if square in memo:
            return memo[square]
        
        min_moves = float('inf')
        
        for dice in range(1, 7):
            next_square = square + dice
            if next_square > target:
                break
            
            final_square = get_destination(next_square)
            result = dfs(final_square)
            
            if result != float('inf'):
                min_moves = min(min_moves, result + 1)
        
        memo[square] = min_moves
        return min_moves
    
    result = dfs(1)
    return result if result != float('inf') else -1


def snakes_and_ladders_a_star(board: List[List[int]]) -> int:
    """
    A* search approach with heuristic.
    
    Time Complexity: O(n^2 log n) where n is the board dimension
    Space Complexity: O(n^2) - priority queue and data structures
    
    Algorithm:
    1. Use A* with heuristic (distance to target / 6)
    2. Priority queue for optimal path exploration
    3. Can be faster than BFS in some cases
    """
    import heapq
    
    n = len(board)
    target = n * n
    
    def get_coordinates(square):
        square -= 1
        row = n - 1 - square // n
        col = square % n
        
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        return row, col
    
    def get_destination(square):
        row, col = get_coordinates(square)
        if board[row][col] != -1:
            return board[row][col]
        return square
    
    def heuristic(square):
        """Minimum moves needed if no snakes/ladders."""
        return (target - square + 5) // 6  # Ceiling division
    
    # A* search
    heap = [(heuristic(1), 0, 1)]  # (f_score, g_score, square)
    g_score = {1: 0}
    
    while heap:
        f, g, current_square = heapq.heappop(heap)
        
        if current_square == target:
            return g
        
        if g > g_score.get(current_square, float('inf')):
            continue
        
        for dice in range(1, 7):
            next_square = current_square + dice
            if next_square > target:
                break
            
            final_square = get_destination(next_square)
            tentative_g = g + 1
            
            if tentative_g < g_score.get(final_square, float('inf')):
                g_score[final_square] = tentative_g
                f_score = tentative_g + heuristic(final_square)
                heapq.heappush(heap, (f_score, tentative_g, final_square))
    
    return -1


# Test cases
def test_snakes_and_ladders():
    """Test all snakes and ladders approaches."""
    
    # Test cases
    test_cases = [
        (
            [[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1]],
            2,
            "No snakes or ladders"
        ),
        (
            [[-1,-1],[-1,30],[-1,-1],[-1,-1]],
            -1,
            "Impossible due to ladder"
        ),
        (
            [[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,35,-1,-1,13,-1],[-1,-1,-1,-1,-1,-1],[-1,15,-1,-1,-1,-1]],
            4,
            "Mixed snakes and ladders"
        ),
        (
            [[-1,-1,-1],[-1,9,-1],[-1,8,9]],
            1,
            "Small board with ladder"
        ),
        (
            [[-1,-1,-1,-1],
             [5,-1,-1,-1],
             [-1,-1,-1,-1],
             [-1,-1,-1,-1]],
            3,
            "4x4 board with one snake"
        ),
        (
            [[-1,4,-1],
             [6,2,-1],
             [-1,3,-1]],
            2,
            "3x3 board with mixed"
        ),
    ]
    
    # Test all approaches
    approaches = [
        ("Standard BFS", snakes_and_ladders_bfs),
        ("Level-order BFS", snakes_and_ladders_level_order_bfs),
        ("Optimized BFS", snakes_and_ladders_optimized_bfs),
        ("Bidirectional BFS", snakes_and_ladders_bidirectional_bfs),
        ("Dijkstra", snakes_and_ladders_dijkstra),
        ("DFS with Memo", snakes_and_ladders_dfs_with_memo),
        ("A* Search", snakes_and_ladders_a_star),
    ]
    
    print("Testing snakes and ladders approaches:")
    print("=" * 50)
    
    for i, (board, expected, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        print(f"Board size: {len(board)}x{len(board[0])}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            try:
                result = func([row[:] for row in board])  # Pass copy to avoid modification
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    # Create larger test case for performance testing
    def create_large_board(n):
        """Create a large board with random snakes and ladders."""
        import random
        
        board = [[-1] * n for _ in range(n)]
        
        # Add some random snakes and ladders
        total_squares = n * n
        num_snakes_ladders = total_squares // 10
        
        for _ in range(num_snakes_ladders):
            # Random position for snake/ladder
            row = random.randint(0, n - 1)
            col = random.randint(0, n - 1)
            
            # Don't place on first or last square
            square_num = (n - 1 - row) * n + col + 1
            if (n - 1 - row) % 2 == 1:
                square_num = (n - 1 - row) * n + (n - 1 - col) + 1
            
            if square_num != 1 and square_num != total_squares:
                # Random destination
                destination = random.randint(1, total_squares)
                if destination != square_num:
                    board[row][col] = destination
        
        return board
    
    large_board = create_large_board(10)
    
    import time
    
    print(f"Testing with large board (10x10):")
    for name, func in approaches:
        try:
            start_time = time.time()
            result = func([row[:] for row in large_board])
            end_time = time.time()
            print(f"{name}: {result} (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_snakes_and_ladders() 