"""
289. Game of Life

Problem:
Given a board with m by n cells, each cell has an initial state live (1) or dead (0). 
Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules:

1. Any live cell with fewer than two live neighbors dies (under-population).
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies (over-population).
4. Any dead cell with exactly three live neighbors becomes a live cell (reproduction).

Write a function to compute the next state (after one update) of the board given its current state.

Follow up:
- Could you solve it in-place? Remember that the board needs to be updated simultaneously.
- Could you solve it with O(1) memory (excluding the input board)?

Example 1:
Input: board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
Output: [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]

Time Complexity: O(m*n) where m and n are board dimensions
Space Complexity: O(1) for in-place solution
"""


def game_of_life(board):
    """
    Update board in-place using state encoding.
    
    States encoding:
    - 0: dead -> dead
    - 1: live -> live
    - 2: live -> dead
    - 3: dead -> live
    
    Args:
        board: 2D list representing the board (modified in-place)
    """
    if not board or not board[0]:
        return
    
    m, n = len(board), len(board[0])
    
    # Directions for 8 neighbors
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    def count_live_neighbors(row, col):
        """Count live neighbors for a cell"""
        count = 0
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < m and 0 <= nc < n:
                # Count original live cells (1 or 2)
                if board[nr][nc] == 1 or board[nr][nc] == 2:
                    count += 1
        return count
    
    # First pass: encode next state
    for i in range(m):
        for j in range(n):
            live_neighbors = count_live_neighbors(i, j)
            
            if board[i][j] == 1:  # Currently live
                if live_neighbors < 2 or live_neighbors > 3:
                    board[i][j] = 2  # Live -> Dead
                # else: stays live (board[i][j] = 1)
            else:  # Currently dead
                if live_neighbors == 3:
                    board[i][j] = 3  # Dead -> Live
                # else: stays dead (board[i][j] = 0)
    
    # Second pass: decode final state
    for i in range(m):
        for j in range(n):
            board[i][j] = board[i][j] % 2


def game_of_life_extra_space(board):
    """
    Update board using extra space for copy.
    
    Args:
        board: 2D list representing the board (modified in-place)
    """
    if not board or not board[0]:
        return
    
    m, n = len(board), len(board[0])
    
    # Create a copy of the board
    copy_board = [row[:] for row in board]
    
    # Directions for 8 neighbors
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    def count_live_neighbors(row, col):
        """Count live neighbors for a cell"""
        count = 0
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < m and 0 <= nc < n:
                count += copy_board[nr][nc]
        return count
    
    # Update board based on rules
    for i in range(m):
        for j in range(n):
            live_neighbors = count_live_neighbors(i, j)
            
            if copy_board[i][j] == 1:  # Live cell
                if live_neighbors < 2 or live_neighbors > 3:
                    board[i][j] = 0  # Dies
                # else: stays alive
            else:  # Dead cell
                if live_neighbors == 3:
                    board[i][j] = 1  # Becomes alive


def game_of_life_bit_manipulation(board):
    """
    Update board using bit manipulation (current state in bit 0, next state in bit 1).
    
    Args:
        board: 2D list representing the board (modified in-place)
    """
    if not board or not board[0]:
        return
    
    m, n = len(board), len(board[0])
    
    # Directions for 8 neighbors
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    def count_live_neighbors(row, col):
        """Count live neighbors for a cell"""
        count = 0
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < m and 0 <= nc < n:
                # Check current state (bit 0)
                count += board[nr][nc] & 1
        return count
    
    # First pass: calculate next state and store in bit 1
    for i in range(m):
        for j in range(n):
            live_neighbors = count_live_neighbors(i, j)
            
            # Current state is in bit 0
            if board[i][j] & 1:  # Currently live
                if live_neighbors == 2 or live_neighbors == 3:
                    board[i][j] |= 2  # Set bit 1 (next state = live)
            else:  # Currently dead
                if live_neighbors == 3:
                    board[i][j] |= 2  # Set bit 1 (next state = live)
    
    # Second pass: shift next state to current state
    for i in range(m):
        for j in range(n):
            board[i][j] >>= 1


def game_of_life_functional(board):
    """
    Update board using functional approach.
    
    Args:
        board: 2D list representing the board (modified in-place)
    """
    if not board or not board[0]:
        return
    
    m, n = len(board), len(board[0])
    
    def get_neighbors(row, col):
        """Get all valid neighbor positions"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < m and 0 <= nc < n:
                    neighbors.append((nr, nc))
        return neighbors
    
    def apply_rules(current_state, live_neighbors):
        """Apply Game of Life rules"""
        if current_state == 1:  # Live cell
            if live_neighbors < 2 or live_neighbors > 3:
                return 0  # Dies
            else:
                return 1  # Survives
        else:  # Dead cell
            if live_neighbors == 3:
                return 1  # Becomes alive
            else:
                return 0  # Stays dead
    
    # Calculate next state for each cell
    next_board = []
    for i in range(m):
        row = []
        for j in range(n):
            neighbors = get_neighbors(i, j)
            live_neighbors = sum(board[nr][nc] for nr, nc in neighbors)
            next_state = apply_rules(board[i][j], live_neighbors)
            row.append(next_state)
        next_board.append(row)
    
    # Copy next state back to original board
    for i in range(m):
        for j in range(n):
            board[i][j] = next_board[i][j]


def game_of_life_infinite_board(board):
    """
    Update board considering it as part of an infinite board.
    
    Args:
        board: 2D list representing the board (modified in-place)
    """
    if not board or not board[0]:
        return
    
    m, n = len(board), len(board[0])
    
    # Find all live cells and their neighbors
    live_cells = set()
    for i in range(m):
        for j in range(n):
            if board[i][j] == 1:
                live_cells.add((i, j))
    
    # Count neighbors for all relevant cells
    neighbor_counts = {}
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for i, j in live_cells:
        for dr, dc in directions:
            nr, nc = i + dr, j + dc
            if 0 <= nr < m and 0 <= nc < n:  # Only consider cells within board
                neighbor_counts[(nr, nc)] = neighbor_counts.get((nr, nc), 0) + 1
    
    # Apply rules
    for i in range(m):
        for j in range(n):
            live_neighbors = neighbor_counts.get((i, j), 0)
            
            if board[i][j] == 1:  # Live cell
                if live_neighbors < 2 or live_neighbors > 3:
                    board[i][j] = 0
            else:  # Dead cell
                if live_neighbors == 3:
                    board[i][j] = 1


# Test cases
if __name__ == "__main__":
    # Test case 1
    board1 = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
    expected1 = [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]
    board1_copy = [row[:] for row in board1]
    game_of_life(board1_copy)
    print(f"Test 1 - Input: {board1}")
    print(f"Expected: {expected1}")
    print(f"StateEncoding: {board1_copy}")
    
    board1_copy = [row[:] for row in board1]
    game_of_life_extra_space(board1_copy)
    print(f"ExtraSpace: {board1_copy}")
    
    board1_copy = [row[:] for row in board1]
    game_of_life_bit_manipulation(board1_copy)
    print(f"BitManipulation: {board1_copy}")
    
    board1_copy = [row[:] for row in board1]
    game_of_life_functional(board1_copy)
    print(f"Functional: {board1_copy}")
    
    board1_copy = [row[:] for row in board1]
    game_of_life_infinite_board(board1_copy)
    print(f"InfiniteBoard: {board1_copy}")
    print()
    
    # Test case 2 - Blinker pattern
    board2 = [[0,0,0],[1,1,1],[0,0,0]]
    expected2 = [[0,1,0],[0,1,0],[0,1,0]]
    board2_copy = [row[:] for row in board2]
    game_of_life(board2_copy)
    print(f"Test 2 - Input: {board2}")
    print(f"Expected: {expected2}")
    print(f"StateEncoding: {board2_copy}")
    
    board2_copy = [row[:] for row in board2]
    game_of_life_extra_space(board2_copy)
    print(f"ExtraSpace: {board2_copy}")
    print()
    
    # Test case 3 - Block pattern (stable)
    board3 = [[1,1],[1,1]]
    expected3 = [[1,1],[1,1]]
    board3_copy = [row[:] for row in board3]
    game_of_life(board3_copy)
    print(f"Test 3 - Input: {board3}")
    print(f"Expected: {expected3}")
    print(f"StateEncoding: {board3_copy}")
    print()
    
    # Test case 4 - All dead
    board4 = [[0,0,0],[0,0,0],[0,0,0]]
    expected4 = [[0,0,0],[0,0,0],[0,0,0]]
    board4_copy = [row[:] for row in board4]
    game_of_life(board4_copy)
    print(f"Test 4 - Input: {board4}")
    print(f"Expected: {expected4}")
    print(f"StateEncoding: {board4_copy}")
    print()
    
    # Test case 5 - Single live cell (dies)
    board5 = [[0,0,0],[0,1,0],[0,0,0]]
    expected5 = [[0,0,0],[0,0,0],[0,0,0]]
    board5_copy = [row[:] for row in board5]
    game_of_life(board5_copy)
    print(f"Test 5 - Input: {board5}")
    print(f"Expected: {expected5}")
    print(f"StateEncoding: {board5_copy}")
    print()
    
    # Test case 6 - Corner cases
    board6 = [[1,0],[0,1]]
    expected6 = [[0,0],[0,0]]
    board6_copy = [row[:] for row in board6]
    game_of_life(board6_copy)
    print(f"Test 6 - Input: {board6}")
    print(f"Expected: {expected6}")
    print(f"StateEncoding: {board6_copy}")
    print()
    
    # Test case 7 - Beehive pattern (stable)
    board7 = [[0,0,0,0],[0,1,1,0],[1,0,0,1],[0,1,1,0]]
    expected7 = [[0,0,0,0],[0,1,1,0],[1,0,0,1],[0,1,1,0]]
    board7_copy = [row[:] for row in board7]
    game_of_life(board7_copy)
    print(f"Test 7 - Input: {board7}")
    print(f"Expected: {expected7}")
    print(f"StateEncoding: {board7_copy}")
    print()
    
    # Test case 8 - Single row
    board8 = [[1,1,1]]
    expected8 = [[0,1,0]]
    board8_copy = [row[:] for row in board8]
    game_of_life(board8_copy)
    print(f"Test 8 - Input: {board8}")
    print(f"Expected: {expected8}")
    print(f"StateEncoding: {board8_copy}")
    print()
    
    # Test case 9 - Single column
    board9 = [[1],[1],[1]]
    expected9 = [[0],[1],[0]]
    board9_copy = [row[:] for row in board9]
    game_of_life(board9_copy)
    print(f"Test 9 - Input: {board9}")
    print(f"Expected: {expected9}")
    print(f"StateEncoding: {board9_copy}")
    print()
    
    # Test case 10 - Multiple generations test
    print("Testing multiple generations:")
    board_multi = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
    print(f"Generation 0: {board_multi}")
    
    for gen in range(1, 4):
        game_of_life(board_multi)
        print(f"Generation {gen}: {board_multi}")
    
    print()
    
    # Test case 11 - Oscillator test (should return to original after 2 generations)
    print("Testing oscillator pattern:")
    board_osc = [[0,0,0],[1,1,1],[0,0,0]]
    original = [row[:] for row in board_osc]
    print(f"Original: {original}")
    
    game_of_life(board_osc)
    print(f"After 1 generation: {board_osc}")
    
    game_of_life(board_osc)
    print(f"After 2 generations: {board_osc}")
    print(f"Back to original: {board_osc == original}") 