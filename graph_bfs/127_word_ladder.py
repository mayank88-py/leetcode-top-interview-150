"""
127. Word Ladder

A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

- Every adjacent pair of words differs by exactly one letter.
- Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
- sk == endWord

Given two words, beginWord and endWord, and a dictionary wordList, return the length of the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

Example 1:
Input: beginWord = "hit", endWord = "cat", wordList = ["hot","dot","dog","lot","log","cot","cat"]
Output: 5
Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> "cot" -> "cat", which is 5 words long.

Example 2:
Input: beginWord = "hit", endWord = "cat", wordList = ["hot","dot","dog","lot","log","cot"]
Output: 0
Explanation: The endWord "cat" is not in wordList, therefore there is no valid transformation sequence.

Constraints:
- 1 <= beginWord.length <= 10
- endWord.length == beginWord.length
- 1 <= wordList.length <= 5000
- wordList[i].length == beginWord.length
- beginWord, endWord, and wordList[i] consist of lowercase English letters.
- beginWord != endWord
- All the words in wordList are unique.
"""

from typing import List, Set, Dict
from collections import deque, defaultdict


def ladder_length_bfs(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    Standard BFS approach (optimal solution).
    
    Time Complexity: O(M^2 * N) where M is word length and N is wordList size
    Space Complexity: O(M^2 * N) - for pattern mapping and queue
    
    Algorithm:
    1. Build pattern mapping for efficient neighbor finding
    2. Use BFS to find shortest path from beginWord to endWord
    3. Return path length or 0 if no path exists
    """
    if endWord not in wordList:
        return 0
    
    # Add beginWord to wordList if not present
    word_set = set(wordList)
    if beginWord not in word_set:
        word_set.add(beginWord)
    
    # Build pattern mapping: pattern -> [words with this pattern]
    patterns = defaultdict(list)
    for word in word_set:
        for i in range(len(word)):
            pattern = word[:i] + "*" + word[i+1:]
            patterns[pattern].append(word)
    
    # BFS
    queue = deque([(beginWord, 1)])
    visited = {beginWord}
    
    while queue:
        current_word, level = queue.popleft()
        
        # Check all patterns for current word
        for i in range(len(current_word)):
            pattern = current_word[:i] + "*" + current_word[i+1:]
            
            for neighbor in patterns[pattern]:
                if neighbor == endWord:
                    return level + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, level + 1))
    
    return 0


def ladder_length_bidirectional_bfs(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    Bidirectional BFS approach.
    
    Time Complexity: O(M^2 * N) where M is word length and N is wordList size
    Space Complexity: O(M^2 * N) - for pattern mapping and queues
    
    Algorithm:
    1. Search from both beginWord and endWord simultaneously
    2. Meet in the middle to find shortest path
    3. Potentially faster than standard BFS
    """
    if endWord not in wordList:
        return 0
    
    word_set = set(wordList)
    if beginWord not in word_set:
        word_set.add(beginWord)
    
    # Build pattern mapping
    patterns = defaultdict(list)
    for word in word_set:
        for i in range(len(word)):
            pattern = word[:i] + "*" + word[i+1:]
            patterns[pattern].append(word)
    
    # Initialize bidirectional search
    forward_queue = deque([(beginWord, 1)])
    backward_queue = deque([(endWord, 1)])
    forward_visited = {beginWord: 1}
    backward_visited = {endWord: 1}
    
    def visit_neighbors(queue, visited, other_visited):
        current_word, level = queue.popleft()
        
        for i in range(len(current_word)):
            pattern = current_word[:i] + "*" + current_word[i+1:]
            
            for neighbor in patterns[pattern]:
                if neighbor in other_visited:
                    return level + other_visited[neighbor]
                
                if neighbor not in visited:
                    visited[neighbor] = level + 1
                    queue.append((neighbor, level + 1))
        
        return None
    
    # Alternate between forward and backward search
    while forward_queue and backward_queue:
        # Search from smaller queue first for efficiency
        if len(forward_queue) <= len(backward_queue):
            result = visit_neighbors(forward_queue, forward_visited, backward_visited)
        else:
            result = visit_neighbors(backward_queue, backward_visited, forward_visited)
        
        if result:
            return result
    
    return 0


def ladder_length_optimized_bfs(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    Optimized BFS with character substitution.
    
    Time Complexity: O(M^2 * N) where M is word length and N is wordList size
    Space Complexity: O(M * N) - word set and queue
    
    Algorithm:
    1. For each word, try all possible single character changes
    2. Check if resulting word is in wordList
    3. Use BFS to find shortest path
    """
    if endWord not in wordList:
        return 0
    
    word_set = set(wordList)
    queue = deque([(beginWord, 1)])
    visited = {beginWord}
    
    while queue:
        current_word, level = queue.popleft()
        
        if current_word == endWord:
            return level
        
        # Try all possible single character changes
        for i in range(len(current_word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != current_word[i]:
                    new_word = current_word[:i] + c + current_word[i+1:]
                    
                    if new_word in word_set and new_word not in visited:
                        if new_word == endWord:
                            return level + 1
                        
                        visited.add(new_word)
                        queue.append((new_word, level + 1))
    
    return 0


def ladder_length_graph_building(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    Graph building approach.
    
    Time Complexity: O(M^2 * N^2) for graph building + O(M^2 * N) for BFS
    Space Complexity: O(M^2 * N^2) - adjacency list
    
    Algorithm:
    1. Pre-build adjacency graph of all words
    2. Use BFS on the graph to find shortest path
    3. More space-intensive but clear separation of concerns
    """
    if endWord not in wordList:
        return 0
    
    # Add beginWord to wordList
    all_words = wordList[:]
    if beginWord not in all_words:
        all_words.append(beginWord)
    
    # Build adjacency graph
    def is_adjacent(word1, word2):
        if len(word1) != len(word2):
            return False
        
        diff_count = 0
        for i in range(len(word1)):
            if word1[i] != word2[i]:
                diff_count += 1
                if diff_count > 1:
                    return False
        
        return diff_count == 1
    
    graph = defaultdict(list)
    for i, word1 in enumerate(all_words):
        for j, word2 in enumerate(all_words):
            if i != j and is_adjacent(word1, word2):
                graph[word1].append(word2)
    
    # BFS on the graph
    queue = deque([(beginWord, 1)])
    visited = {beginWord}
    
    while queue:
        current_word, level = queue.popleft()
        
        if current_word == endWord:
            return level
        
        for neighbor in graph[current_word]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, level + 1))
    
    return 0


def ladder_length_level_by_level_bfs(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    Level-by-level BFS approach.
    
    Time Complexity: O(M^2 * N) where M is word length and N is wordList size
    Space Complexity: O(M^2 * N) - pattern mapping and level sets
    
    Algorithm:
    1. Process words level by level
    2. Use sets to track current and next level words
    3. Clear level-by-level progression
    """
    if endWord not in wordList:
        return 0
    
    word_set = set(wordList)
    
    # Build pattern mapping
    patterns = defaultdict(list)
    for word in word_set:
        for i in range(len(word)):
            pattern = word[:i] + "*" + word[i+1:]
            patterns[pattern].append(word)
    
    current_level = {beginWord}
    visited = {beginWord}
    level = 1
    
    while current_level:
        next_level = set()
        
        for word in current_level:
            for i in range(len(word)):
                pattern = word[:i] + "*" + word[i+1:]
                
                for neighbor in patterns[pattern]:
                    if neighbor == endWord:
                        return level + 1
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
        
        current_level = next_level
        level += 1
    
    return 0


def ladder_length_dfs_with_memo(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    DFS with memoization approach (not optimal for this problem).
    
    Time Complexity: O(M^2 * N) where M is word length and N is wordList size
    Space Complexity: O(M^2 * N) - memoization cache + recursion stack
    
    Algorithm:
    1. Use DFS with memoization to find shortest path
    2. Cache results to avoid recomputation
    3. Not optimal for shortest path problems but shows alternative approach
    """
    if endWord not in wordList:
        return 0
    
    word_set = set(wordList)
    memo = {}
    
    def dfs(current_word, target, visited):
        if current_word == target:
            return 1
        
        if current_word in memo:
            return memo[current_word]
        
        min_length = float('inf')
        
        # Try all possible single character changes
        for i in range(len(current_word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != current_word[i]:
                    new_word = current_word[:i] + c + current_word[i+1:]
                    
                    if (new_word in word_set and new_word not in visited):
                        visited.add(new_word)
                        result = dfs(new_word, target, visited)
                        if result != 0:
                            min_length = min(min_length, result + 1)
                        visited.remove(new_word)
        
        result = min_length if min_length != float('inf') else 0
        memo[current_word] = result
        return result
    
    return dfs(beginWord, endWord, {beginWord})


def ladder_length_iterative_deepening(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    Iterative deepening approach.
    
    Time Complexity: O(M^2 * N * D) where D is the depth of solution
    Space Complexity: O(M * D) - recursion stack
    
    Algorithm:
    1. Try DFS with increasing depth limits
    2. Return when solution is found at minimum depth
    3. Memory efficient but potentially slower
    """
    if endWord not in wordList:
        return 0
    
    word_set = set(wordList)
    
    def dfs_with_limit(current_word, target, visited, depth, max_depth):
        if depth > max_depth:
            return False
        
        if current_word == target:
            return True
        
        # Try all possible single character changes
        for i in range(len(current_word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != current_word[i]:
                    new_word = current_word[:i] + c + current_word[i+1:]
                    
                    if new_word in word_set and new_word not in visited:
                        visited.add(new_word)
                        if dfs_with_limit(new_word, target, visited, depth + 1, max_depth):
                            return True
                        visited.remove(new_word)
        
        return False
    
    # Try increasing depths
    max_possible_depth = len(wordList) + 1
    for depth in range(1, max_possible_depth + 1):
        if dfs_with_limit(beginWord, endWord, {beginWord}, 1, depth):
            return depth
    
    return 0


def ladder_length_a_star(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    A* search approach with heuristic.
    
    Time Complexity: O(M^2 * N * log N) where M is word length and N is wordList size
    Space Complexity: O(M^2 * N) - priority queue and data structures
    
    Algorithm:
    1. Use A* with heuristic (number of different characters)
    2. Priority queue for optimal path exploration
    3. Can be faster than BFS in some cases
    """
    import heapq
    
    if endWord not in wordList:
        return 0
    
    word_set = set(wordList)
    
    def heuristic(word):
        """Number of different characters from endWord."""
        return sum(1 for i in range(len(word)) if word[i] != endWord[i])
    
    # Priority queue: (f_score, g_score, word)
    heap = [(heuristic(beginWord), 0, beginWord)]
    visited = set()
    
    while heap:
        f_score, g_score, current_word = heapq.heappop(heap)
        
        if current_word in visited:
            continue
        
        visited.add(current_word)
        
        if current_word == endWord:
            return g_score + 1
        
        # Try all possible single character changes
        for i in range(len(current_word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != current_word[i]:
                    new_word = current_word[:i] + c + current_word[i+1:]
                    
                    if new_word in word_set and new_word not in visited:
                        new_g_score = g_score + 1
                        new_f_score = new_g_score + heuristic(new_word)
                        heapq.heappush(heap, (new_f_score, new_g_score, new_word))
    
    return 0


# Test cases
def test_ladder_length():
    """Test all word ladder approaches."""
    
    # Test cases
    test_cases = [
        ("hit", "cog", ["hot","dot","dog","lot","log","cot","cog"], 5, "Standard example"),
        ("hit", "cog", ["hot","dot","dog","lot","log"], 0, "No path exists"),
        ("a", "c", ["a","b","c"], 2, "Simple transformation"),
        ("hot", "dog", ["hot","dog"], 0, "Direct transformation impossible"),
        ("hot", "dog", ["hot","hog","dog"], 3, "Simple path"),
        ("red", "tax", ["ted","tex","red","tax","tad","den","rex","pee"], 4, "Complex path"),
        ("hit", "hit", ["hot","hit"], 1, "Same word"),
        ("qa", "sq", ["si","go","se","cm","so","ph","mt","db","mb","sb","kr","ln","tm","le","av","sm","ar","ci","ca","br","ti","ba","to","ra","fa","yo","ow","sn","ya","cr","po","fe","ho","ma","re","or","rn","au","ur","rh","sr","tc","lt","lo","as","fr","nb","yb","if","pb","ge","th","pm","rb","sh","co","ga","li","ha","hz","no","bi","di","hi","qa","pi","os","uh","wm","an","me","mo","na","la","st","er","sc","ne","mn","mi","am","ex","pt","io","be","fm","ta","tb","ni","mr","pa","he","lr","sq","ye"], 5, "Long word list"),
        ("red", "tax", [], 0, "Empty word list"),
        ("a", "a", ["b"], 1, "Same begin and end word"),
    ]
    
    # Test all approaches
    approaches = [
        ("Standard BFS", ladder_length_bfs),
        ("Bidirectional BFS", ladder_length_bidirectional_bfs),
        ("Optimized BFS", ladder_length_optimized_bfs),
        ("Graph Building", ladder_length_graph_building),
        ("Level-by-level BFS", ladder_length_level_by_level_bfs),
        ("DFS with Memoization", ladder_length_dfs_with_memo),
        ("Iterative Deepening", ladder_length_iterative_deepening),
        ("A* Search", ladder_length_a_star),
    ]
    
    print("Testing word ladder approaches:")
    print("=" * 50)
    
    for i, (beginWord, endWord, wordList, expected, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        print(f"Begin: '{beginWord}', End: '{endWord}', WordList: {wordList[:5]}{'...' if len(wordList) > 5 else ''}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            try:
                result = func(beginWord, endWord, wordList[:])  # Pass copy to avoid modification
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    # Create larger test case for performance testing
    def create_large_test_case():
        """Create a larger test case."""
        import random
        import string
        
        word_length = 4
        num_words = 500
        words = set()
        
        # Generate random words
        while len(words) < num_words:
            word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            words.add(word)
        
        word_list = list(words)
        begin_word = word_list[0]
        end_word = word_list[-1]
        
        return begin_word, end_word, word_list[1:-1]
    
    begin_word, end_word, word_list = create_large_test_case()
    
    import time
    
    print(f"Testing with large dataset (500 words, length 4):")
    for name, func in approaches:
        try:
            start_time = time.time()
            result = func(begin_word, end_word, word_list[:])
            end_time = time.time()
            print(f"{name}: {result} (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_ladder_length() 