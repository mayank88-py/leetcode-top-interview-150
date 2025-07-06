"""
433. Minimum Genetic Mutation

A gene string can be represented by an 8-character long string, with choices from 'A', 'C', 'G', and 'T'.

Suppose we need to investigate a mutation from a gene string start to a gene string end where one mutation is defined as one single character changed in the gene string.

For example, "AACCGGTT" --> "AACCGGTA" is one mutation.

There is also a gene bank bank that records all the valid gene mutations. A gene mutation is valid only if it is in the gene bank.

Given the two gene strings start and end and the gene bank bank, return the minimum number of mutations needed to mutate from start to end. If there is no such a mutation, return -1.

Note that the start point is assumed to be valid, so it might not be included in the gene bank.

Example 1:
Input: start = "AACCGGTT", end = "AACCGGTA", bank = ["AACCGGTA"]
Output: 1

Example 2:
Input: start = "AACCGGTT", end = "AAACGGTA", bank = ["AACCGGTA","AACCGCTA","AAACGGTA"]
Output: 2

Example 3:
Input: start = "AAAAACCC", end = "AACCCCCC", bank = ["AAAACCCC","AAACCCCC","AACCCCCC"]
Output: 3

Constraints:
- start.length == 8
- end.length == 8
- 0 <= bank.length <= 10
- bank[i].length == 8
- start, end, and bank[i] consist of only the characters ['A', 'C', 'G', 'T'].
"""

from typing import List, Set
from collections import deque, defaultdict


def min_mutation_bfs(start: str, end: str, bank: List[str]) -> int:
    """
    Standard BFS approach (optimal solution).
    
    Time Complexity: O(N * M * 4) where N is bank size and M is gene length
    Space Complexity: O(N) - queue and visited set
    
    Algorithm:
    1. Use BFS to find shortest mutation path from start to end
    2. For each gene, try all possible single character mutations
    3. Only consider mutations that are in the gene bank
    """
    if end not in bank:
        return -1
    
    bank_set = set(bank)
    queue = deque([(start, 0)])
    visited = {start}
    genes = ['A', 'C', 'G', 'T']
    
    while queue:
        current_gene, mutations = queue.popleft()
        
        if current_gene == end:
            return mutations
        
        # Try all possible single character mutations
        for i in range(len(current_gene)):
            for gene in genes:
                if gene != current_gene[i]:
                    new_gene = current_gene[:i] + gene + current_gene[i+1:]
                    
                    if new_gene in bank_set and new_gene not in visited:
                        if new_gene == end:
                            return mutations + 1
                        
                        visited.add(new_gene)
                        queue.append((new_gene, mutations + 1))
    
    return -1


def min_mutation_bidirectional_bfs(start: str, end: str, bank: List[str]) -> int:
    """
    Bidirectional BFS approach.
    
    Time Complexity: O(N * M * 4) where N is bank size and M is gene length
    Space Complexity: O(N) - two queues and visited sets
    
    Algorithm:
    1. Search from both start and end simultaneously
    2. Meet in the middle to find shortest path
    3. Potentially faster than standard BFS
    """
    if end not in bank:
        return -1
    
    bank_set = set(bank)
    genes = ['A', 'C', 'G', 'T']
    
    # Initialize bidirectional search
    forward_queue = deque([(start, 0)])
    backward_queue = deque([(end, 0)])
    forward_visited = {start: 0}
    backward_visited = {end: 0}
    
    def explore(queue, visited, other_visited):
        current_gene, mutations = queue.popleft()
        
        for i in range(len(current_gene)):
            for gene in genes:
                if gene != current_gene[i]:
                    new_gene = current_gene[:i] + gene + current_gene[i+1:]
                    
                    if new_gene in bank_set:
                        if new_gene in other_visited:
                            return mutations + 1 + other_visited[new_gene]
                        
                        if new_gene not in visited:
                            visited[new_gene] = mutations + 1
                            queue.append((new_gene, mutations + 1))
        
        return None
    
    # Alternate between forward and backward search
    while forward_queue and backward_queue:
        # Search from smaller queue first
        if len(forward_queue) <= len(backward_queue):
            result = explore(forward_queue, forward_visited, backward_visited)
        else:
            result = explore(backward_queue, backward_visited, forward_visited)
        
        if result is not None:
            return result
    
    return -1


def min_mutation_optimized_bfs(start: str, end: str, bank: List[str]) -> int:
    """
    Optimized BFS with early termination and preprocessing.
    
    Time Complexity: O(N * M * 4) where N is bank size and M is gene length
    Space Complexity: O(N) - bank set and queue
    
    Algorithm:
    1. Preprocess bank for faster lookups
    2. Early termination when target is found
    3. Optimized mutation generation
    """
    if end not in bank:
        return -1
    
    bank_set = set(bank)
    if start not in bank_set:
        bank_set.add(start)
    
    # BFS with optimizations
    queue = deque([(start, 0)])
    visited = {start}
    genes = ['A', 'C', 'G', 'T']
    
    while queue:
        current_gene, mutations = queue.popleft()
        
        # Try all single character mutations
        for i in range(8):  # Gene length is always 8
            original_char = current_gene[i]
            
            for new_char in genes:
                if new_char != original_char:
                    new_gene = current_gene[:i] + new_char + current_gene[i+1:]
                    
                    if new_gene == end:
                        return mutations + 1
                    
                    if new_gene in bank_set and new_gene not in visited:
                        visited.add(new_gene)
                        queue.append((new_gene, mutations + 1))
    
    return -1


def min_mutation_level_order_bfs(start: str, end: str, bank: List[str]) -> int:
    """
    Level-order BFS approach.
    
    Time Complexity: O(N * M * 4) where N is bank size and M is gene length
    Space Complexity: O(N) - level sets and visited set
    
    Algorithm:
    1. Process genes level by level
    2. Each level represents mutations with same count
    3. Clear separation between mutation counts
    """
    if end not in bank:
        return -1
    
    bank_set = set(bank)
    current_level = {start}
    visited = {start}
    mutations = 0
    genes = ['A', 'C', 'G', 'T']
    
    while current_level:
        next_level = set()
        
        for current_gene in current_level:
            if current_gene == end:
                return mutations
            
            # Try all single character mutations
            for i in range(len(current_gene)):
                for gene in genes:
                    if gene != current_gene[i]:
                        new_gene = current_gene[:i] + gene + current_gene[i+1:]
                        
                        if new_gene in bank_set and new_gene not in visited:
                            visited.add(new_gene)
                            next_level.add(new_gene)
        
        current_level = next_level
        mutations += 1
    
    return -1


def min_mutation_graph_building(start: str, end: str, bank: List[str]) -> int:
    """
    Graph building approach.
    
    Time Complexity: O(N^2 * M) for graph building + O(N) for BFS
    Space Complexity: O(N^2) - adjacency list
    
    Algorithm:
    1. Pre-build adjacency graph of all valid genes
    2. Use BFS on the graph to find shortest path
    3. More space-intensive but clear separation of concerns
    """
    if end not in bank:
        return -1
    
    all_genes = bank[:]
    if start not in all_genes:
        all_genes.append(start)
    
    # Build adjacency graph
    def is_one_mutation(gene1, gene2):
        if len(gene1) != len(gene2):
            return False
        
        diff_count = 0
        for i in range(len(gene1)):
            if gene1[i] != gene2[i]:
                diff_count += 1
                if diff_count > 1:
                    return False
        
        return diff_count == 1
    
    graph = defaultdict(list)
    for i, gene1 in enumerate(all_genes):
        for j, gene2 in enumerate(all_genes):
            if i != j and is_one_mutation(gene1, gene2):
                graph[gene1].append(gene2)
    
    # BFS on the graph
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        current_gene, mutations = queue.popleft()
        
        if current_gene == end:
            return mutations
        
        for neighbor in graph[current_gene]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, mutations + 1))
    
    return -1


def min_mutation_pattern_matching(start: str, end: str, bank: List[str]) -> int:
    """
    Pattern matching approach using wildcard patterns.
    
    Time Complexity: O(N * M * 4) where N is bank size and M is gene length
    Space Complexity: O(N * M) - pattern mapping
    
    Algorithm:
    1. Build pattern mapping similar to Word Ladder
    2. Use patterns with wildcards for efficient neighbor finding
    3. BFS using pattern-based neighbor discovery
    """
    if end not in bank:
        return -1
    
    # Build pattern mapping
    patterns = defaultdict(list)
    bank_set = set(bank)
    if start not in bank_set:
        bank_set.add(start)
    
    for gene in bank_set:
        for i in range(len(gene)):
            pattern = gene[:i] + "*" + gene[i+1:]
            patterns[pattern].append(gene)
    
    # BFS using patterns
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        current_gene, mutations = queue.popleft()
        
        if current_gene == end:
            return mutations
        
        # Check all patterns for current gene
        for i in range(len(current_gene)):
            pattern = current_gene[:i] + "*" + current_gene[i+1:]
            
            for neighbor in patterns[pattern]:
                if neighbor not in visited:
                    if neighbor == end:
                        return mutations + 1
                    
                    visited.add(neighbor)
                    queue.append((neighbor, mutations + 1))
    
    return -1


def min_mutation_dfs_with_memo(start: str, end: str, bank: List[str]) -> int:
    """
    DFS with memoization approach (not optimal for shortest path).
    
    Time Complexity: O(N * M * 4) where N is bank size and M is gene length
    Space Complexity: O(N) - memoization cache + recursion stack
    
    Algorithm:
    1. Use DFS with memoization to find shortest path
    2. Not optimal for shortest path problems but shows alternative
    3. Can lead to stack overflow for large inputs
    """
    if end not in bank:
        return -1
    
    bank_set = set(bank)
    memo = {}
    genes = ['A', 'C', 'G', 'T']
    
    def dfs(current_gene, target, visited):
        if current_gene == target:
            return 0
        
        if current_gene in memo:
            return memo[current_gene]
        
        min_mutations = float('inf')
        
        # Try all possible single character mutations
        for i in range(len(current_gene)):
            for gene in genes:
                if gene != current_gene[i]:
                    new_gene = current_gene[:i] + gene + current_gene[i+1:]
                    
                    if new_gene in bank_set and new_gene not in visited:
                        visited.add(new_gene)
                        result = dfs(new_gene, target, visited)
                        if result != float('inf'):
                            min_mutations = min(min_mutations, result + 1)
                        visited.remove(new_gene)
        
        memo[current_gene] = min_mutations
        return min_mutations
    
    result = dfs(start, end, {start})
    return result if result != float('inf') else -1


def min_mutation_dijkstra(start: str, end: str, bank: List[str]) -> int:
    """
    Dijkstra's algorithm approach (overkill for this problem).
    
    Time Complexity: O(N * M * 4 * log N) where N is bank size and M is gene length
    Space Complexity: O(N) - priority queue and distance array
    
    Algorithm:
    1. Use Dijkstra's algorithm to find shortest path
    2. All edges have weight 1, so BFS is more efficient
    3. Demonstrates alternative graph algorithm
    """
    import heapq
    
    if end not in bank:
        return -1
    
    bank_set = set(bank)
    if start not in bank_set:
        bank_set.add(start)
    
    # Dijkstra's algorithm
    dist = {gene: float('inf') for gene in bank_set}
    dist[start] = 0
    heap = [(0, start)]
    genes = ['A', 'C', 'G', 'T']
    
    while heap:
        current_dist, current_gene = heapq.heappop(heap)
        
        if current_gene == end:
            return current_dist
        
        if current_dist > dist[current_gene]:
            continue
        
        # Try all single character mutations
        for i in range(len(current_gene)):
            for gene in genes:
                if gene != current_gene[i]:
                    new_gene = current_gene[:i] + gene + current_gene[i+1:]
                    
                    if new_gene in bank_set:
                        new_dist = current_dist + 1
                        
                        if new_dist < dist[new_gene]:
                            dist[new_gene] = new_dist
                            heapq.heappush(heap, (new_dist, new_gene))
    
    return -1


# Test cases
def test_min_mutation():
    """Test all minimum genetic mutation approaches."""
    
    # Test cases
    test_cases = [
        ("AACCGGTT", "AACCGGTA", ["AACCGGTA"], 1, "Single mutation"),
        ("AACCGGTT", "AAACGGTA", ["AACCGGTA","AACCGCTA","AAACGGTA"], 2, "Two mutations"),
        ("AAAAACCC", "AACCCCCC", ["AAAACCCC","AAACCCCC","AACCCCCC"], 3, "Three mutations"),
        ("AACCGGTT", "AACCGGTA", [], -1, "End not in bank"),
        ("AACCGGTT", "AACCGGTT", ["AACCGGTT"], 0, "Start equals end"),
        ("AACCGGTT", "AACCGGTA", ["AACCGGTA","AACCGCTA","AAACGGTA"], 1, "Multiple paths, shortest is 1"),
        ("AAAAAAAA", "CCCCCCCC", ["AAAAAAAC","AAAAAACA","AAAAACAA","AAAACAAA","AAACAAAA","AACAAAAA","ACAAAAAA","CAAAAAAA","CAAAAAAC","CAAAAACA","CAAAACAA","CAAACAAA","CAACAAAA","CACAAAAA","CCAAAAAA","CCCCCCCC"], 8, "Long path"),
        ("AACCGGTT", "AACCGGTA", ["AACCGGTA","AACCGGTC","AACCGGTG"], 1, "Multiple valid mutations"),
        ("AAAAAAAT", "CCCCCCCC", ["AAAAAAAT","AAAAAACT","AAAAACCT","AAAAACTT","AAAACCTT","AAACCCTT","AACCCCTT","ACCCCTT","CCCCCTT","CCCCCCCT","CCCCCCCC"], -1, "Impossible mutation path"),
        ("AATTGGCC", "AATTGGCC", [], 0, "Start equals end, empty bank"),
    ]
    
    # Test all approaches
    approaches = [
        ("Standard BFS", min_mutation_bfs),
        ("Bidirectional BFS", min_mutation_bidirectional_bfs),
        ("Optimized BFS", min_mutation_optimized_bfs),
        ("Level-order BFS", min_mutation_level_order_bfs),
        ("Graph Building", min_mutation_graph_building),
        ("Pattern Matching", min_mutation_pattern_matching),
        ("DFS with Memo", min_mutation_dfs_with_memo),
        ("Dijkstra", min_mutation_dijkstra),
    ]
    
    print("Testing minimum genetic mutation approaches:")
    print("=" * 50)
    
    for i, (start, end, bank, expected, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        print(f"Start: '{start}', End: '{end}', Bank: {bank}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            try:
                result = func(start, end, bank[:])  # Pass copy to avoid modification
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
        import itertools
        
        # Generate some valid gene sequences
        genes = ['A', 'C', 'G', 'T']
        bank = []
        
        # Create a path from start to end
        start = "AAAAAAAA"
        end = "TTTTTTTT"
        
        # Add intermediate genes with single mutations
        current = list(start)
        for i in range(8):
            current[i] = 'T'
            bank.append(''.join(current))
        
        # Add some additional random genes
        for i in range(50):
            random_gene = ''.join([genes[j % 4] for j in range(i, i + 8)])
            if random_gene not in bank:
                bank.append(random_gene)
        
        return start, end, bank
    
    start, end, bank = create_large_test_case()
    
    import time
    
    print(f"Testing with larger dataset (start: {start}, end: {end}, bank size: {len(bank)}):")
    for name, func in approaches:
        try:
            start_time = time.time()
            result = func(start, end, bank[:])
            end_time = time.time()
            print(f"{name}: {result} (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_min_mutation() 