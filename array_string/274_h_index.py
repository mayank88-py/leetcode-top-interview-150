"""
274. H-Index

Problem:
Given an array of integers citations where citations[i] is the number of citations a researcher
received for their ith paper, return compute the researcher's h-index.

According to the definition of h-index on Wikipedia: A scientist has an index h if h of his/her N papers
have at least h citations each, and the other N − h papers have no more than h citations each.

If there are several possible values of h, the maximum one is taken as the h-index.

Example 1:
Input: citations = [3,0,6,1,5]
Output: 3
Explanation: [3,0,6,1,5] means the researcher has 5 papers in total and each of them had received 
3, 0, 6, 1, 5 citations respectively.
Since the researcher has 3 papers with at least 3 citations each and the remaining two with no more 
than 3 citations each, their h-index is 3.

Example 2:
Input: citations = [1,3,1]
Output: 1

Time Complexity: O(n log n) for sorting approach, O(n) for counting sort
Space Complexity: O(1) for sorting approach, O(n) for counting sort
"""


def h_index_sorting(citations):
    """
    Calculate H-Index using sorting approach.
    
    Args:
        citations: List of citation counts
    
    Returns:
        H-Index value
    """
    if not citations:
        return 0
    
    citations.sort(reverse=True)
    h_index = 0
    
    for i, citation in enumerate(citations):
        # i+1 is the number of papers with at least 'citation' citations
        # citation is the number of citations for current paper
        h = min(i + 1, citation)
        h_index = max(h_index, h)
    
    return h_index


def h_index_counting_sort(citations):
    """
    Calculate H-Index using counting sort approach.
    
    Args:
        citations: List of citation counts
    
    Returns:
        H-Index value
    """
    if not citations:
        return 0
    
    n = len(citations)
    # Count papers with each citation count
    # Index i stores count of papers with i citations
    # Use n+1 size to handle papers with > n citations
    counts = [0] * (n + 1)
    
    for citation in citations:
        # Papers with > n citations are counted at index n
        counts[min(citation, n)] += 1
    
    # Calculate h-index from right to left
    papers_with_at_least_h_citations = 0
    
    for h in range(n, -1, -1):
        papers_with_at_least_h_citations += counts[h]
        
        # If we have at least h papers with h or more citations
        if papers_with_at_least_h_citations >= h:
            return h
    
    return 0


def h_index_binary_search(citations):
    """
    Calculate H-Index using binary search approach.
    
    Args:
        citations: List of citation counts
    
    Returns:
        H-Index value
    """
    if not citations:
        return 0
    
    citations.sort()
    n = len(citations)
    left, right = 0, n
    
    while left < right:
        mid = (left + right + 1) // 2
        
        # Count papers with at least mid citations
        papers_with_at_least_mid = 0
        for citation in citations:
            if citation >= mid:
                papers_with_at_least_mid += 1
        
        if papers_with_at_least_mid >= mid:
            left = mid
        else:
            right = mid - 1
    
    return left


def h_index_simple(citations):
    """
    Calculate H-Index using simple approach.
    
    Args:
        citations: List of citation counts
    
    Returns:
        H-Index value
    """
    if not citations:
        return 0
    
    n = len(citations)
    h_index = 0
    
    # Try each possible h value from 0 to n
    for h in range(n + 1):
        papers_with_at_least_h = sum(1 for c in citations if c >= h)
        
        if papers_with_at_least_h >= h:
            h_index = h
    
    return h_index


# Test cases
if __name__ == "__main__":
    # Test case 1
    citations1 = [3, 0, 6, 1, 5]
    result1a = h_index_sorting(citations1.copy())
    result1b = h_index_counting_sort(citations1)
    result1c = h_index_binary_search(citations1.copy())
    result1d = h_index_simple(citations1)
    print(f"Test 1 - Expected: 3, Sorting: {result1a}, Counting: {result1b}, Binary: {result1c}, Simple: {result1d}")
    
    # Test case 2
    citations2 = [1, 3, 1]
    result2a = h_index_sorting(citations2.copy())
    result2b = h_index_counting_sort(citations2)
    result2c = h_index_binary_search(citations2.copy())
    result2d = h_index_simple(citations2)
    print(f"Test 2 - Expected: 1, Sorting: {result2a}, Counting: {result2b}, Binary: {result2c}, Simple: {result2d}")
    
    # Test case 3 - Empty array
    citations3 = []
    result3a = h_index_sorting(citations3.copy())
    result3b = h_index_counting_sort(citations3)
    result3c = h_index_binary_search(citations3.copy())
    result3d = h_index_simple(citations3)
    print(f"Test 3 - Expected: 0, Sorting: {result3a}, Counting: {result3b}, Binary: {result3c}, Simple: {result3d}")
    
    # Test case 4 - All zeros
    citations4 = [0, 0, 0, 0]
    result4a = h_index_sorting(citations4.copy())
    result4b = h_index_counting_sort(citations4)
    result4c = h_index_binary_search(citations4.copy())
    result4d = h_index_simple(citations4)
    print(f"Test 4 - Expected: 0, Sorting: {result4a}, Counting: {result4b}, Binary: {result4c}, Simple: {result4d}")
    
    # Test case 5 - Single paper
    citations5 = [100]
    result5a = h_index_sorting(citations5.copy())
    result5b = h_index_counting_sort(citations5)
    result5c = h_index_binary_search(citations5.copy())
    result5d = h_index_simple(citations5)
    print(f"Test 5 - Expected: 1, Sorting: {result5a}, Counting: {result5b}, Binary: {result5c}, Simple: {result5d}")
    
    # Test case 6 - High citations
    citations6 = [10, 8, 5, 4, 3]
    result6a = h_index_sorting(citations6.copy())
    result6b = h_index_counting_sort(citations6)
    result6c = h_index_binary_search(citations6.copy())
    result6d = h_index_simple(citations6)
    print(f"Test 6 - Expected: 4, Sorting: {result6a}, Counting: {result6b}, Binary: {result6c}, Simple: {result6d}")
    
    # Test case 7 - All same values
    citations7 = [4, 4, 4, 4]
    result7a = h_index_sorting(citations7.copy())
    result7b = h_index_counting_sort(citations7)
    result7c = h_index_binary_search(citations7.copy())
    result7d = h_index_simple(citations7)
    print(f"Test 7 - Expected: 4, Sorting: {result7a}, Counting: {result7b}, Binary: {result7c}, Simple: {result7d}") 