# Sliding Window 
General Template 

```python
    def Solution(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
		""" 
        l = 0 # start the left at zero 
        rtn = 0
		
		# the right should be the one iterating 
        for r in range(len(s)): 
			# This is the counter condition. Different question may have different condition 
			# This can have multiple conditions 
            while l < r: 
				# increase left pointer to make it invalid/valid again
                l += 1 
				
            # update the return value or do any checks to see if the substring/subarray is valid 
            if checkValidity: 
                rtn += 1 
        
        return rtn 
```

#### Sliding Window #1: Substring needs to decrease
https://leetcode.com/problems/find-all-anagrams-in-a-string/
```python
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
		""" 
        l = 0 
        rtn = [] 
        original_ht = Counter(list(p))
        curr_ht = defaultdict(int)
        for r in range(len(s)): 
            curr_ht[s[r]] += 1
            while l < r and r - l + 1 > len(p): 
                curr_ht[s[l]] -= 1
                if curr_ht[s[l]] == 0:
                    # need to make sure the 0 values are being deleted 
                    del curr_ht[s[l]]
                l += 1 
            # hash-table comparision compares the values so this is valid 
            if curr_ht == original_ht: 
                rtn.append(l) 
        
        return rtn 
```
## Slow / Fast Pointer 

## BFS 
- Make sure that we are not adding the same element back into the queue or there will be a recursive stack overflow
- 
## DFS 

### DFS where we not including the same number in the output 
https://leetcode.com/problems/combination-sum-iii/
```python
    def combinationSum3(self, k, n):
        rtn = [] 
        def dfs(k,n,curr,visited,index): 
            if n < 0: return 
            if len(curr) == k and n == 0: 
                rtn.append(curr[::])
            for i in range(index,10): 
                dfs(k,n-i,curr+[i],visited,i+1)

        dfs(k,n, [], set(), 1)
        
        return rtn 
``` 

#### DFS where we are keep tracking of already visited values
https://leetcode.com/problems/word-search/
```python
    def exist(self, board, word):
	
        visited = set() 
        
        def dfs(row,col,curr_index):
            if row >= 0 and row < len(board) and col >= 0 and col < len(board[0]) and (row,col) not in visited:
                if curr_index == len(word):
                    return True 
                
                if board[row][col] != word[curr_index]:
                    return False 
            
                visited.add((row,col))
                result = dfs(row+1, col, curr_index+1) or dfs(row-1, col, curr_index+1) or dfs(row, col+1, curr_index+1) or dfs(row, col-1, curr_index+1)
                visited.remove((row,col))
                return result
            else:
                return False 
                    
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    if dfs(i,j,0):
                        return True 
                    
        return False 
```

### Backtracking where we don't want the same combination with the same numbers
[2,3,4] -> making sure that we are not outputting [2,3,2] and [3,2,2]
```python
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        
        rtn = [] 
        #candidates.sort()
        def dfs(curr,curr_target,index):
            print(curr)
            if curr_target < 0: 
                return 
            elif curr_target == 0: 
                rtn.append(curr[::])
            else:
                for c in range(index, len(candidates)): 
                    dfs(curr+[candidates[c]], curr_target-candidates[c],c)
        dfs([], target,0)
        
        return rtn 
```

## Binary Search 
Use-case: *For all Problems* 
Resource: https://leetcode.com/discuss/general-discussion/786126/python-powerful-ultimate-binary-search-template-solved-many-problems
```python
def binary_search(array) -> int:
    def condition(value) -> bool:
        pass

    left, right = min(search_space), max(search_space) # could be [0, n], [1, n] etc. Depends on problem
    while left < right:
        mid = left + (right - left) // 2
        if condition(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
                    if nums[mid] >= target: --> USED WHEN finding the minimum index where its still target
					if nums[mid] > target: then return left - 1 ---> find the index one ABOVE the where it equals target
## Backtracking

## Linked List

#### Linked List #1: Cycle in Linked List - Tort/Hare Algo
Cycle detection // Fast+Slow pointer: 
https://leetcode.com/problems/linked-list-cycle-ii/
Notes:  The slow pointer moves one step at a time while the fast pointer moves two steps at a time
```python
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = head 
        fast = head 
        entry = head 
      
        while fast.next and fast: 
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                while entry != slow: 
                    slow = slow.next
                    entry = entry.next
                return entry 

        return None
```

## Trie 
https://leetcode.com/problems/implement-trie-prefix-tree/

```python 
from collections import defaultdict

class TrieNode(object):
    def __init__(self):
        self.nodes = { } 
        self.isEnd = False 

class Trie(object):
    def __init__(self):
        self.root = TrieNode() 
        
    def insert(self, word):
        """
        :type word: str
        :rtype: None
        """
        curr = self.root
        for w in word: 
            newNode = curr.nodes.get(w, TrieNode())
            curr.nodes[w] = newNode
            curr = newNode
        
        curr.isEnd = True 

    def search(self, word):
        """
        :type word: str
        :rtype: bool
        """
        currTrie = self.root
        for w in word: 
            if w not in currTrie.nodes:
                return False 
            currTrie = currTrie.nodes[w]
        
        return currTrie.isEnd
```

## Tree 

### Example 1: Finding Height of each node, bottom->up 
https://leetcode.com/problems/find-leaves-of-binary-tree/
```python
class Solution(object):
    def findLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        
        def dfs(curr):
            if not curr: 
                return 0 
            height = 1 + max(dfs(curr.left), dfs(curr.right)) 
            return height
        
        dfs(root)
	```
