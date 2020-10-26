# Definition for a binary tree node.
import copy
from collections import deque
import heapq

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right


def prettyPrintTree(node, prefix="", isLeft=True):
    if not node:
        print("Empty Tree")
        return

    if node.right:
        prettyPrintTree(node.right, prefix + ("│   " if isLeft else "    "), False)

    print(prefix + ("└── " if isLeft else "┌── ") + str(node.val))

    if node.left:
        prettyPrintTree(node.left, prefix + ("    " if isLeft else "│   "), True)


class Solution:
    def countSubstrings(self, s: str) -> int:
        pass

    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        M = [[-1]*n for i in range(n)]
        for i in range(n): 
            M[i][i]= True
            if(i<(n-1)):
                M[i][i+1]= (s[i]==s[i+1])

        # memoisation
        def m(i,j):
            if(M[i][j]!=-1):
                return M[i][j]
            else:
                res = (m(i+1,j-1) and s[i]==s[j])
                M[i][j]= res
                return res

        maxx = 0
        _i, _j = 0,0
        for i in range(n):
            for j in range(i+1,n):
                if(m(i,j)):
                    if( (j-i)+1 > maxx):
                        maxx = (j-i)+1
                        _i =i
                        _j =j
              
                        
        return s[_i:(_j+1)]      


    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1 = len(text1) +1
        n2= len(text2) +1

        M = [[0]*n2 for i in range(n1)]

        for i in range(n1-1)[::-1]:
            for j in range(0, n2-1)[::-1]:
                if(text1[i]==text2[j]):
                    M[i][j]= 1+ M[i+1][j+1]
                else:
                    M[i][j] = max(M[i+1][j], M[i][j+1])

        chars = ""
        i, j = 0, 0
        while( i < (n1-1) and j < (n2-1)):
            if(text1[i]==text2[j]):
                chars+=text1[i]
                i+=1
                j+=1
            else:
                if( M[i+1][j] > M[i][j+1]):
                    i+=1
                else:
                    j+=1


        return M[0][0], chars


    def uniquePaths(self, m: int, n: int) -> int:
        M = [[1]* n for i in range(m)]

        for i in range(m):
            for j in range(n):
                if(i== 0 and j==0): 
                    M[i][j]=1
                elif(i>0 and j==0):
                    M[i][j] = M[i-1][j]
                elif(i==0 and j>0):
                    M[i][j]= M[i][j-1]
                else:
                    M[i][j]= M[i-1][j] + M[i][j-1]

        return M[m-1][n-1]

    def lengthOfLongestSubstring(self, s: str) -> int:
        if(len(s)==0): return 0
        else:
            init, last = 0, 1
            max_length = 1
            lastseen = {}
            lastseen[s[0]]=0

            while last < len(s):
                # visit next char
                if(s[last] in lastseen):
                    print("here")
                    value = lastseen[s[last]]+1
                    if(value>init):
                        init = value

                print("{} = {}:{}".format(s,init, last))
                
                if((last-init+1)> max_length):
                    max_length = last-init +1

                lastseen[s[last]]= last
                print(lastseen)
                last+=1

            return max_length

    def longestConsecutive(self, nums) -> int:
        if(len(nums)==0):
            return 0
        else:
            my_set = set(nums)
            current_max  = 1
            current_num = None
            for num in my_set:
                pass
            return current_max

            

    
    def invertTree(self, root: TreeNode) -> TreeNode:
        if(not root):
            return root
        else:   
            aux = root.left
            root.left = self.invertTree(root.right)
            root.right = self.invertTree(aux)
            return root

        



if __name__== "__main__":
    solution = Solution()
    #tree = TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)), TreeNode(7, TreeNode(6), TreeNode(9)))
    #print(solution.longestPalindrome("banana"))
    #prettyPrintTree(tree)
    #prettyPrintTree(solution.invertTree(tree))

    #print(solution.longestCommonSubsequence("", ""))
    #print(solution.uniquePaths(3,))
    #print(solution.longestConsecutive([100,4,200,1,3,2]))

    ibm_prices = [108.68, 109.65, 121.01, 122.78, 120.16]
    print(heapq.nsmallest(3, ibm_prices))
    print(heapq.nlargest(3, ibm_prices))


    