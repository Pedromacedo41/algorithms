
from collections import deque, namedtuple

class Solution:
    def openLock(self, deadends, target: str) -> int:
        if("0000" in deadends): return -1
        else:
            vis = set()
            vis.add("0000")
            vis = vis.union(set(deadends))
            myqueue = deque([("0000", 0)])
            step = 0
            last_visited = "0000"

            while myqueue:
                current_node, depth = myqueue.popleft()
                if(current_node == target): return depth
                for node in self.all_neighboors(current_node):
                    if(node not in vis): 
                        myqueue.append((node, depth+1))

                    vis.add(node)
            return -1
        
    def neighbor(self, num, pos):
       number = int(num[pos])
       up = (number+1)%10 
       down = (9 if number==0  else (number-1))
       return [(num[:pos]+ str(digit)+num[(pos+1):]) for  digit in [down, up]]


    def all_neighboors(self, num):
        neighs = []
        for i in range(4):
            neighs.extend(self.neighbor(num, i))
        return neighs

      
if __name__== "__main__":

    solution = Solution()
    deadends = ["0201","0101","0102","1212","2002"]
    deadends2 = ["8887","8889","8878","8898","8788","8988","7888","9888"]
    target = "0202"
    target2 = "8888"

    print(solution.openLock(deadends, "0202"))
    #print(solution.all_neighboors("0020"))