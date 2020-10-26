import numpy as np
INF = np.exp(31) - 1
from collections import namedtuple, deque
import numpy as np


class Solution:
    def wallsAndGates(self, rooms) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        roots = []
        for i in range(len(rooms)):
            for j in range(len(rooms[0])):
                if(rooms[i][j]==0): roots.append([i,j])

        directions =[[1,0], [0,1],[-1,0], [0, -1]] 

        myqueue = deque()
        step = 1
    
        for root in roots:
            for direction in directions: 
                node = [a+b for a,b in zip(root, direction)]

                # allowed directions
                if (node[0] >= 0 and node[0] < len(rooms) and node[1] >=0 and node[1] < len(rooms[0])):
                    if(rooms[node[0]][node[1]] == INF): 
                        rooms[node[0]][node[1]] = step
                        myqueue.append(node)

        if(myqueue): last_node_step = myqueue[-1]
        step+=1
       
        while myqueue:
            curret_node = myqueue.popleft()
            for direction in directions: 
                node =  [a+b for a,b in zip(curret_node, direction)]

                # allowed directions
                if (node[0] >= 0 and node[0] < len(rooms) and node[1] >=0 and node[1] < len(rooms[0])):
                    if(rooms[node[0]][node[1]] == INF): 
                        rooms[node[0]][node[1]] = step
                        myqueue.append(node)

            if(curret_node == last_node_step): 
                if(myqueue): last_node_step = myqueue[-1]
                step+=1

    def numIslands(self, grid: List[List[str]]) -> int:
        pass

        




if __name__== "__main__":
    testcase = [[INF, -1, 0, INF], [INF, INF, INF, -1], [INF, -1, INF, -1], [0, -1, INF, INF]]
    testcase2 = [[INF, 0, INF, INF, 0, INF, -1, INF]]
    solution = Solution()
    solution.wallsAndGates([])

        