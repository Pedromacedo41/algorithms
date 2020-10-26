from collections import deque, namedtuple

class DSU(object):
    def __init__(self, size):
        self.par = range(size)
        self.rnk = [0] * size

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        elif self.rnk[xr] < self.rnk[yr]:
            self.par[xr] = yr
        elif self.rnk[xr] > self.rnk[yr]:
            self.par[yr] = xr
        else:
            self.par[yr] = xr
            self.rnk[xr] += 1
        return True


class Solution:
    def numIslands(self, grid) -> int:
        
        elems = set()
        pair = namedtuple("pair", "x, y")

        directions =[[1,0], [0,1],[-1,0], [0, -1]] 
        n_islands = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if(grid[i][j]=="1"): elems.add(pair(i,j))

        while len(elems)>0:
            my_queue = deque()
            my_queue.append(elems.pop())

            while my_queue:
                node = my_queue.popleft()
                for direction in directions: 
                    node_neigh =  pair(*[a+b for a,b in zip(node, direction)])
    
                    if (node_neigh[0] >= 0 
                        and node_neigh[0] < len(grid) 
                        and node_neigh[1] >=0 
                        and node_neigh[1] < len(grid[0]) 
                        and grid[node_neigh[0]][node_neigh[1]] == "1"
                        ## not visited
                        and node_neigh in elems):

                        my_queue.append(node_neigh)
                        elems.remove(node_neigh)

            n_islands+=1

        return n_islands



if __name__ == "__main__":
    solution = Solution()
    grid = [
        ["1","1","0","1","1"],
        ["1","1","1","0","0"],
        ["0","0","1","1","0"],
        ["0","0","0","1","1"]
        ]
    print(solution.numIslands(grid))

