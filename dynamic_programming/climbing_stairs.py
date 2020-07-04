# 746. Min Cost Climbing Stairs

class Solution:
    
    def minCost_2(self, cost, n): 
        #Base case
        if n == 1:
            return cost[0]
        #Base case
        if n == 2:
            return min(cost[0], cost[1]) 
        
        first = cost[n-1] + self.minCost(cost, n-1)
        second = cost[n-2] + self.minCost(cost, n-2)
        
        return min(first, second)

    def minCost(self, cost, n): 
        #Base case 
        if n





        return -1
 
    
    def minCostClimbingStairs(self, cost: [int]) -> int:
        return self.minCost(cost, len(cost))

sol = Solution()
result = sol.minCostClimbingStairs(cost = [10, 15, 20])
print(result)