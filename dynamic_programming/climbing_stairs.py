# 746. Min Cost Climbing Stairs

class Solution: 
    def minCost(self, cost, n): 
       
        dp = [None]*n
        
        dp[0] = cost[0]
        dp[1] = cost[1] 

        for i in range(2, n):
            dp[i] = cost[i] + min(dp[i-1], dp[i-2])

        return min(dp[n-1], dp[n-2]) 
    
    def minCostClimbingStairs(self, cost: [int]) -> int:
        return self.minCost(cost, len(cost))

sol = Solution()
result = sol.minCostClimbingStairs(cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1])
print(result)