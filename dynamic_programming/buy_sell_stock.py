class Solution:
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/submissions/
    def maxProfit(self, prices: [int]) -> int:
        if not prices:
            return 0 

        profit = 0
        buy_stock = prices[0]

        for i in range(1,len(prices)):
            current_price = prices[i] 

            if current_price < buy_stock:
                buy_stock = current_price 
            
            profit = max(profit, current_price - buy_stock) 
            
        return profit

solution = Solution()
# print(solution.maxProfit(prices= [7, 2, 5, 4, 1, 3, 6, 8]))
print(solution.maxProfit2(prices = [7,1,5,3,6,4]))