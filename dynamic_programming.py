
val = [60, 100, 180] 
wt  = [20, 20,  30] 
capacity   = 50
length   = len(val)

def knapsack_memoization(capacity, length):

    #Initialize with '0'
    K = [[0 for x in range(capacity + 1)] for _ in range(length + 1)] 

    for l in range(length+1):
        for c in range(capacity+1):  
            if l == 0 or  c == 0 :
                K[l][c] = 0
            elif wt[l-1] <= c:
                K[l][c] = max( val[l-1] + K[l-1][c - wt[l-1]], 
                               K[l-1][c])  
                        
            else:
                K[l][c] = K[l-1][c]
                
    return K[length][capacity]
     

def knapsack_recursive(W, n):
    
    if n == 0 or wt == 0 :
        return 0
        
    if wt[n-1] <= W: 
        max_value = max( 
                        val[n-1] + knapsack_recursive( W - wt[n-1], n-1), # include weight
                        knapsack_recursive(W,  n-1) # exclude weight
                    )
        return max_value
    
    return knapsack_recursive(W, n-1) #exclude weight

max_value = knapsack_recursive(capacity, length)
print(max_value)

max_value = knapsack_memoization(capacity, length)
print(max_value)



