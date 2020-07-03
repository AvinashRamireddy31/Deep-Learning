
val = [60, 100, 50] 
wt  = [2, 3, 1] 
capacity = 3
length   = len(val)

def knapsack_tabulation(capacity, length):

    #Initialize with '0'
    K = [[0 for _ in range(capacity + 1)] for _ in range(length + 1)] 

    for l in range(length+1):
        for c in range(capacity+1):  
            if l == 0 or c == 0 : # stopping criteria
                K[l][c] = 0
            elif wt[l-1] <= c:
                K[l][c] = max( val[l-1] + K[l-1][c - wt[l-1]], #include
                               K[l-1][c])  # exclude
                        
            else:
                K[l][c] = K[l-1][c] # exclude

    for x in range(length+1):
        print("x:", x, end='\t')
        for y in range(capacity+1):
            print(K[x][y], end='\t')
        print(" ") 
                
    return K[length][capacity]
     

def knapsack_recursive(W, n):
    
    if n == 0 or wt == 0 :
        return 0
        
    if wt[n-1] <= W: 
        max_value = max( 
                        val[n-1] + knapsack_recursive( W - wt[n-1], n-1), # include weight: add respective value and reduce capacity 
                        knapsack_recursive(W,  n-1) # exclude weight
                    )
        return max_value
    
    return knapsack_recursive(W, n-1) #exclude weight

max_value = knapsack_recursive(capacity, length)
print("recursive:",max_value)

max_value = knapsack_tabulation(capacity, length)
print("Tabulation:", max_value)



