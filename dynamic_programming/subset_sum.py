
wt  = [3, 34, 4, 12, 5, 2] 
target = 6
length = len(wt)

def is_subset_sum_tabulation(length, target):
    K = [[False for _ in range(target+ 1)] for _ in range(length+1)]

    for i in range(length+1):
        K[i][0] = True #First weight column with '0' in all rows

    for j in range(1, target+1): #Excluding first 0,0 indexes
        K[0][j] = False 
    
    for l in range(length + 1):
        for c in range(target + 1):
            if wt[l-1] <= c:
                K[l][c] = K[l-1][ c - wt[l-1]] or K[l-1][c] 
            else:
                K[l][c] = K[l-1][c]

    # Display logic
    print("Tabulation matrix")
    for x in range(length+1):
        print("x:", x, end='\t')
        for y in range(target+1):
            print(K[x][y], end='\t')
        print(" ") 
    print("  ")
                
    return K[length][target]


def is_subset_sum_recursive(length, target):
    # stopping criteria
    if target == 0:
        return True
    
    # stopping criteria
    if target != 0 and length == 0:
        return False
    
    #Recursive check
    if wt[length-1] <= target: 
         #include or #exclude
        return is_subset_sum_recursive(length-1, target - wt[length-1]) or is_subset_sum_recursive(length-1, target)
    else: 
        #exclude current item
        return is_subset_sum_recursive(length-1, target)
    

result = is_subset_sum_tabulation(length, target)
print("Tabulation:",result)

result = is_subset_sum_recursive(length, target)
print("Recursive:",result)



    