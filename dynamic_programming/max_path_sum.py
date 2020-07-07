def max_path_sum(grid):  
    if not grid:
        return 0 
        
    m, n = len(data), len(data[0])

    for i in range(m):
        for j in range(n):  
            if i == 0 and j == 0:
                continue  # Do nothing for first item.
            
            if i > 0 and j > 0:
                grid[i][j] += min( grid[i-1][j], grid[i][j-1])
            elif i > 0:
                grid[i][j] += grid[i-1][j]
            elif j > 0:
                grid[i][j] += grid[i][j-1] 
            
             
    
    return grid[-1][-1]  

    
data = [ 
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
] 
result = max_path_sum(data)
print("Minimum path sum is",result)
    