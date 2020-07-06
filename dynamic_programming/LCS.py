# https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/
def LCS(x, y, m, n): 
    if m == 0 or n == 0:
        return 0

    if mat[m][n] != -1:
        return mat[m][n]

    if x[m-1] == y[n-1]:
        mat[m][n] = 1 + LCS(x, y, m-1, n-1)
        return mat[m][n]
    else:
        mat[m][n] = max( LCS(x, y, m-1, n), LCS(x, y, m, n-1))
        return mat[m][n]

    


# Driver program to test the above function 
X = "AGGTAB"
Y = "GXTXAYB"

m = len(X)
n = len(Y)

mat = [[-1 for x in range(n+1)] for x in range(m+1)] 
print ("Length of LCS is ", LCS(X, Y, m, n)) 


  

#Print longest common subsequence
# https://www.geeksforgeeks.org/printing-longest-common-subsequence-set-2-printing/
def LCS_Print(x, y, m, n): 

    s = set() 

    if m == 0 or n == 0:
        s.add("")
        return s
    
    # If the last characters of X and Y are same 
    if x[m-1] == y[n-1]:
        tmp = LCS_Print(x, y, m-1, n-1) 

        # append current character to all possible substring
        for string in tmp: 
            s.add(string + x[m - 1]) 

    # If the last characters of X and Y are not same
    else:
        # return max( LCS(x, y, m-1, n), LCS(x, y, m, n-1)) 
        if mat[m - 1][n] >= mat[m][n - 1]: 
            s = LCS_Print(x, y, m - 1, n)

        if mat[m][n - 1] >= mat[m - 1][n]: 
            tmp = LCS_Print(x, y, m, n - 1) 
        
        for i in tmp: 
            s.add(i) 
    return s


    

print ("LCS", LCS_Print(X, Y, m, n))