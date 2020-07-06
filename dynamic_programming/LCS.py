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
        mat[m][n] = max(
                        LCS(x, y, m-1, n), 
                        LCS(x, y, m, n-1)
                        )
        return mat[m][n]

    


# Driver program to test the above function 
X = "GAGATB"
Y = "XATGTYBC"

X = "abcd"
Y = "abdc"

m = len(X)
n = len(Y)

mat = [[-1 for x in range(n+1)] for x in range(m+1)] 
print ("Length of LCS is ", LCS(X, Y, m, n)) 


  

#Print longest common subsequence
def LCS_Print(x, y, m, n): 
    
    matched_string_size = mat[m][n]  # initialize with last value of matrix which has longest length.
    result = [""] * matched_string_size 
    i = m
    j = n 
    
    while i > 0 and j > 0:
        if x[i-1] == y[j-1]:
            # This is bottom-up approach, so first value is stored in last place of matrix. So we insert at last place to get the proper order
            matched_string_size -= 1
            result[matched_string_size] = x[i-1] 

            i -= 1
            j -= 1  

        elif mat[i-1][j] > mat[i][j-1]:
            i -= 1
        else:
            j -= 1

    return "".join(result) # Convert list to string



    

print ("Longest subsequence ", LCS_Print(X, Y, m, n))