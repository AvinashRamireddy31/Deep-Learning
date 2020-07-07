def countKdivPairs(A, n, K): 
      
    # Create a frequency array to count 
    # occurrences of all remainders when 
    # divided by K 
    freq = [0 for i in range(K)] 
  
    # To store count of pairs. 
    ans = 0
  
    # Traverse the array, compute the remainder 
    # and add k-remainder value hash count to ans 
    for i in range(n): 
        rem = A[i] % K 
        if (rem != 0): 
            ans += freq[K - rem] 
        else: 
            ans += freq[0] 
  
        # Increment count of remainder in hash map 
        freq[rem] += 1
  
    return ans 
    
# Driver code 
A = [2, 3, 1, 7, 5, 3] 
n = len(A) 
K = 4
print(countKdivPairs(A, n, K)) 