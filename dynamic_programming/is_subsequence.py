class Solution: 
    def isSubsequence(self, s: str, t: str) -> bool:
        t_flag = 0
        cnt = 0
        for i in range(len(s)):
            for j in range(t_flag, len(t)):
                if s[i] == t[j]:
                    cnt += 1
                    t_flag = j+1
                    break
                
        if cnt == len(s):
            return True
        else:
            return False





    def isSubsequence_1(self, s: str, t: str) -> bool: 
        if s == t or not s: 
            return True
            
        if not t: 
            return False

        s_pointer = 0
        t_pointer = 0

        while(t_pointer < len(t)):
            if t[t_pointer] == s[s_pointer]:
                s_pointer += 1  

                if s_pointer == len(s):
                    return True

            t_pointer +=  1
        
        return False


        

sol = Solution()
print(sol.isSubsequence(s = "axc", t = "ahbgdc"))