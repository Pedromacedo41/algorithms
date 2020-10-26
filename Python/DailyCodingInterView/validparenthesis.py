from collections import deque

def validparentsis(string):
    stack = deque()
    openParenthesis = ["(", "[", "{"]
    closingParenthesis =[")", "]", "}"]

    for char in string:
        if(char in closingParenthesis):
            if not stack: 
                return False
            elif (closingParenthesis.index(char) != openParenthesis.index(stack.pop())):
                return False

        else:
            stack.append(char)
    
    return len(stack)==0

if __name__== "__main__":
    print(validparentsis("{(())}[]()"))
    #stack = deque([1,2,3])

