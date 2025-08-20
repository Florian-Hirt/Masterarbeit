from perturbation import perturbation

test_function = """from sys import stdin
input = stdin.readline

inp = lambda : list(map(int,input().split()))

def update(i , t):

    global ans

    if(i + 1 < n and a[i] == a[i + 1]):
        ans += t * (i + 1)
    else:
        ans += t * (n - i) * (i + 1)

    return ans

def answer():

    global ans

    ans = 0
    for i in range(n):

        update(i , 1)


    for q in range(m):
        i , x = inp()
        i -= 1

        if(i >= 0):update(i - 1 , -1)
        update(i , -1)

        a[i] = x
        if(i >= 0):update(i - 1 , 1)
        update(i , 1)

        print(ans)

        
for T in range(1):

    n , m = inp()
    a = inp()
    
    answer()
"""


perturbed_function = perturbation(test_function)

print(perturbed_function[0])

compiled_function = compile(perturbed_function[0], '<string>', 'exec')
exec(compiled_function)

