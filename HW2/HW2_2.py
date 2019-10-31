import sys

def Facotrial(n):
    return n * Facotrial(n - 1) if n > 1 else 1

def C(N, m):
    if N - m < m:
        m = N - m
    res = 1
    for i in range(m):
        res *= N
        N -= 1
    res /= Facotrial(m)
    return res

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python HW2_2.py <testfile>')
    else:
        with open(sys.argv[1], 'r', encoding='utf-8') as file:
            testcase = file.readlines()
        a = int(input('input a: '))
        b = int(input('input b: '))
        for i in range(len(testcase)):
            line = testcase[i].strip()
            N = len(line)
            m = 0
            for char in line:
                m += 1 if char == '1' else 0
            P = m / N
            likelihood = C(N, m) * (P ** m) * ((1 - P) ** (N - m))
            print(f'case {i + 1}: {line}')
            print(f'Likelihood: {likelihood}')
            print(f'Beta prior:\ta = {a}\tb = {b}')
            a += m
            b += (N - m)
            print(f'Beta posterior:\ta = {a}\tb = {b}\n')