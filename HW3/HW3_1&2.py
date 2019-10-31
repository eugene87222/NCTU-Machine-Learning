import numpy as np
import sys, math

def UnivariateGaussianDataGenerator(m, std):
    return m + std * (sum(np.random.uniform(0, 1, 12)) - 6)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python ./HW3_1&2.py <mean> <variance>')
    else:
        m_src = float(sys.argv[1])
        var_src = float(sys.argv[2])
        std_src = math.sqrt(var_src)
        print(f'Data point source function: N({m_src}, {var_src})\n')
        n = 0
        m = 0
        m_pre = 0
        M2 = 0
        M2_pre = 0
        while True:
            new_data = UnivariateGaussianDataGenerator(m_src, std_src)
            print(f'Add data point: {new_data}')
            n += 1
            m = ((n - 1) * m_pre + new_data) / n
            M2 = M2_pre + (new_data - m_pre) * (new_data - m)
            print(f'Mean = {m} Variance = {M2 / n}')
            if abs(m - m_pre) < 5e-5 and abs((M2 / n) - (M2_pre / (n - 1))) < 5e-5:
                break
            m_pre = m
            M2_pre = M2