import scipy.stats as st
import numpy as np

def test_f(vals):
#    np.random.seed(1)
    np.random.shuffle(vals)
    n = len(vals)
    r = int(n/2)
    group_1 = vals[:r]
    group_2 = vals[r:] 
    f = (sum(group_1)/len(group_1))/(sum(group_2)/len(group_2))
    #print(f)
    #print(2*r)
    #print(2*(n-r))
    t_alph = .05/2
    stat = st.f(26,24)
    upper = stat.ppf(q=1-t_alph) 
    lower = stat.ppf(q=t_alph)
    if f < lower:
        return 1 
    if f > upper:
        return 1
    else:
        return 0
    
total_counts = [79.919,
3.081,
0.062,
1.961,
5.845,
3.027,
6.505,
0.021,
0.013,
0.123,
6.769,
59.899,
1.192,
34.760,
5.009,
18.387,
0.141,
43.565,
24.420,
0.433,
144.695,
2.663,
17.967,
0.091,
9.003,
0.941,
0.878,
3.371,
2.157,
7.579,
0.624,
5.380,
3.148,
7.078,
23.960,
0.590,
1.928,
0.300,
0.002,
0.543,
7.004,
31.764,
1.005,
1.147,
0.219,
3.217,
14.382,
1.008,
2.336,
4.562
]

a = st.weibull_min.fit(total_counts)
print(a)
#a = []
#for x in range(10000):
#    e = test_f(total_counts)
#    if e==1:
#        a.append(e)
#print(len(a)/10000)
