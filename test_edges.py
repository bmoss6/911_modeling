import math
import scipy.stats as st

edges = []
for j in range(1,20):
    edges.append(-1* 3.999 * math.log(1-j/20))


print(edges)
e = st.expon(0,3.999)

x = [y/20 for y in range(1,20)]
a = e.ppf(x)
print(a)
