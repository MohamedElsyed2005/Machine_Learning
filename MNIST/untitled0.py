import numpy as np 

np.random.seed(42)

theta = 90 
cosine =  np.cos(np.deg2rad(theta))

x =  0.56666 
x = np.round(x,2)

a = np.array([range(i,i+3) for i in [2,4,6]])

# =============================================================================
# np.random.random((shape))  np.random.random(size) 
# np.random.uniform(start, end, no of element)
# np.random.normal(n,m,d) random normal distribution matrix from n-1 to m with d element
# np.random.randint(n) random int from 0 to n-1 
# np.random.randint(n, size = m)
# np.random.randint(n,z, size = m)
# np.random.randint(n,z, (m,d))
# np.random.randint(size = (n,m,d))
# np.random.rand
# np.random.choice
# np.random.shuffle
# =============================================================================


arr = np.random.uniform(1,4,2)
arr1 =  np.random.random((2,2))
arr2 =  np.random.normal(5,5,5)
arr3 =  np.random.randint(1, 4, (2,2))
arr4 = np.random.rand(2,2)

y = [1,2,3,4,5]
np.random.choice(y)
np.random.shuffle(y)


# =============================================================================
# np.zeros((n,m))
# np.ones((n,m))
# np.eye(n)
# =============================================================================

w = np.arange(0,100,25)

u = np.linspace(0, 100,5)

diag = np.diag([1,2,3,4])
no = np.count_nonzero(a > 4)

np.isclose(arr1, arr4, rtol = 0.2)
