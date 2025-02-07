# multiply scaler and m x m 

import numpy as np

arr = np.random.uniform(1,4,2)
arr1 =  np.random.random((2,2))
arr2 =  np.random.normal(5,5,5)
arr3 =  np.random.randint(1, 4, (2,2))
arr4 = np.random.rand(2,2)



arr1 = np.multiply(arr1, 2)
arr1 = np.power(arr1, 2)

np.add.reduce(arr3, axis = 0)
np.multiply.reduce(arr3)

a =  [1,2,3]
arr5 = np.multiply.outer(a, a)

np.add.accumulate(arr3)
np.multiply.accumulate(a)


v1 = np.array([3,-4]) 
v2 = np.array([2,3])

np.linalg.norm(v1,np.inf)

np.linalg.det(arr5)
np.linalg.inv(arr5) # det != 0 
np.linalg.trace(arr5)
np.linalg.eig(arr5)

a =  np.array([0,1,2,3,4,5,6,7,8,9])
a[3:9]
a[3:9:2]
a[-1]


mtx =  np.array([[1,2,3],[4,5,6],[7,8,9]])
# mtx[n] n row 
mtx[1]
# mtx[n:m] n row to m row
mtx[1:3]
# mtx[-1] last row
mtx[-1]

mtx[:2,1:3]
mtx[1:3,:2]

mtx[:,-1]
mtx2 =  mtx.copy()
mtx[0,0:3:2] = [10,15] 

x = np.array([11,22,33,44,55,66,77,88])
x1, x2, x3 = np.split(x, (2,4))

# vstack => vertical stack  == concatenate([a,b], axis = 1)
# hstack => herizotnal stack  == concatenate([a,b], axis = 0)

v3 = np.vstack((v1,v2))
v4 =  np.hstack((v1,v2))

np.concatenate((v1,v2), axis = 0)

np.var(mtx)
np.cov(mtx)

# add matrix sub
# same size 
z=  mtx * mtx2
z1 =  mtx / mtx2

# n m   m 
m = np.dot(mtx, mtx2)

m.mean()
m.meam(axis = 0 )
m.std()
m.var()

np.corrcoef(m)

ii = np.sort(mtx2, axis = 0 )
i = np.sort(mtx2, axis = 1)
