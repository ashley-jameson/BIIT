import numpy as np

icomplex = complex(0,1)
rotation_generators = 0.5 * np.array([
[[0.,1.],[1.,0.]],
[[0.,-icomplex],[icomplex,0.]],
[[1.,0.],[0.,-1.]]
])

s = 4
A = np.array(
    [
        [0,   0,   0,   0],
        [0.5, 0,   0,   0],
        [0,   0.5, 0,   0],
        [0,   0,   1.0, 0]
    ]
)
b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
c = np.array([0, 0.5, 0.5, 1.0])

m1, m2, m3 = 2, 2, -1

def matrix_multiply(a,b):
    n1,m1 = a.shape
    n2,m2 = b.shape
    ans = np.zeros((n1,m2),dtype="complex128")
    for row in range(n1):
        row_value = a[row]
        for column in range(m2):
            column_value = b.T[column]
            ans[row,column] = np.vdot(row_value,column_value)
    return ans

def commutator(a,b):
    return matrix_multiply(a,b) - matrix_multiply(b,a)

def condition_check(val, type="matrix"):
    if type == "matrix":
        a,b,c,d = np.real(val[0,0]), np.imag(val[0,0]), np.real(val[1,0]), np.imag(val[1,0])
    elif type == "vector":
        a,b,c,d = np.real(val[0]), np.imag(val[0]), np.real(val[1]), np.imag(val[1])
    return a*a + b*b + c*c + d*d

def algebra(sequence):
    sequence = sequence.upper()
    counts = [sequence.count(l) for l in ['A','T','G','C']]
    return np.sum([counts[i]*letter_array[i] for i in range(4)],axis=0)

def rkmk_step(Y,y,n,h=1e-15):
    k = np.zeros((s,2,2), dtype="complex128")
    I1 = Y(y,n)
    k[0] = Y(y,n)
    for i in range(1,s):
        u = h * np.sum([A[i,j] * k[j] for j in range(i)], axis=0)
        u_tilda = u + (((c[i] * h) / 6) * commutator(I1, u))
        k[i] = Y(matrix_multiply(y, expm(u_tilda)), n)

    I2 = ((m1 * (k[1] - I1)) + (m2 * (k[2] - I1)) + (m3 * (k[3] - I1))) / h
    v = h * np.sum([b[j] * k[j] for j in range(s)], axis=0)
    v_tilda = v + ((h / 4) * commutator(I1,v)) + ((h**2 / 24) * commutator(I2,v))
    y = matrix_multiply(y, expm(v_tilda))
    print(condition_check(y))
    if not np.isclose(condition_check(y),1.,rtol=1e-05, atol=1e-08):
        return 'NaN'
    else:
        return y

class C2:
    def __init__(self, y = np.array([[complex(0,0), complex(0,0)], [complex(1,0), complex(0,0)]])):
        self.y = y
    def y(self):
        return self._y
    def y(self, value):
        if not np.isclose(np.square(np.abs(value[0])) + np.square(np.abs(value[1])), 1.0):
            raise ValueError
        self._y = value
