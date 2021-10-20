import numpy as np

a = np.random.random((2, 2))
b = np.random.random((2, 2))

with open('test.npy', 'wb') as f:
    np.save(f, a)
    np.save(f, b)

np.savez('test.npz', *[a, b])

with open('test.npy', 'rb') as f:
    a1 = np.load(f)
    b1 = np.load(f)

ab = np.load("test.npz")

print(a1, b1)
print()
print(ab)