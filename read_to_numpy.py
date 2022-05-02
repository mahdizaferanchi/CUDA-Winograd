import numpy as np

win_out = np.fromfile('tensors/winograd_out.bin', dtype=np.single).reshape((16, 16, 128)).transpose((2, 0, 1))
win_out_pooled = np.fromfile('tensors/winograd_out_pooled.bin', dtype=np.single).reshape((9, 9, 128)).transpose((2, 0, 1))
cudnnout = np.fromfile('tensors/cudnnout.bin', dtype=np.single).reshape((14, 14, 128)).transpose((2, 0, 1))
pooled = np.fromfile('tensors/pooled.bin', dtype=np.single).reshape((7, 7, 128)).transpose((2, 0, 1))

print('win out')
print(win_out[10][:8])
print(win_out.shape)
print('')

# print('win out pooled')
# print(win_out_pooled[0][:8])
# # print(win_out_pooled[66][1:-1, 1:-1])
# print(win_out_pooled.shape)
# print('')

# print('cudnn out')
# print(cudnnout[0][:4])
# print(cudnnout.shape)
# print('')

print('cudnn out pooled')
print(pooled[10][:8])
# print(pooled[66])
print(pooled.shape)
print('')


At = np.array([[ 1, 1, 1, 1, 1, 0],
               [ 0, 1,-1, 2,-2, 0],
               [ 0, 1, 1, 4, 4, 0],
               [ 0, 1,-1, 8,-8, 1]])
coefs = []

m = 3
n = 1
for i in range(6):
    for b in range(6):
        coefs.append(At[m][b]*At[n][i])
# print(coefs)
