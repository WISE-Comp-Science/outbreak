import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

h_nn = 1.0
a_nn = 5.0

ni = 100
nj = 100

h = np.ones((ni, nj)) * h_nn
a = np.ones((ni, nj)) * a_nn
x = np.zeros((ni, nj))
dxdt = np.zeros((ni, nj))

def get_nn(x, i, j):
    nn = []
    deltas_i, deltas_j = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    for di, dj in zip(deltas_i.flatten(), deltas_j.flatten()):
        ii = i + di
        jj = j + dj
        if ii < ni and ii >= 0 and jj < nj and jj >= 0 and not (di==0 and dj==0):
            nn.append(x[ii][jj])
    return np.array(nn)
    
def rhs(x, t):
    dxdt = - h * t
    for i in range(ni):
        for j in range(nj):
            xnn = get_nn(x, i, j)
            dxdt[i][j] += np.sum(a[i][j] * xnn)
    return dxdt

# Initialize disease!
x[0][0] = 1.0

# Run!
dt = 0.1
nt = 1000

frames = []
figure = plt.figure()

for i in range(nt):
    t  = dt * i
    x += dt * rhs(x, t)
    x  = np.maximum(np.minimum(x, np.ones((ni, nj))), np.zeros((ni, nj)))
    frames.append([plt.imshow(x, cmap='Reds')])

fig_animate = animation.ArtistAnimation(figure, frames, blit=True)
fig_animate.save("outbreak.mp4")
