import numpy as np
import math
import matplotlib.pyplot as plt

point1 = [5,2]
point2 = [3,4]
point3 = [8,5]
set = [point1, point2, point3]
x_s = [point[0] for point in set]
y_s = [point[1] for point in set]
x_space = np.linspace(np.min(x_s)-2,np.max(x_s)+2)
ylim = (np.min(y_s)-2,np.max(y_s)+2)
N = 100

def y(x, theta, r):
    return (1/math.sin(theta))*(r-x*math.cos(theta))

fig, ax = plt.subplots()
ax.set_ylim(ylim)
for point in set:
    ax.scatter(point[0], point[1])


count = 0
maps = []
for point in set:
    param = []
    for theta in np.linspace(np.pi/N, np.pi, N):
        r = point[0] * math.cos(theta) + point[1] * math.sin(theta)
        param.append(theta)
        param.append(r)
        ax.plot(x_space,y(x_space,theta,r))
        plt.title(count)
        count += 1
        plt.pause(0.005)
    maps.append(param)


fig, ax = plt.subplots()
for i in range(len(set)):
    ax.scatter(maps[i][::2],maps[i][1::2])

for map in maps:
    map[1::2]


for i,r in enumerate(maps[0][1::2]):
    if abs(r - maps[1][(i*2)+1]) < 0.05:
        y_x = y(x, maps[0][i*2], maps[0][(i*2)+1])
        ax.plot(x, y_x)

plt.show()