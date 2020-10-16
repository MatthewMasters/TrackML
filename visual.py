import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trackml.dataset import load_event, load_dataset
import os, math

path_to_train = "train"
event_prefix = "event000001000"

hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

def helix(theta,r):
    x = r*math.cos(theta)-(2*theta)
    y = r*math.sin(theta)+500
    z = 2*np.pi*theta*5+280
    return x,y,z

helix_space = []

for theta in np.linspace(2.8,20*np.pi,400):
    x,y,z = helix(theta,500)
    helix_space.extend([x,y,z])

particle2 = particles.loc[particles.nhits == particles.nhits.max()].iloc[1]
print(particle2.particle_id)
#p_traj_surface['']
p_traj_surface2 = truth[truth.particle_id == particle2.particle_id][['tx', 'ty', 'tz']]

p_traj2 = (p_traj_surface2
          .append({'tx': particle2.vx, 'ty': particle2.vy, 'tz': particle2.vz}, ignore_index=True)
          .sort_values(by='tz'))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax = Axes3D(fig)

ax.plot(
    xs=p_traj2.tx,
    ys=p_traj2.ty,
    zs=p_traj2.tz, marker='o')

ax.plot(
    xs=helix_space[::3],
    ys=helix_space[1::3],
    zs=helix_space[2::3], marker='o')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z  (mm) -- Detection layers')
plt.title('Trajectories of two particles as they cross the detection surface ($Z$ axis).')
plt.show()

x_s = hits.x.values
y_s = hits.y.values
z_s = hits.z.values