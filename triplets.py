import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from trackml.dataset import load_event, load_dataset
from trackml_utils import score_event_fast, show_3Dplot
from mpl_toolkits.mplot3d import Axes3D
import os

path_to_test = "test"
path_to_train = "train"
event_prefix = "event00000%d"
event = 1000
event_id = event_prefix % event
hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_id))


class CalcMatrix(object):
	def __init__(self, scale=1):
		self.scale = scale

	def hit_matrix(self, hits):
		x = hits.x.values
		y = hits.y.values
		z = hits.z.values

		r  = np.sqrt(x**2+y**2)
		hits['r'] = r
		return hits

cm = CalcMatrix()
hit_matrix = cm.hit_matrix(hits)
x0 = hit_matrix[(hit_matrix.r < 50) & (hit_matrix.z > -600) & (hit_matrix.z < 600)][['x', 'y', 'z']]
x1 = hit_matrix[(hit_matrix.r > 50) | ((hit_matrix.z < -600) | (hit_matrix.z > 600))][['x', 'y', 'z']]

fig,ax = plt.subplots(figsize=(15,15))
ax = Axes3D(fig)
ax.scatter(
			xs=x0.x,
			ys=x0.y,
			zs=x0.z,
			c='b',
			edgecolors='face')

ax.scatter(
			xs=x1.x,
			ys=x1.y,
			zs=x1.z,
			c='r',
			edgecolors='face',
			s=1)


show_3Dplot(ax)