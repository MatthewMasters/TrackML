import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster.dbscan_ import dbscan
from scipy.optimize import differential_evolution
from trackml.dataset import load_event, load_dataset
from trackml_utils import score_event_fast, extend, create_one_event_submission
from trackml_visual import show_3Dplot, plot_tracks_from_particle_id, plot_tracks_from_submission
import math, time
import multiprocessing as mp
from tqdm import tqdm

path_to_test = "data/test"
path_to_train = "data/train_1"

def main():
    print('\n')
    print('-'*50)
    print('Starting job...')
    print('-'*50)
    t0 = time.time()

    event_id = '000001000'
    event_path = os.path.join(path_to_train, 'event'+event_id)
    hits, cells, particles, truth = load_event(event_path)

    truth = truth.merge(hits, on=['hit_id'], how='left')
    
    grid_delta = 20
    grid_plane_size = int(2060/grid_delta)
    grid_z_size = int(5920/grid_delta)

    a, b = (grid_plane_size/2)+1, (grid_plane_size/2)+1
    r = grid_plane_size/2

    y,x = np.ogrid[-a:grid_plane_size-a, -b:grid_plane_size-b]
    mask = x*x + y*y <= r*r

    array = np.zeros((grid_plane_size, grid_plane_size, grid_z_size))
    
    #for i in range(grid_z_size):
    #    array[:,:,i][mask] = 255

    array[mask,:] = 1

    print(array.shape)

    x,y,z = np.nonzero(array)
    print(x)
    print(y)
    print(z)

    fig, ax = plt.subplots(figsize=(12,12))
    ax = Axes3D(fig)

    skipstep = 100#50000

    ax.scatter(
        xs=x[::skipstep],
        ys=y[::skipstep],
        zs=z[::skipstep])

    plt.show()

    #array_stack = np.copy(array)

    #for _ in range(grid_z_size):
    #    print(array_stack.shape)
    #    array_stack = np.dstack((array_stack, array))

    #print(array_stack.shape)

    #plt.imshow(array)
    #plt.show()

    #print(array)

    print('-'*50)
    print('Success!')
    t1 = time.time()
    print('Total time', (t1-t0)/60)
    print('-'*50)
    print('\n'*2)

main()