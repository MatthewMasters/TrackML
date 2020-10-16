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
from sys import getsizeof


path_to_test = "data/test"
path_to_train = "train"

def main():
    print('\n')
    print('-'*50)
    print('Starting job...')
    print('-'*50)
    t0 = time.time()

    #event_id = '000001000'
    for event_num in range(1):
        event_id = '00000' + str(1000+event_num)
        print(event_id)
        event_path = os.path.join(path_to_train, 'event'+event_id)
        hits, cells, particles, truth = load_event(event_path)

        truth = truth.merge(hits, on=['hit_id'], how='left')
        #truth = truth.merge(particles,  on=['particle_id'], how='left')

        truth = truth.assign(r_abs = np.abs(np.sqrt( truth.x**2 + truth.y**2)))

        #df = truth.copy()
        #df = df.assign(r   = np.sqrt( df.x**2 + df.y**2))
        #df = df.assign(d   = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
        #df = df.assign(a   = np.arctan2(df.y, df.x))
        #df = df.assign(cosa= np.cos(df.a))
        #df = df.assign(sina= np.sin(df.a))
        #df = df.assign(phi = np.arctan2(df.z, df.r))
        #truth['d_key'] = truth.volume_id.map(str) + '-' + truth.layer_id.map(str) + '-' + truth.module_id.map(str)

        all_keys = []
        for particle_id in truth.particle_id.unique():
            p_hits = (truth[truth.particle_id == particle_id]).sort_values(by='r_abs')
            first_hit = p_hits.iloc[0]
            d_key = '%d' % (first_hit['volume_id']) #, first_hit['layer_id'], first_hit['module_id']
            all_keys.append(first_hit['volume_id'])

        x,c = np.unique(all_keys, return_counts=True)
        print(x,c)
        fig,ax = plt.subplots()
        ax.bar(x,c)
        plt.show()

        #arr_length = len(np.unique(truth.d_key.values))

        #print(arr_length)

    #X = np.zeros((arr_length, arr_length))
    #print(X)
    #print(getsizeof(X))

    #seed_tracks(event_id, df, 0.01008)

    #plot_hits_by_color(event_id, truth)
    #vol_8_hits(event_id, truth)

    print('-'*50)
    print('Success!')
    t1 = time.time()
    print('Total time', (t1-t0)/60)
    print('-'*50)
    print('\n'*2)



def plot_hits_by_color(event_id, truth):
    fig, ax = plt.subplots(figsize=(10,10))
    #ax = Axes3D(fig)    

    pids,c = np.unique(truth[['particle_id']].values.astype(np.int64), return_counts=True)
    pids = pids[np.where(c < 21)]
    for pid in pids[::500]:
        hits = truth[truth.particle_id == pid][['x','y','z']]
        ci = 0
        if len(hits) == 1:
            continue
        for idx,hit in hits.iterrows():
            color = 'C%d' % ci
            ax.scatter( hit.x,
                        hit.y,
                        #zs = hit.z,
                        color=color)
            ci += 1
            if ci == 10:
                break

    #ax.set_xlim([-1000,1000])
    #ax.set_ylim([-1000,1000])
    ax.set_xlim([-200,200])
    ax.set_ylim([-200,200])
    plt.show()


def vol_8_hits(event_id, truth):
    pids,c = np.unique(truth[['particle_id']].values.astype(np.int64), return_counts=True)
    pids = pids[np.where(c < 21)]
    i = 0
    ii = 0
    for pid in pids:
        hits = truth[truth.particle_id == pid][['volume_id', 'layer_id']]
        for idx,hit in hits.iterrows():
            if hit.volume_id == 8 and hit.layer_id == 2:
                ii += 1
        i += 1
    print('Total particles: %d' % i)
    print('Particles that hit vol 8: %d' % ii)
    print('Percentage %.3f' % (ii/i))


main()