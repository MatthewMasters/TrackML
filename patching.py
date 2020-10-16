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
import pp

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

    df = truth.copy()
    df = df.assign(r   = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(d   = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(a   = np.arctan2(df.y, df.x))
    df = df.assign(cosa= np.cos(df.a))
    df = df.assign(sina= np.sin(df.a))
    df = df.assign(phi = np.arctan2(df.z, df.r))

    fig, ax = plt.subplots(figsize=(12,12))
    ax = Axes3D(fig)
    #ax.scatter(xs=seed.a,
    #           ys=seed.r,
    #           zs=seed.z,
    #           c='gray',
    #           s=1)

    d_delta = 200
    overlap = 0.3
    its = math.ceil((1050/d_delta))
    for i in range(its):
        start = (i*d_delta)-(d_delta*overlap)
        if start < 0:
            start = 0
        end = (i*d_delta)+d_delta
        print(start, end)
        seed_tracks(event_id, df, start, end, ax)
    
    ax.set_xlabel('a')
    ax.set_ylabel('r')
    ax.set_zlabel('z  (mm)')
    plt.show()

    print('-'*50)
    print('Success!')
    t1 = time.time()
    print('Total time', (t1-t0)/60)
    print('-'*50)
    print('\n'*2)

##### IDEAS
# 
# Perform dbscan on different ranges of d that overlap
# Then determine which groups share hits
# Group smaller clusters together into final tracks
#
#
#

def seed_tracks(event_id, df, start_d, end_d, ax):
    seed = df.loc[df.d>start_d]
    seed = seed.loc[seed.d<end_d]
    N = len(seed)

    p = seed[['particle_id']].values.astype(np.int64)
    x,y,z,r,a,cosa,sina,phi = seed[['x', 'y', 'z', 'r', 'a', 'cosa', 'sina', 'phi']].values.astype(np.float32).T

    particle_ids = np.unique(p)
    particle_ids = particle_ids[particle_ids!=0]
    num_particle_ids = len(particle_ids)

    # do dbscan here =======================================
    data   = np.column_stack([a, z/r*0.1])

    _,l = dbscan(data, eps=0.01, min_samples=1,)


    #print(len(truth))
    #print(len(seed))
    #print(len(submission))
    #print(len(l))

    seed['l'] = pd.Series(l, index=seed.index)
    #print(seed)
    submission = pd.DataFrame(columns=['event_id', 'hit_id', 'track_id'],
        data=np.column_stack(([int(event_id),]*len(seed), seed.hit_id.values, l))
    ).astype(int)
    
    score = score_event_fast(seed, submission)
    print(score)

    predicted_tracks,counts = np.unique(l, return_counts=True)
    predicted_tracks = predicted_tracks[counts>1]

    for predicted_track in predicted_tracks[::100]:
        track_hits = seed[seed.l == predicted_track]
        ax.plot(xs=track_hits.a,
                ys=track_hits.r,
                zs=track_hits.z)

main()