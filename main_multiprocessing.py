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
    
    weights = [[1, 1.1274, 0.4242, 0.2332, 0.0036, 0.00819],
               [1, 1.1272, 0.4241, 0.2336, 0.0035, 0.00820]]
    test_scoring(weights,multiprocessing=False)
    #GA_eval()
    
    #create_submission()
    
    print('-'*50)
    print('Success!')
    t1 = time.time()
    print('Total time', (t1-t0)/60)
    print('-'*50)
    print('\n'*2)


def GA_eval():
    bounds = [(1,1.3), (0.3,0.5), (0.2,0.3), (0,0.005), (0,0.01)]
    results = differential_evolution(test_scoring, bounds)
    winning_weights = results.x
    winning_score = results.fun
    print(winning_weights)
    print(winning_score)


# Testing run for one or more events
def test_scoring(weights,multiprocessing=False):
    print('Weights: ')
    print('Barrel\t' + str(weights[0]))
    print('Caps  \t' + str(weights[1]))
    print('-'*50)
    #weights = np.insert(weights, 0, 1)
    event_prefix = "event00000%d"
    if multiprocessing:
        jobs = []
        pool=mp.Pool(processes=4) 
    else:
        test_dataset_submissions = []
        scores = []
    for event in [1000 + i for i in range(1)]:
        event_id = event_prefix % event
        hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_id))
        if multiprocessing:
            jobs.append(pool.apply_async(add_submission, args=(event,hits,weights,True,truth,True)))
        else:
            results = add_submission(event,hits,weights,True,truth,False)
            test_dataset_submissions.append(results[0])
            scores.append(results[1])
    
    if multiprocessing:
        test_dataset_submissions = [job.get()[0] for job in jobs]
        scores = [job.get()[1] for job in jobs]

    print('Avg of events: ')
    avg_score = sum(scores)/float(len(scores))
    print(avg_score)
    return avg_score

# Submission run for all test events
def create_submission():
    dataset_submissions = []
    jobs = []
    pool=mp.Pool(processes=4) 
    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):
        jobs.append(pool.apply_async(add_submission, args=(event_id,hits)))

    # Create submission file
    dataset_submissions = [job.get() for job in jobs]
    submission = pd.concat(dataset_submissions, axis=0)
    submission.to_csv('results/submission_predict2.csv.gz', index=False, compression='gzip')

def angle_predict(hits,m,i,rz_shift,eps,weights):
    aa = hits.a+m*(hits.r+0.000005*(hits.r**2))/1000*(i/2)/180*3.141
            
    hits['f0'] = np.sin(aa)
    hits['f1'] = np.cos(aa)
    
    hits_b = hits[hits.type == 'b']['hit_id']
    hits_c = hits[hits.type == 'c']['hit_id']

    ss = StandardScaler()
    X = ss.fit_transform(np.column_stack([hits.f0.values, hits.f1.values, hits.z1.values, hits.z2.values, hits.xr.values, hits.yr.values]))
    
    X_b = np.multiply(np.vstack([X[ex-1] for ex in hits_b.values]),weights[0])
    X_c = np.multiply(np.vstack([X[ex-1] for ex in hits_c.values]),weights[1])

    Xw = np.zeros(X.shape)

    Xw[hits_b.values-1] = X_b[range(len(hits_b.values))]
    Xw[hits_c.values-1] = X_c[range(len(hits_c.values))]

    eps = eps + (i*0.000005)

    _,labels = dbscan(Xw, eps=eps, min_samples=1, algorithm='auto', n_jobs=4)

    unique,reverse,count = np.unique(labels,return_counts=True,return_inverse=True)
    c = count[reverse]
    c[np.where(labels==0)]=0
    if abs(rz_shift) < 0.1:
        c[np.where(c>20)]=0
    else:
        c[np.where(c>8)]=0
    return (labels,c)

class Clusterer(object):
    def __init__(self, weights, eps=0.0035, rz_scale=1.0):
        self.eps = eps
        self.rz_scale = rz_scale
        self.weights = weights


    def predict(self, hits):
        hits.loc[(hits.volume_id == 8) | (hits.volume_id == 13) | (hits.volume_id == 17), 'type'] = 'b'
        hits.loc[(hits.type.isnull() == True), 'type'] = 'c'
        

        x = hits.x.values
        y = hits.y.values
        zi = hits.z.values

        jobs = []
        pool=mp.Pool(processes=4) 
        for rz_shift in [0]: #np.linspace(-15,15,9): #-27.5 is entire luminous region
            z = (zi + rz_shift) * self.rz_scale
            r = np.sqrt(x*x+y*y)
            hits['r']  = r
            d = np.sqrt(x*x+y*y+z*z)
            hits['d']  = d
            hits['a']  = np.arctan2(y,x)
            hits['xr'] = x/r
            hits['yr'] = y/r
            hits['z1'] = z/r
            hits['z2'] = z/d

            base_its = 232
            #start_its = int(rz_shift*5)
            start_its = 0
            m = 1
            for i in range(start_its,base_its):
                for m in [-1, 1]:
                    #m = m * (-1)
                    jobs.append(pool.apply_async(angle_predict, args=(hits,m,i,rz_shift,self.eps,self.weights)))

        results = [job.get() for job in jobs]

        labels, counts = results[0]

        for i in range(1,len(results)):
            l,c = results[i]
            idx = np.where((c-counts>0))[0]
            labels[idx] = l[idx] + labels.max()
            counts[idx] = c[idx]
        return labels

def runCluster(model,hits):
    return model.predict(hits)

def add_submission(event_id,hits,weights,return_score=False,truth=[],v=False):
    if v:
        print("Starting event %s" % event_id)
        print('Starting clustering...')
        t2 = time.time()

    if False: #Ensemble clustering
        arr_size = len(hits.hit_id.values)

        its = 5
        jobs = []
        pool=mp.Pool(processes=4) 

        for eps in np.linspace(0.0033, 0.0037, its):
            model = Clusterer(eps=eps,rz_scale=1)
            jobs.append(pool.apply_async(runCluster, args=(model, hits)))
        label_array = [job.get() for job in jobs]

        print(label_array)

        new_labels = {}
        label_start = 1
        for hit_idx in range(len(hits.hit_id.values)):
            if hit_idx in new_labels:
                continue
            else:
                hit_labels = []
                for label_set in label_array:
                    label = label_set[hit_idx]
                    all_label_hits = np.where(label_set == label)
                    hit_labels.extend(all_label_hits[0])
                new_labels[hit_idx] = label_start
                if len(set(hit_labels)) == 1:
                    new_labels[hit_labels[0]] = 0
                else:
                    for x in set(hit_labels):
                        count = hit_labels.count(x)
                        if count > math.floor(its/2):
                            new_labels[x] = label_start
                    label_start += 1
            
        labels = []
        for key in sorted(new_labels.keys()):
            labels.append(new_labels[key])
        one_submission = create_one_event_submission(event_id, hits, labels)

    if False: #Traditional clustering
        model = Clusterer(weights,eps=0.0035,rz_scale=0.5)
        labels = model.predict(hits)
        one_submission = create_one_event_submission(event_id, hits, labels)
    
    if True:
        pickel = 'pickel/pre-cluster0_5618_bf.pkl'
        #one_submission.to_pickle(pickel)
        one_submission = pd.read_pickle(pickel)
        #one_submission = pd.read_csv('bayes.csv')

    if v:
        t3 = time.time()
        print('Ending clustering...\nTime (min) %f\n' % ((t3-t2)/60))

    if False: # Backfitting
        if v:
            print('Starting backfitting...')
            t4 = time.time()
        for i in tqdm(range(8)):
            one_submission = extend(one_submission, hits)
        if v:
            t5 = time.time()
            print('Ending backfitting...\nTime (min) %f\n' % ((t5-t4)/60))

    if True: #Plots 5 ground truth tracks and all predicted tracks that intersect hits
        fig, ax = plt.subplots(figsize=(12,12))
        ax = Axes3D(fig)
        ground_truth_tracks = truth.groupby('particle_id').filter(lambda x: len(x) > 2)
        ground_truth_tracks = np.random.choice(ground_truth_tracks.particle_id.unique(), 1)
        plot_tracks_from_particle_id(ax,ground_truth_tracks,truth,hits,'grey')
        for track in ground_truth_tracks:
            assoc_predictions = []
            track_hits = truth[truth.particle_id == track][['hit_id']]
            print(track_hits)
            for track_hit in track_hits.itertuples():
                assoc_predictions.append(one_submission[one_submission.hit_id == track_hit.hit_id][['track_id']])
            assoc_predictions = pd.concat(assoc_predictions).track_id.unique()
            plot_tracks_from_submission(ax,assoc_predictions,one_submission,hits)
        show_3Dplot(ax)    


    if False: #Plots 5 predicted tracks and all ground truth tracks at intersecting hits
        fig, ax = plt.subplots(figsize=(12,12))
        ax = Axes3D(fig)
        submission_tracks = one_submission.groupby('track_id').filter(lambda x: len(x) > 2)
        submission_tracks = np.random.choice(submission_tracks.track_id.unique(), 5)
        plot_tracks_from_submission(ax,submission_tracks,one_submission,hits,'b')
        for track in submission_tracks:
            assoc_particles = []
            track_hits = one_submission[one_submission.track_id == track][['hit_id']]
            for track_hit in track_hits.itertuples():
                assoc_particles.append(truth[truth.hit_id == track_hit.hit_id][['particle_id']])
            assoc_particles = pd.concat(assoc_particles).particle_id.unique()
            plot_tracks_from_particle_id(ax,assoc_particles,truth,hits,'g')
        plt.show()

    if False: # Filters unrealistic tracks to 0, doesn't change score
        for track in one_submission.groupby('track_id').filter(lambda x: 4 > len(x) > 1).track_id.unique():
            p_hits = one_submission[one_submission.track_id == track][['hit_id']]
            p_traj = [hits[hits.hit_id == hit[0]][['x','y','z']] for hit in p_hits.itertuples()]
            p_traj = pd.concat(p_traj).sort_values(by='z')
            if len(p_traj) == 2:
                x0,y0,z0 = p_traj.iloc[0]
                x1,y1,z1 = p_traj.iloc[1]
                maxdis = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
            else:
                x0,y0,z0 = p_traj.iloc[0]
                x1,y1,z1 = p_traj.iloc[1]
                x2,y2,z2 = p_traj.iloc[2]
                dis01 = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
                dis12 = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
                maxdis = max([dis01, dis12])
            if maxdis > 1000:
                one_submission.loc[one_submission.track_id == track, 'track_id'] = 0

    if False: #Plot starting positions/initial hits from ground truth
        fig, ax = plt.subplots(figsize=(12,12))
        ax = Axes3D(fig)
        ground_truth_tracks = truth.particle_id.unique()
        hits0 = []
        for track in ground_truth_tracks:
            p_hits = truth[truth.particle_id == track][['hit_id']]
            p_hits_coords = pd.concat([hits[hits.hit_id == p_hit[0]][['x','y','z']] for p_hit in p_hits.itertuples()]).sort_values(by='z')
            print(p_hits_coords)
            if abs(p_hits_coords.iloc[0].z) < abs(p_hits_coords.iloc[-1].z):
                hit0 = p_hits_coords.iloc[0]
            else:
                hit0 = p_hits_coords.iloc[-1]
            print(hit0)
            hits0.append(hit0)
        hits0 = pd.concat(hits0)
        ax.scatter(
            xs=hits0.x,
            ys=hits0.y,
            zs=hits0.z)
        plt.show()

    if False: #Find single hit particles
        fig, ax = plt.subplots(figsize=(12,12))
        ax = Axes3D(fig)
        pids = truth.groupby('particle_id').filter(lambda x: len(x) < 2).particle_id.unique()
        coords = []
        for pid in pids:
            hit_id = truth[truth.particle_id == pid][['hit_id']]
            coord = hits[hits.hit_id == hit_id.values[0,0]][['x','y','z']]
            coords.append(coord)
        coords = pd.concat(coords)
        ax.scatter(
            xs=coords.x,
            ys=coords.y,
            zs=coords.z)
        show_3Dplot(ax)

    if True:
        score = score_event_fast(truth, one_submission)
        print('Event ID', event_id, score)
        print('-'*50)
        return one_submission, score
    else:
        print('Event ID', event_id)
        return one_submission


main()
