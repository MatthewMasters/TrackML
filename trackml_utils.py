import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import pandas as pd

# Score
def score_event_fast(truth, submission):
    truth = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    df = truth.groupby(['track_id', 'particle_id']).hit_id.count().to_frame('count_both').reset_index()
    truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])

    df1 = df.groupby(['particle_id']).count_both.sum().to_frame('count_particle').reset_index()
    truth = truth.merge(df1, how='left', on='particle_id')
    df1 = df.groupby(['track_id']).count_both.sum().to_frame('count_track').reset_index()
    truth = truth.merge(df1, how='left', on='track_id')
    truth.count_both *= 2
    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].weight.sum()
    return score

def extend(submission,hits,limit=0.04, num_neighbours=18):
    df = submission.merge(hits,  on=['hit_id'], how='left')
    df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(arctan2 = np.arctan2(df.z, df.r))

    for angle in range(-90,90,1):
        print ('\r %f'%angle, end='',flush=True)
        #df1 = df.loc[(df.arctan2>(angle-0.5)/180*np.pi) & (df.arctan2<(angle+0.5)/180*np.pi)]
        df1 = df.loc[(df.arctan2>(angle-1.5)/180*np.pi) & (df.arctan2<(angle+1.5)/180*np.pi)]

        min_num_neighbours = len(df1)
        if min_num_neighbours<3: continue

        hit_ids = df1.hit_id.values
        x,y,z = df1[['x', 'y', 'z']].values.T
        r  = (x**2 + y**2)**0.5
        r  = r/1000
        a  = np.arctan2(y,x)
        c = np.cos(a)
        s = np.sin(a)
        #tree = KDTree(np.column_stack([a,r]), metric='euclidean')
        tree = KDTree(np.column_stack([c, s, r]))


        track_ids = list(df1.track_id.unique())
        num_track_ids = len(track_ids)
        min_length=3

        for i in range(num_track_ids):
            p = track_ids[i]
            if p==0: continue

            idx = np.where(df1.track_id==p)[0]
            if len(idx)<min_length: continue

            if angle>0:
                idx = idx[np.argsort( z[idx])]
            else:
                idx = idx[np.argsort(-z[idx])]


            ## start and end points  ##
            idx0,idx1 = idx[0],idx[-1]
            a0 = a[idx0]
            a1 = a[idx1]
            r0 = r[idx0]
            r1 = r[idx1]
            c0 = c[idx0]
            c1 = c[idx1]
            s0 = s[idx0]
            s1 = s[idx1]

            da0 = a[idx[1]] - a[idx[0]]  #direction
            dr0 = r[idx[1]] - r[idx[0]]
            direction0 = np.arctan2(dr0,da0)

            da1 = a[idx[-1]] - a[idx[-2]]
            dr1 = r[idx[-1]] - r[idx[-2]]
            direction1 = np.arctan2(dr1,da1)



            ## extend start point
            _,ns = tree.query([[c0, s0, r0]], k=min(num_neighbours, min_num_neighbours))
            ns = np.concatenate(ns)

            direction = np.arctan2(r0 - r[ns], a0 - a[ns])
            diff = 1 - np.cos(direction - direction0)
            ns = ns[(r0 - r[ns] > 0.01) & (diff < (1 - np.cos(limit)))]
            for n in ns: df.loc[df.hit_id == hit_ids[n], 'track_id'] = p

            ## extend end point
            _,ns = tree.query([[c1, s1, r1]], k=min(num_neighbours, min_num_neighbours))
            ns = np.concatenate(ns)

            direction = np.arctan2(r[ns] - r1, a[ns] - a1)
            diff = 1 - np.cos(direction - direction1)
            ns = ns[(r[ns] - r1 > 0.01) & (diff < (1 - np.cos(limit)))]
            for n in ns:  df.loc[df.hit_id == hit_ids[n], 'track_id'] = p

    #print ('\r')
    df = df[['event_id', 'hit_id', 'track_id']]
    return df

def extend_old(submission,hits):
    df = submission.merge(hits,  on=['hit_id'], how='left')
    df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(arctan2 = np.arctan2(df.z, df.r))

    for angle in range(-180,180,1):
        #df1 = df.loc[(df.arctan2>(angle-0.5)/180*np.pi) & (df.arctan2<(angle+0.5)/180*np.pi)]
        df1 = df.loc[(df.arctan2>(angle-1.0)/180*np.pi) & (df.arctan2<(angle+1.0)/180*np.pi)]

        min_num_neighbours = len(df1)
        if min_num_neighbours<4: continue

        hit_ids = df1.hit_id.values
        x = df1.x.values
        y = df1.y.values
        z = df1.z.values
        r  = (x**2 + y**2)**0.5
        r  = r/1000
        a  = np.arctan2(y,x)
        tree = KDTree(np.column_stack([a,r]))

        track_ids = list(df1.track_id.unique())
        num_track_ids = len(track_ids)
        min_length=3

        for i in range(num_track_ids):
            p = track_ids[i]
            if p==0: continue

            idx = np.where(df1.track_id==p)[0]
            if len(idx)<min_length: continue

            if angle>0:
                idx = idx[np.argsort( z[idx])]
            else:
                idx = idx[np.argsort(-z[idx])]


            ## start and end points  ##
            idx0,idx1 = idx[0],idx[-1]
            a0 = a[idx0]
            a1 = a[idx1]
            r0 = r[idx0]
            r1 = r[idx1]

            da0 = a[idx[1]] - a[idx[0]]  #direction
            dr0 = r[idx[1]] - r[idx[0]]
            direction0 = np.arctan2(dr0,da0) 

            da1 = a[idx[-1]] - a[idx[-2]]
            dr1 = r[idx[-1]] - r[idx[-2]]
            direction1 = np.arctan2(dr1,da1) 


            ## extend start point
            dis, ns = tree.query([[a0,r0]], k=min(20,min_num_neighbours))
            ns = np.concatenate(ns)
            direction = np.arctan2(r0-r[ns],a0-a[ns])
            ns = ns[(r0-r[ns]>0.01) &(np.fabs(direction-direction0)<0.04)]

            for n in ns:
                df.loc[ df.hit_id==hit_ids[n],'track_id' ] = p 

            ## extend end point
            dis, ns = tree.query([[a1,r1]], k=min(20,min_num_neighbours))
            ns = np.concatenate(ns)

            direction = np.arctan2(r[ns]-r1,a[ns]-a1)
            ns = ns[(r[ns]-r1>0.01) &(np.fabs(direction-direction1)<0.04)] 

            for n in ns:
                df.loc[ df.hit_id==hit_ids[n],'track_id' ] = p
    #print ('\r')
    df = df[['event_id', 'hit_id', 'track_id']]
    return df

def get_hits_from_particle_id(truth,id):
    p_traj = (truth[truth.particle_id == id]).sort_values(by='tz')
    return p_traj

def sample_particle_tracks(particles,N):
    sample_particles = particles.sample(N)
    return sample_particles

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission