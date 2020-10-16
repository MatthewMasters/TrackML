import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster.dbscan_ import dbscan
import hdbscan
from trackml.dataset import load_event, load_dataset
from trackml_utils import score_event_fast, show_3Dplot
import math, time
import multiprocessing as mp


path_to_test = "data/test"
path_to_train = "data/train_1"


def main():
	print('-'*50)
	print('Starting job...')
	print('Remember, dont CTRL-C')
	print('-'*50)
	t0 = time.time()
	
	
	test_scoring()
	#create_submission()
	#visualize_prediction_v_truth()

	print('-'*50)
	print('Success!')
	t1 = time.time()
	print('Total time', t1-t0)
	print('-'*50)

def plot_tracks(ax,tracks,submission,hits):
	for predicted_track in tracks: #[1::40]
		p_hits = (submission[submission.track_id == predicted_track][['hit_id']])
		# Write so that it finds x,y,z from hit id
		frames = []
		for p_hit in p_hits.hit_id.values:
			frames.append(hits[hits.hit_id == p_hit][['x', 'y', 'z']])
		p_traj = pd.concat(frames).sort_values(by='z')
		ax.plot(
			xs=p_traj.x,
			ys=p_traj.y,
			zs=p_traj.z)

def visualize_prediction_v_truth():
	fig,ax = plt.subplots(figsize=(15,15))
	ax = Axes3D(fig)
	event = 1000
	event_prefix = "event00000%d"
	weights = [0.1,0.01,0.1,1,1]
	event_id = event_prefix % event
	hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_id))
	one_submission, score = add_submission(event,hits,weights,True,truth)
	plot_tracks(ax,one_submission.track_id.unique(),hits)
	show_3Dplot(ax)
	

#  Uncoil helices, parameterize tracks, cluster
class Clusterer(object):
	def __init__(self, eps=0.0035, rz_scale=1):
		self.eps = eps
		self.rz_scale = rz_scale

	def predict(self, hits, weights):
		x = hits.x.values
		y = hits.y.values
		z = self.rz_scale * hits.z.values

		r  = np.sqrt(x**2+y**2)
		d  = np.sqrt(x**2+y**2+z**2)
		a  = np.arctan2(y,x)
		zr = z/r
		dr = d/r
		hits['d'] = d

		w0, w1, w2, w3, w4 = weights

		ss = StandardScaler()

		results = []
		dzi = -0.00010
		for step in [11]:#range(21): #0.00060/121/-60
			dz = dzi + (step*0.00001)
			f0 = w0*(a + (dz*z*np.sign(z)))
			f1 = w1*(zr)
			f2 = w2*(f0/zr)
			f3 = w3*(1/zr)
			f4 = w4*(f2+f3)

			X = ss.fit_transform(np.column_stack([f0, f1, f2, f3, f4]))

			eps = self.eps - (abs(step-10)*0.000015)

			_,labels = dbscan(X, eps=eps, min_samples=1, algorithm='auto', n_jobs=4)
			
			unique,reverse,count = np.unique(labels,return_counts=True,return_inverse=True)
			c = count[reverse]
			c[np.where(labels==0)]=0
			c[np.where(c>20)]=0
			results.append((labels,c))

		labels, counts = results[0]

		for i in range(1,len(results)):
			l,c = results[i]
			idx = np.where((c-counts>0))[0]
			labels[idx] = l[idx] + labels.max()
			counts[idx] = c[idx]

		return labels

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

def find_incomplete_tracks(tracks, hits):
	flagged_tracks = []
	for track in tracks.track_id.unique():
		p_hits = (tracks[tracks.track_id == track][['hit_id']])
		c = 0
		cc = 0
		for p_hit in p_hits.hit_id.values:
			p_traj = hits[hits.hit_id == p_hit][['x', 'y', 'z']]
			r = math.sqrt(p_traj.x.values**2 + p_traj.y.values**2)
			c += 1
			#if r > 400:
			#	cc += 1
			if r < 1000 and -2800 < p_traj.z.values < 2800:
				cc += 1
		if c == cc and c > 1:
			flagged_tracks.append(track)
	return flagged_tracks

def line_func(t,p1,p2):
	x = p1.x + t*(p2.x-p1.x)
	y = p1.y + t*(p2.y-p1.y)
	z = p1.z + t*(p2.z-p1.z)

	return x,y,z

def point_line_dis(p1,p2,x0): #p1 and p2 define the line, x0 is the point to test
	t = -(np.dot(np.subtract(p1,x0),np.subtract(p2,p1))/((abs(np.subtract(p2,p1)))**2))
	d = np.sqrt(((p1.x-x0.x)+(p2.x-p1.x)*t)**2+((p1.y-x0.y)+(p2.y-p1.y)*t)**2+((p1.z-x0.z)+(p2.z-p1.z)*t)**2)
	return d

def backfit_track(tracks,submission,hits):
	for track in tracks:
		p_hits = (submission[submission.track_id == track][['hit_id']])
		traj_hits = []
		for p_hit in p_hits.hit_id.values:
			traj_hits.append(hits[hits.hit_id == p_hit][['x', 'y', 'z', 'd']])
		p_traj = pd.concat(traj_hits).sort_values(by='z')
		if abs(p_traj.iloc[0].d) < abs(p_traj.iloc[-1].d):
			p1 = p_traj.iloc[0]
			p2 = p_traj.iloc[1]
		else:
			p1 = p_traj.iloc[-1]
			p2 = p_traj.iloc[-2]
		hits['xsign'] = np.sign(hits['x']) == np.sign(p2.x)
		hits['ysign'] = np.sign(hits['y']) == np.sign(p2.y)
		hits['zsign'] = np.sign(hits['z']) == np.sign(p2.z)
		quad_hits = hits[(hits.xsign == True) & (hits.ysign == True) & (hits.zsign == True)][['hit_id','x', 'y', 'z']]
		#dis_mat = {}
		#for idx,hit in quad_hits.iterrows():
		#	hit_id,x,y,z = hit
		#	dis = point_line_dis([p1.x,p1.y,p1.z],[p2.x,p2.y,p2.z],[x,y,z])
		#	dis_mat[hit_id] = dis
		#closest = min(dis_mat, key=dis_mat.get)
		#submission[submission.track_id == track][['hit_id']] = closest
	return submission

def test_scoring():
	event_prefix = "event00000%d"
	jobs = [] 
	weights = [0.1,0.01,0.1,1,1]
	pool=mp.Pool(processes=1) 
	for event in [1000 + i for i in range(1)]:
		event_id = event_prefix % event
		hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_id))
		jobs.append(pool.apply_async(add_submission, args=(event,hits,weights,True,truth)))
	
	test_dataset_submissions = [job.get()[0] for job in jobs]
	scores = [job.get()[1] for job in jobs]

	print('Avg of events: ')
	print(sum(scores)/float(len(scores)))


def add_submission(event_id,hits,weights=[1,1,1,1,1],return_score=False,truth=[]):
	model = Clusterer(eps=0.0035,rz_scale=1.7)
	labels = model.predict(hits,weights)
	one_submission = create_one_event_submission(event_id, hits, labels)

	t8 = time.time()
	flagged_tracks = find_incomplete_tracks(one_submission, hits)
	t9 = time.time()
	print('flag time (min) ', (t9-t8)/60)

	#fig,ax = plt.subplots(figsize=(15,15))
	#ax = Axes3D(fig)
	#plot_tracks(ax,set(flagged_tracks),one_submission,hits)
	#show_3Dplot(ax)

	backfit_submission = backfit_track(flagged_tracks,one_submission,hits)

	if return_score:
		score = score_event_fast(truth, backfit_submission)
		print('Event ID', event_id, score)
		return backfit_submission, score
	else:
		print('Event ID', event_id)
		return backfit_submission


def create_submission():
	dataset_submissions = []
	jobs = []
	pool=mp.Pool(processes=16) 
	for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):
		jobs.append(pool.apply_async(add_submission, args=(event_id,hits)))

	# Create submission file
	dataset_submissions = [job.get() for job in jobs]
	submission = pd.concat(dataset_submissions, axis=0)
	submission.to_csv('results/submission.csv.gz', index=False, compression='gzip')


main()
