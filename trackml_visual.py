import pandas as pd
import matplotlib.pyplot as plt

# Visualization
def show_2Dplot(ax):
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.show()

def show_3Dplot(ax):
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim([-1000,1000])
    ax.set_ylim([-1000,1000])
    ax.set_zlim([-3000,3000])
    ax.set_zlabel('Z  (mm) -- Detection layers')
    plt.show()


def plot_tracks(ax,tracks,submission,hits,color=None):
    for predicted_track in tracks[1::]:
        p_hits = (submission[submission.track_id == predicted_track][['hit_id']])
        frames = []
        for p_hit in p_hits.hit_id.values:
            frames.append(hits[hits.hit_id == p_hit][['x', 'y', 'z']])
        p_traj = pd.concat(frames).sort_values(by='z')
        if color:
            ax.plot(
                xs=p_traj.x,
                ys=p_traj.y,
                zs=p_traj.z,
                c=color)
        else:
            ax.plot(
                xs=p_traj.x,
                ys=p_traj.y,
                zs=p_traj.z)

def plot_tracks_from_submission(ax,submission_tracks,submission,hits,color=None):
    for track in submission_tracks:
        p_hits = (submission[submission.track_id == track][['hit_id']])
        frames = []
        for p_hit in p_hits.hit_id.values:
            frames.append(hits[hits.hit_id == p_hit][['x', 'y', 'z']])
        p_traj = pd.concat(frames).sort_values(by='z')
        if color:
            #ax.scatter(
                #xs=p_traj.x,
                #ys=p_traj.y,
                #zs=p_traj.z,
                #c=color)
            ax.plot(
                xs=p_traj.x,
                ys=p_traj.y,
                zs=p_traj.z,
                c=color,
                linewidth=5)
        else:
            ax.scatter(
                xs=p_traj.x,
                ys=p_traj.y,
                zs=p_traj.z)
            ax.plot(
                xs=p_traj.x,
                ys=p_traj.y,
                zs=p_traj.z)

def plot_tracks_from_truth(ax,truth_tracks,hits,color=None):
    for track in truth_tracks:
        p_hits = (truth[truth.particle_id == track][['hit_id']])
        frames = []
        for p_hit in p_hits.hit_id.values:
            frames.append(hits[hits.hit_id == p_hit][['x', 'y', 'z']])
        p_traj = pd.concat(frames).sort_values(by='z')
        if color:
            ax.plot(
                xs=p_traj.x,
                ys=p_traj.y,
                zs=p_traj.z,
                c=color)
        else:
            ax.plot(
                xs=p_traj.x,
                ys=p_traj.y,
                zs=p_traj.z)

            
def plot_tracks_from_particle_id(ax,particle_ids,truth,hits,color=None):
    for particle in particle_ids:
        if particle == 0:
            continue
        p_hits = (truth[truth.particle_id == particle][['hit_id']])
        frames = []
        for p_hit in p_hits.hit_id.values:
            frames.append(hits[hits.hit_id == p_hit][['x', 'y', 'z']])
        p_traj = pd.concat(frames).sort_values(by='z')
        if color:
            #ax.plot(
            #    xs=p_traj.x,
            #    ys=p_traj.y,
            #    zs=p_traj.z,
            #    c=color)
            ax.scatter(
                xs=p_traj.x,
                ys=p_traj.y,
                zs=p_traj.z,
                c=color,
                s=100)
        else:
            ax.plot(
                xs=p_traj.x,
                ys=p_traj.y,
                zs=p_traj.z)