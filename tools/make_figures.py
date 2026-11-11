from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_bags import evaluate


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    records=[evaluate(p) for p in sorted(Path('data/original').glob('*.db3'))]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for i,(ax,r) in enumerate(zip(axes,records),1):
        ref=r['reference'];xy=r['xy']
        ax.plot(*zip(*ref),color='#a5aeb7',linestyle='--',linewidth=2,label='Reference')
        ax.plot(*zip(*xy),color='#087e82',linewidth=1.6,label='Recorded trajectory')
        ax.set(title=f'Path {i} · repetition 2',xlabel='x (m)',ylabel='y (m)',aspect='equal')
        ax.grid(alpha=.15);ax.legend(loc='best',fontsize=8)
    out=Path('docs/assets');out.mkdir(parents=True,exist_ok=True)
    for ext in ['svg','png','pdf']:fig.savefig(out/f'trajectories.{ext}',dpi=180)
    plt.close(fig)
    r=records[2];fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True,layout='constrained')
    for topic,color,label in [('/cmd_vel_mpc','#8493a2','Nominal MPC'),('/tyr/cmd_vel','#087e82','Executed command')]:
        rows=r['commands'][topic];t=[x[0] for x in rows]
        axes[0].plot(t,[x[1] for x in rows],label=label,color=color,linewidth=1)
        axes[1].plot(t,[x[2] for x in rows],label=label,color=color,linewidth=1)
    axes[0].set_ylabel('v (m/s)');axes[1].set_ylabel('r (rad/s)');axes[1].set_xlabel('Time relative to first odometry sample (s)')
    axes[0].legend();axes[0].set_title('Original Path 3, repetition 2: recorded command streams')
    for ax in axes:ax.grid(alpha=.15)
    for ext in ['svg','png','pdf']:fig.savefig(out/f'commands-path3.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
