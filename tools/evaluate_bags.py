"""

CTE is the unsigned Euclidean distance from every recorded /tyr/odom position
to its nearest /generated_path line segment, with no alignment, time trimming,
or pooling of repetitions. The bag contains both streams in the map frame.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import numpy as np


class CDR:
    def __init__(self, data):
        if bytes(data[:2]) != b'\x00\x01':
            raise ValueError('only little-endian CDR encapsulation is supported')
        self.data, self.offset = data, 4

    def value(self, fmt, alignment):
        self.offset=4+((self.offset-4+alignment-1)//alignment)*alignment
        value=struct.unpack_from('<'+fmt,self.data,self.offset)[0]
        self.offset+=struct.calcsize(fmt)
        return value

    def string(self):
        length=self.value('I',4)
        value=self.data[self.offset:self.offset+length-1].decode()
        self.offset+=length
        return value

    def header(self):
        self.value('i',4);self.value('I',4)
        return self.string()

    def pose(self):
        return [self.value('d',8) for _ in range(7)]


def messages(con,topic):
    return con.execute('SELECT timestamp,data FROM messages JOIN topics '
                       'ON messages.topic_id=topics.id WHERE name=? ORDER BY timestamp',(topic,))


def evaluate(path):
    con=sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True)
    if con.execute('PRAGMA quick_check').fetchone()[0]!='ok':
        raise ValueError(f'corrupt database: {path}')
    points=[];frames=set();times=[]
    for ts,data in messages(con,'/tyr/odom'):
        r=CDR(data);frames.add(r.header());r.string()
        points.append(r.pose()[:2]);times.append(ts/1e9)
    paths=list(messages(con,'/generated_path'))
    if not points or not paths:
        raise ValueError('required odometry or reference is missing')
    if len(paths)!=1:
        raise ValueError('expected one static reference; handle updates explicitly')
    r=CDR(paths[0][1]);frame=r.header();reference=[]
    for _ in range(r.value('I',4)):
        if r.header()!=frame: raise ValueError('reference pose frame mismatch')
        reference.append(r.pose()[:2])
    if frames!={frame}: raise ValueError('odometry/reference frame mismatch')
    xy=np.asarray(points);ref=np.asarray(reference)
    if len(ref)<2: raise ValueError('reference needs two points')
    a=ref[:-1];v=np.diff(ref,axis=0);vv=np.maximum(np.sum(v*v,axis=1),1e-20)
    errors=[]
    for chunk in np.array_split(xy,max(1,int(np.ceil(len(xy)/500)))):
        w=chunk[:,None,:]-a[None,:,:]
        t=np.clip(np.sum(w*v,axis=2)/vv,0,1)
        delta=w-t[:,:,None]*v
        errors.extend(np.sqrt(np.sum(delta*delta,axis=2).min(axis=1)))
    errors=np.asarray(errors)
    t0=times[0]
    commands={}
    for topic in ['/cmd_vel_mpc','/tyr/cmd_vel']:
        values=[]
        for ts,data in messages(con,topic):
            r=CDR(data);twist=[r.value('d',8) for _ in range(6)]
            values.append([ts/1e9-t0,twist[0],twist[5]])
        commands[topic]=values
    con.close()
    return dict(run=path.stem,frame=frame,n_odom=len(xy),
                duration_s=times[-1]-times[0],rms_cte_m=float(np.sqrt(np.mean(errors**2))),
                max_cte_m=float(errors.max()),mean_cte_m=float(errors.mean()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                reference=ref.tolist(),xy=xy.tolist(),t=[t-t0 for t in times],
                cte=errors.tolist(),commands=commands)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('data/original'))
    parser.add_argument('--output',type=Path,default=Path('data/derived'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    summaries=[];web=[]
    for path in sorted(args.input.glob('*.db3')):
        result=evaluate(path)
        summary={k:v for k,v in result.items() if k not in ['reference','xy','t','cte','commands']}
        summaries.append(summary)
        with (args.output/(path.stem+'_cte.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['t_s','x_m','y_m','cte_m'])
            w.writerows((t,*xy,e) for t,xy,e in zip(result['t'],result['xy'],result['cte']))
        for topic,rows in result['commands'].items():
            with (args.output/(path.stem+'_'+topic.strip('/').replace('/','_')+'.csv')).open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['t_s','v_mps','r_radps']);w.writerows(rows)
        # Only decimate display traces. Metrics above use all odometry samples.
        step=max(1,len(result['xy'])//700)
        web.append(dict(summary,reference=result['reference'],xy=result['xy'][::step],
                        cte=list(zip(result['t'][::step],result['cte'][::step])),
                        commands={k:v[::max(1,len(v)//700)] for k,v in result['commands'].items()}))
    (args.output/'verified_metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
    if summaries:
        with (args.output/'verified_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    assets=Path('docs/assets');assets.mkdir(parents=True,exist_ok=True)
    (assets/'recordings.js').write_text('window.ARCHIVE = '+json.dumps(web)+';\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
