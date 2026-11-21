"""Ten-minute scheduler recovery, early checkpoint release and result reporting."""
import csv,json,subprocess,datetime,fcntl,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
lock=(OUT/'watch.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit
record=json.loads((OUT/'submission.json').read_text());now=datetime.datetime.now(datetime.timezone.utc).isoformat()
# Training tasks moved off ckpt-all to dedicated GPUs keep their task index under a new job id.
moved=record.get('moved_train',{})
ids=[record['smoke_job'],record['train_job']]+[j['job'] for j in record['eval_jobs']]+list(moved.values())
def command(args):return subprocess.run(args,capture_output=True,text=True,timeout=60)
q=command(['squeue','-r','-h','-j',','.join(ids),'-o','%i|%T|%N|%R']);q.check_returncode()
active={a:dict(state=b,node=c,reason=d) for a,b,c,d in (l.split('|',3) for l in q.stdout.splitlines() if l.count('|')>=3)}
a=command(['sacct','-j',','.join(ids),'-X','-n','-P','--format=JobID,State']);a.check_returncode()
ended={x[0]:x[1].split()[0] for x in (l.split('|') for l in a.stdout.splitlines()) if len(x)>1}
retryfile=OUT/'requeues.json';retries=json.loads(retryfile.read_text()) if retryfile.exists() else {}
releasefile=OUT/'released_checkpoints.json';released=json.loads(releasefile.read_text()) if releasefile.exists() else {}
transient={'PREEMPTED','NODE_FAIL','BOOT_FAIL','TIMEOUT'};issues=[];rows=[]
def runpath(task,smoke=False):
 setting,algo,seed,variant=task
 return OUT/('smoke' if smoke else 'runs')/setting/(algo+'__'+variant)/f'seed_{seed}'
def inspect(jid,marker,kind,task,log):
 state='COMPLETED' if marker.exists() else active.get(jid,{}).get('state',ended.get(jid,'UNKNOWN'))
 if state=='COMPLETED' and not marker.exists():state='MISSING_OUTPUT'
 row=dict(job=jid,kind=kind,task=task,state=state,node=active.get(jid,{}).get('node',''))
 if active.get(jid,{}).get('reason')=='DependencyNeverSatisfied':
  row['state']='DEPENDENCY_FAILED'
 if state in transient:
  r=command(['scontrol','requeue',jid]);row['requeue_returncode']=r.returncode
  if r.returncode==0:row['state']='REQUEUED';retries[jid]=retries.get(jid,0)+1
  else:row['error']=r.stderr[-1000:]
 if row['state'] not in {'COMPLETED','RUNNING','PENDING','CONFIGURING','COMPLETING','REQUEUED'}:
  row['log_tail']=log.read_text()[-2000:] if log.exists() else '';issues.append(row.copy())
 rows.append(row)
# The smoke array was cancelled on 2026-10-19; training and evals no longer wait on it.
for i,task in enumerate([] if record.get('smoke_cancelled') else record['smoke_tasks']):
 jid=f"{record['smoke_job']}_{i}";inspect(jid,runpath(task,True)/'smoke_evals/step_0000002/COMPLETED','smoke',task,OUT/f'smoke_{jid}.log')
for i,task in enumerate(record['train_tasks']):
 jid=moved.get(str(i),f"{record['train_job']}_{i}");inspect(jid,runpath(task)/'TRAINED','train',task,OUT/f'train_{jid}.log')
for j in record['eval_jobs']:
 task=j['task'];i=j['train_task'];root=runpath(task)
 for k in range(10):
  idx=i*10+k;jid=f"{j['job']}_{idx}";ck=root/f'step_{(k+1)*1000:07d}.pt';marker=root/'checkpoint_evals_n64'/ck.stem/'COMPLETED'
  # The initial dependency is the corresponding complete training task. Once an
  # atomic checkpoint exists, only the successful smoke gate remains necessary.
  if ck.exists() and not marker.exists() and jid not in released and active.get(jid,{}).get('state')=='PENDING':
   r=command(['scontrol','update',f'JobId={jid}','Dependency='+('' if record.get('smoke_cancelled') else f"afterok:{record['smoke_job']}")])
   if r.returncode==0:released[jid]=dict(checkpoint=str(ck),released_utc=now)
   else:issues.append(dict(job=jid,error='Early release failed: '+r.stderr[-1000:]))
  inspect(jid,marker,'eval',task,OUT/f'eval_{jid}.log')
retryfile.write_text(json.dumps(retries,indent=2));releasefile.write_text(json.dumps(released,indent=2))
counts={kind:{state:sum(r['kind']==kind and r['state']==state for r in rows) for state in sorted({r['state'] for r in rows if r['kind']==kind})} for kind in ['smoke','train','eval']}
health=dict(checked_utc=now,counts=counts,issues=issues,jobs=rows,interval_minutes=10)
(OUT/'health.json').write_text(json.dumps(health,indent=2))
# Keep a per-seed/per-split audit, choosing one whole checkpoint curve using argmax.
metrics=[]
for task in record['train_tasks']:
 for marker in runpath(task).glob('checkpoint_evals_n64/step_*/COMPLETED'):
  with (marker.parent/'metrics.csv').open() as f:metrics.extend(csv.DictReader(f))
def writecsv(path,data):
 if not data:return
 tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 tmp.replace(path)
writecsv(OUT/'all_checkpoint_results.csv',metrics)
groups={}
for r in metrics:groups.setdefault(tuple(r[k] for k in ['setting','algo','variant','train_seed','split']),[]).append(r)
selected=[];selection=[]
for key,rs in sorted(groups.items()):
 eligible=[r for r in rs if r['strategy']=='max' and (r['algo'] not in ['bc_unet','dt_bon'] or int(r['n'])==1)]
 if rs[0]['algo'] in ['bc_unet','dt_bon']:winner=min(eligible,key=lambda r:(-float(r['success_rate']),int(r['checkpoint']),int(r['n'])))
 else:
  # ST ties at the max success are broken by higher N=1 success, then smallest N, then earliest step.
  top=max(float(r['success_rate']) for r in eligible);n1={r['checkpoint']:float(r['success_rate']) for r in eligible if int(r['n'])==1}
  winner=min((r for r in eligible if float(r['success_rate'])==top),key=lambda r:(-n1.get(r['checkpoint'],-1.),int(r['n']),int(r['checkpoint'])))
 selected.extend(r for r in rs if r['checkpoint']==winner['checkpoint'])
 selection.append(dict(zip(['setting','algo','variant','train_seed','split'],key),checkpoint=int(winner['checkpoint']),selection_n=int(winner['n']),selection_success=float(winner['success_rate']),selection_strategy='max',complete_checkpoint_sweep=len({r['checkpoint'] for r in rs})==10))
writecsv(OUT/'selected_results.csv',selected);(OUT/'selection.json').write_text(json.dumps(selection,indent=2))
# An event record is produced only when results or actionable issue contents change.
event_payload=dict(counts=counts,issues=issues,selection=selection)
fingerprint=hashlib.sha256(json.dumps(event_payload,sort_keys=True).encode()).hexdigest();fp=OUT/'last_update_fingerprint'
if not fp.exists() or fp.read_text()!=fingerprint:
 with (OUT/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(checked_utc=now,**event_payload))+'\n')
 fp.write_text(fingerprint)
print(json.dumps(dict(checked_utc=now,counts=counts,issues=len(issues),selected_curves=len(selection))),flush=True)
raise SystemExit(1 if issues else 0)
