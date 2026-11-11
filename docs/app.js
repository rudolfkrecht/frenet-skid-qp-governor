/* Display only original archived data. No synthetic replay is shown as evidence. */
const NS='http://www.w3.org/2000/svg';
function el(name,attrs={},text){const n=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function plot(svg,series,{equal=false,dark=false,xLabel='x (m)',yLabel='y (m)'}={}){
  svg.replaceChildren();const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
  const pad={l:58,r:24,t:22,b:46},pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;
  const all=series.flatMap(s=>s.xy).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
  if(!all.length)return;
  let xmin=Math.min(...all.map(p=>p[0])),xmax=Math.max(...all.map(p=>p[0])),ymin=Math.min(...all.map(p=>p[1])),ymax=Math.max(...all.map(p=>p[1]));
  let dx=Math.max(xmax-xmin,.2),dy=Math.max(ymax-ymin,.2);xmin-=dx*.07;xmax+=dx*.07;ymin-=dy*.12;ymax+=dy*.12;
  if(equal){const scale=Math.max((xmax-xmin)/pw,(ymax-ymin)/ph);const cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;xmin=cx-scale*pw/2;xmax=cx+scale*pw/2;ymin=cy-scale*ph/2;ymax=cy+scale*ph/2;}
  const X=x=>pad.l+(x-xmin)/(xmax-xmin)*pw,Y=y=>H-pad.b-(y-ymin)/(ymax-ymin)*ph;
  const muted=dark?'#a2b6c3':'#73818a',grid=dark?'#2b4050':'#e6eae7';
  for(let i=0;i<=4;i++){let x=xmin+(xmax-xmin)*i/4,y=ymin+(ymax-ymin)*i/4;svg.append(el('line',{x1:X(x),x2:X(x),y1:pad.t,y2:H-pad.b,stroke:grid}),el('line',{x1:pad.l,x2:W-pad.r,y1:Y(y),y2:Y(y),stroke:grid}));svg.append(el('text',{x:X(x),y:H-pad.b+20,'text-anchor':'middle',fill:muted,'font-size':10},x.toFixed(1)),el('text',{x:pad.l-10,y:Y(y)+4,'text-anchor':'end',fill:muted,'font-size':10},y.toFixed(2)));}
  for(const s of series){const d=s.xy.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(2)+','+Y(p[1]).toFixed(2)).join(' ');svg.append(el('path',{d,fill:'none',stroke:s.color||'#087e82','stroke-width':s.width||2,'stroke-dasharray':s.dash||'','stroke-linejoin':'round'}));}
  svg.append(el('text',{x:pad.l+pw/2,y:H-6,'text-anchor':'middle',fill:muted,'font-size':11},xLabel),el('text',{x:16,y:pad.t+ph/2,'text-anchor':'middle',transform:`rotate(-90 16 ${pad.t+ph/2})`,fill:muted,'font-size':11},yLabel));
}
function update(index){const r=window.ARCHIVE[index];document.querySelector('#runTitle').textContent=`Path ${index+1} · repetition 2`;document.querySelector('#rms').innerHTML=(r.rms_cte_m*100).toFixed(3)+' <span>cm</span>';document.querySelector('#max').innerHTML=(r.max_cte_m*100).toFixed(3)+' <span>cm</span>';document.querySelector('#samples').textContent=r.n_odom.toLocaleString('en');document.querySelector('#runFile').textContent=r.run+'.db3';document.querySelectorAll('[data-path]').forEach(b=>{const active=Number(b.dataset.path)===index;b.classList.toggle('selected',active);b.setAttribute('aria-pressed',String(active));});plot(document.querySelector('#trajectoryPlot'),[{xy:r.reference,color:'#a2acb3',dash:'6 5',width:2.3},{xy:r.xy,color:'#087e82',width:1.8}],{equal:true});plot(document.querySelector('#commandPlot'),[{xy:r.commands['/cmd_vel_mpc'].map(p=>[p[0],p[2]]),color:'#a2acb3',width:1.4},{xy:r.commands['/tyr/cmd_vel'].map(p=>[p[0],p[2]]),color:'#087e82',width:1.4}],{xLabel:'Time (s)',yLabel:'r (rad/s)'});}
document.querySelectorAll('[data-path]').forEach(b=>b.addEventListener('click',()=>update(Number(b.dataset.path))));
if(window.ARCHIVE?.length===3){const r=window.ARCHIVE[2];plot(document.querySelector('#heroPlot'),[{xy:r.reference,color:'#778c9c',dash:'6 5',width:3},{xy:r.xy,color:'#73c9bc',width:2}],{equal:true,dark:true});update(2);}
