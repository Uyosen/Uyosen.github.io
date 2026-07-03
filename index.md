---
layout: page
permalink: /
title: 首页
---

<style>
.hero{display:flex;align-items:center;justify-content:space-between;padding:44px 40px;background:linear-gradient(135deg,#080818 0%,#14143a 45%,#0b1f30 100%);border-radius:14px;margin-bottom:36px;color:#e0e0e0;position:relative;overflow:hidden;min-height:290px}
.hero-text{flex:0 0 55%;z-index:2}
.hero h1{font-size:2.7em;margin:0 0 10px;color:#fff;letter-spacing:3px}
.hero .subtitle{font-size:1em;opacity:.82;margin-bottom:18px;line-height:1.65}
.hero .tags{display:flex;flex-wrap:wrap;gap:9px}
.hero .tag{background:rgba(255,255,255,.09);padding:5px 14px;border-radius:20px;font-size:.84em;border:1px solid rgba(255,255,255,.13)}
#nn-canvas{position:absolute;top:0;right:0;width:55%;height:100%;z-index:1}
@media(max-width:720px){.hero{flex-direction:column;padding:28px 20px;min-height:240px}.hero-text{flex:none;width:100%}#nn-canvas{width:100%;opacity:.25}}
.edu-bar{display:flex;justify-content:center;gap:28px;flex-wrap:wrap;margin:12px 0 8px;font-size:.92em;color:#888}
.demo-wrap{margin:36px 0;padding:24px;background:#111827;border-radius:12px;overflow:hidden}
.demo-wrap h2{color:#60a5fa;margin-top:0;font-size:1.35em}
.demo-wrap .desc{color:#b0b8c8;font-size:.9em;margin-bottom:16px}
.gw-toolbar{display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.btn{padding:7px 16px;border:none;border-radius:6px;cursor:pointer;font-size:.88em;font-weight:600;transition:all .2s}
.btn-primary{background:#3b82f6;color:#fff}.btn-primary:hover{background:#2563eb}
.btn-green{background:#22c55e;color:#fff}.btn-green:hover{background:#16a34a}
.btn-gray{background:#4b5563;color:#fff}.btn-gray:hover{background:#6b7280}
.btn:disabled{opacity:.5;cursor:not-allowed}
.gw-info{display:flex;gap:16px;font-size:.88em;color:#9ca3af}
.gw-grid{display:inline-grid;gap:2px;padding:8px;background:#1f2937;border-radius:8px;margin:0 auto}
.gw-grid:focus{outline:2px solid #60a5fa;outline-offset:2px}
.gw-cell{width:48px;height:48px;display:flex;align-items:center;justify-content:center;font-size:18px;border-radius:4px;transition:background .12s}
.status-bar{margin-top:10px;font-size:.88em;color:#9ca3af;min-height:22px}
.sep{border:0;height:1px;background:linear-gradient(90deg,transparent,#333,transparent);margin:40px 0}
.exc-canvas-wrap{text-align:center;margin:12px 0}
.exc-canvas-wrap canvas{border-radius:8px;cursor:crosshair;max-width:100%}
.highlight-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;text-align:center;margin:16px 0}
.hl-card{padding:18px 10px;background:#f8f9fa;border-radius:8px;border:1px solid #e5e7eb}
.hl-card .num{font-size:1.8em;font-weight:700;color:#1a1a2e}
.hl-card .label{font-size:.82em;color:#666;margin-top:4px}
.footer-links{text-align:center;margin:28px 0}
.footer-links a{color:#3b82f6;font-size:1.05em;text-decoration:none}
.footer-links a:hover{text-decoration:underline}
</style>

<div class="hero">
<div class="hero-text">
<h1>吴悦晨</h1>
<p class="subtitle">从游戏通关到机器人操作，用人工智能赋予智能体真实世界的决策能力。</p>
</div>
<canvas id="nn-canvas"></canvas>
</div>

<hr class="sep">

<!-- ===================== GridWorld RL Demo ===================== -->
<div class="demo-wrap">
<h2>🎮 强化学习 playground</h2>
<p class="desc">在 8×8 网格地图中，从起点 🟦 到达终点 🟩，灰色方块为障碍物。<br>你可以亲自操控，也可以观看 AI 通过 Q-learning 训练后自动寻路——看看谁用的步数更少！</p>
<div class="gw-toolbar">
<button class="btn btn-primary" id="gw-human-btn" onclick="gwStartHuman()">🕹️ 人类挑战</button>
<button class="btn btn-green" id="gw-ai-btn" onclick="gwStartAI()">🤖 AI自动寻路</button>
<button class="btn btn-gray" onclick="gwReset()">↺ 重置</button>
<button class="btn btn-gray" onclick="gwRandomize()">🎲 随机地图</button>
<div class="gw-info"><span id="gw-steps">步数: 0</span><span id="gw-msg">点击上方按钮开始</span></div>
</div>
<div id="gw-board" class="gw-grid" tabindex="0"></div>
<div class="status-bar" id="gw-status">💡 提示: 人类用方向键移动，AI 经过 3000 轮 Q-learning 训练后自动寻路</div>
</div>

<hr class="sep">

<!-- ===================== Excavator Simulator ===================== -->
<div class="demo-wrap">
<h2>🤖 智能挖掘机模拟器</h2>
<p class="desc">2D 挖掘机机械臂交互演示。点击画布任意位置移动臂端，或观看 AI 自动执行「挖掘→提升→转运→卸料」完整作业流程。</p>
<div class="gw-toolbar">
<button class="btn btn-green" id="exc-auto-btn" onclick="excToggleAuto()">▶ AI自动作业</button>
<button class="btn btn-gray" onclick="excReset()">↺ 重置</button>
<span id="exc-msg" style="font-size:.88em;color:#9ca3af">💡 点击画布任意位置移动机械臂</span>
</div>
<div class="exc-canvas-wrap"><canvas id="exc-canvas" width="600" height="320"></canvas></div>
</div>

<div class="footer-links">
<a href="/cv">📄 查看完整简历 → 论文 · 专利 · 项目详情</a>
</div>

<script>
/* ========== GridWorld Q-Learning ========== */
(function(){
var GW=8,CS=48;
var grid=[
[0,0,0,0,0,0,1,0],
[0,1,1,0,1,0,0,0],
[0,0,0,0,1,0,1,0],
[0,1,0,1,0,0,0,0],
[0,1,0,0,0,1,0,0],
[0,0,0,1,0,0,0,1],
[0,1,0,0,0,1,0,0],
[0,0,0,1,0,0,0,2]];
var dr=[-1,1,0,0],dc=[0,0,-1,1];
var sr=0,sc=0,gr=7,gc=7;
var pr=sr,pc=sc,steps=0,running=false,isAI=false,Q={},animId=null;

function sk(r,c){return r*GW+c;}
function gq(r,c,a){var k=sk(r,c)*4+a;return Q[k]||0;}
function sq(r,c,a,v){Q[sk(r,c)*4+a]=v;}

function bfs(startR,startC){
var vis=[],q=[];
for(var i=0;i<GW;i++)vis.push(new Array(GW).fill(false));
vis[startR][startC]=true;q.push({r:startR,c:startC});
while(q.length>0){
var cur=q.shift();
for(var a=0;a<4;a++){
var nr=cur.r+dr[a],nc=cur.c+dc[a];
if(nr>=0&&nr<GW&&nc>=0&&nc<GW&&!vis[nr][nc]&&grid[nr][nc]!==1){
vis[nr][nc]=true;q.push({r:nr,c:nc});}}}
return vis;}

function generateRandomMap(){
var newGrid=[];
for(var i=0;i<GW;i++)newGrid.push(new Array(GW).fill(0));
var obsCount=Math.floor(Math.random()*8)+12;
for(var i=0;i<obsCount;i++){
var r=Math.floor(Math.random()*GW),c=Math.floor(Math.random()*GW);
newGrid[r][c]=1;}
var empty=[];
for(var i=0;i<GW;i++)for(var j=0;j<GW;j++)if(newGrid[i][j]===0)empty.push({r:i,c:j});
if(empty.length<2)return generateRandomMap();
var idx1=Math.floor(Math.random()*empty.length);
var idx2=(idx1+Math.floor(Math.random()*(empty.length-1))+1)%empty.length;
var start=empty[idx1],goal=empty[idx2];
newGrid[start.r][start.c]=0;newGrid[goal.r][goal.c]=0;
var reachable=bfs(start.r,start.c);
if(!reachable[goal.r][goal.c])return generateRandomMap();
grid=newGrid;sr=start.r;sc=start.c;gr=goal.r;gc=goal.c;}

function train(ep){
Q={};
for(var e=0;e<ep;e++){
var r=sr,c=sc;
for(var s=0;s<200;s++){
if(r===gr&&c===gc)break;
var eps=0.15,a;
if(Math.random()<eps){a=Math.floor(Math.random()*4);}
else{a=0;for(var i=1;i<4;i++)if(gq(r,c,i)>gq(r,c,a))a=i;}
var nr=r+dr[a],nc=c+dc[a];
if(nr<0||nr>=GW||nc<0||nc>=GW||grid[nr][nc]===1){nr=r;nc=c;}
var rw=0;if(nr===gr&&nc===gc)rw=100;else if(nr===r&&nc===c)rw=-5;else rw=-1;
var mx=gq(nr,nc,0);for(var j=1;j<4;j++){var v=gq(nr,nc,j);if(v>mx)mx=v;}
sq(r,c,a,gq(r,c,a)+0.1*(rw+0.95*mx-gq(r,c,a)));
r=nr;c=nc;
}}}
train(3000);

function draw(){
var b=document.getElementById('gw-board');
b.style.gridTemplateColumns='repeat('+GW+','+CS+'px)';
b.innerHTML='';
for(var r=0;r<GW;r++)for(var c=0;c<GW;c++){
var d=document.createElement('div');d.className='gw-cell';
if(grid[r][c]===1){d.style.background='#374151';d.textContent='⬛';}
else if(r===gr&&c===gc){d.style.background='#065f46';d.textContent='🏁';}
else if(r===sr&&c===sc&&!running){d.style.background='#1e3a5f';d.textContent='🚩';}
else{d.style.background='#1f2937';}
b.appendChild(d);}
// player
if(running||pr!==sr||pc!==sc){
var i=pr*GW+pc;
if(b.children[i]){b.children[i].style.background='#dc2626';b.children[i].textContent='🤖';}}
// show learned policy arrows after AI run
if(isAI&&!running&&pr===gr&&pc===gc){
for(var r2=0;r2<GW;r2++)for(var c2=0;c2<GW;c2++){
if(grid[r2][c2]!==0||(r2===gr&&c2===gc))continue;
var ba=0;for(var a2=1;a2<4;a2++)if(gq(r2,c2,a2)>gq(r2,c2,ba))ba=a2;
if(gq(r2,c2,ba)>0){var arrows=['↑','↓','←','→'];
var idx=r2*GW+c2;if(b.children[idx])b.children[idx].textContent=arrows[ba];
}}}}

function setMsg(t){document.getElementById('gw-msg').textContent=t;}
function setSteps(t){document.getElementById('gw-steps').textContent=t;}
function setStatus(t){document.getElementById('gw-status').textContent=t;}

function checkGoal(){
if(pr===gr&&pc===gc){
running=false;
setMsg('✅ 到达终点！共 '+steps+' 步');
setStatus(isAI?'🤖 AI 找到了最优路径！地图上显示学到的策略方向':'🎉 恭喜到达终点！试试点击「AI自动寻路」看看 AI 的最优路径');
draw();return true;}
return false;}

window.gwStartHuman=function(){
if(running)return;if(animId){clearTimeout(animId);animId=null;}
pr=sr;pc=sc;steps=0;running=true;isAI=false;
draw();setSteps('步数: 0');setMsg('🕹️ 用方向键/WASD 移动');
setStatus('🕹️ 方向键或 WASD 移动，避开障碍物到达绿色终点');
document.getElementById('gw-board').focus();};

window.gwStartAI=function(){
if(running)return;if(animId){clearTimeout(animId);animId=null;}
pr=sr;pc=sc;steps=0;running=true;isAI=true;
setMsg('🤖 AI 移动中...');setStatus('🧠 AI 经过 3000 轮 Q-learning 训练，正在执行最优策略...');
draw();
function mv(){
if(!running)return;
var ba=0;for(var a=1;a<4;a++)if(gq(pr,pc,a)>gq(pr,pc,ba))ba=a;
var nr=pr+dr[ba],nc=pc+dc[ba];
if(nr>=0&&nr<GW&&nc>=0&&nc<GW&&grid[nr][nc]!==1){pr=nr;pc=nc;}
steps++;setSteps('步数: '+steps);draw();
if(!checkGoal())animId=setTimeout(mv,250);}
animId=setTimeout(mv,400);};

window.gwReset=function(){
if(animId){clearTimeout(animId);animId=null;}
running=false;isAI=false;pr=sr;pc=sc;steps=0;
draw();setSteps('步数: 0');setMsg('点击上方按钮开始');
setStatus('💡 提示: 人类用方向键移动，AI 经过 3000 轮 Q-learning 训练后自动寻路');};

window.gwRandomize=function(){
if(animId){clearTimeout(animId);animId=null;}
running=false;isAI=false;
generateRandomMap();
train(3000);
pr=sr;pc=sc;steps=0;
draw();setSteps('步数: 0');setMsg('🎲 新地图已生成！');
setStatus('🧠 AI 已在新地图上重新训练，点击按钮开始挑战');};

document.addEventListener('keydown',function(e){
if(!running||isAI)return;
var a=-1;
switch(e.key){case'ArrowUp':case'w':case'W':a=0;break;case'ArrowDown':case's':case'S':a=1;break;
case'ArrowLeft':case'a':case'A':a=2;break;case'ArrowRight':case'd':case'D':a=3;break;}
if(a<0)return;e.preventDefault();
var nr=pr+dr[a],nc=pc+dc[a];
if(nr>=0&&nr<GW&&nc>=0&&nc<GW&&grid[nr][nc]!==1){pr=nr;pc=nc;}
steps++;setSteps('步数: '+steps);draw();checkGoal();});

draw();
})();

/* ========== Excavator Simulator ========== */
(function(){
var canvas=document.getElementById('exc-canvas'),ctx=canvas.getContext('2d');
var W=canvas.width,H=canvas.height;
var bx=155,by=H-48,L1=88,L2=82;
var sA=0.7,eA=1.0,tSA=0.7,tEA=1.0;
var autoOn=false,autoIdx=0,autoT=0,autoPause=0;
var trail=[];
var tbX=bx,trackOffset=0;

var seq=[
    {s:0.7,e:1.0,bx:200,w:0},
    {s:1.8,e:-0.3,bx:200,w:0},
    {s:1.6,e:-0.5,bx:200,w:25},
    {s:0.8,e:0.6,bx:200,w:0},
    {s:-0.3,e:1.5,bx:200,w:0},
    {s:-0.5,e:1.3,bx:460,w:0},
    {s:-0.3,e:1.0,bx:460,w:25},
    {s:0.5,e:0.5,bx:460,w:0},
    {s:0.7,e:1.0,bx:200,w:0}];

function ik(tx,ty){
var dx=tx-bx,dy=ty-(by-24);
var d=Math.sqrt(dx*dx+dy*dy);
var mr=L1+L2-3;
if(d>mr){var sc2=mr/d;dx*=sc2;dy*=sc2;d=mr;}
if(d<Math.abs(L1-L2)+5)d=Math.abs(L1-L2)+5;
var ce=(d*d-L1*L1-L2*L2)/(2*L1*L2);
ce=Math.max(-1,Math.min(1,ce));
var eAng=Math.acos(ce);
var at=Math.atan2(dx,dy);
var b=Math.atan2(L2*Math.sin(eAng),L1+L2*Math.cos(eAng));
var sAng=at-b;
return{s:sAng,e:eAng};}

function armPos(s,e){
var ex=bx+L1*Math.sin(s),ey=(by-24)+L1*Math.cos(s);
var ta=s-e;
var hx=ex+L2*Math.sin(ta),hy=ey+L2*Math.cos(ta);
return{bx:bx,by:by-24,ex:ex,ey:ey,hx:hx,hy:hy};}

function drawSeg(x1,y1,x2,y2,w,col){
ctx.lineCap='round';ctx.lineWidth=w+3;ctx.strokeStyle='#1a1a1a';
ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
ctx.lineWidth=w;ctx.strokeStyle=col;
ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();}

function draw(){
ctx.clearRect(0,0,W,H);
// sky gradient
var bg=ctx.createLinearGradient(0,0,0,H);
bg.addColorStop(0,'#0c1929');bg.addColorStop(1,'#162033');
ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
// ground
ctx.fillStyle='#3d2b1f';ctx.fillRect(0,H-48,W,48);
ctx.fillStyle='#2d4a1f';ctx.fillRect(0,H-50,W,6);
// dirt pile (dig area)
ctx.fillStyle='#5c3a1e';ctx.beginPath();
ctx.arc(340,H-48,22,Math.PI,0);ctx.fill();
// dump area marker
ctx.strokeStyle='rgba(255,200,50,0.25)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
ctx.strokeRect(440,H-110,100,62);ctx.setLineDash([]);
ctx.fillStyle='rgba(255,200,50,0.12)';ctx.fillRect(440,H-110,100,62);
// labels
ctx.font='11px sans-serif';ctx.textAlign='center';
ctx.fillStyle='rgba(255,150,100,0.55)';ctx.fillText('挖掘区',340,H-56);
ctx.fillStyle='rgba(255,200,50,0.55)';ctx.fillText('卸料区',490,H-115);
// tracks
ctx.fillStyle='#292524';ctx.fillRect(bx-42,by+8,12,26);
ctx.fillStyle='#292524';ctx.fillRect(bx+30,by+8,12,26);
for(var ti=0;ti<8;ti++){
ctx.fillStyle='#44403c';
ctx.fillRect(bx-42, by+10+ti*3.2, 10, 2);
ctx.fillRect(bx+32, by+10+(ti+trackOffset)%8*3.2, 10, 2);}
// base
ctx.fillStyle='#d97706';ctx.fillRect(bx-32,by-4,64,22);
ctx.fillStyle='#92400e';ctx.fillRect(bx-38,by+16,76,10);
ctx.fillStyle='#78350f';
ctx.beginPath();ctx.arc(bx-28,by+28,7,0,Math.PI*2);ctx.fill();
ctx.beginPath();ctx.arc(bx+28,by+28,7,0,Math.PI*2);ctx.fill();
ctx.fillRect(bx-38,by+24,76,6);
// cabin
ctx.fillStyle='#b45309';ctx.fillRect(bx-8,by-30,28,28);
ctx.fillStyle='#93c5fd';ctx.fillRect(bx-4,by-27,20,14);
// trail
for(var i=0;i<trail.length;i++){
var al=(i/trail.length)*0.35;
ctx.fillStyle='rgba(251,191,36,'+al+')';
ctx.beginPath();ctx.arc(trail[i].x,trail[i].y,2.5,0,Math.PI*2);ctx.fill();}
// arm
var p=armPos(sA,eA);
drawSeg(p.bx,p.by,p.ex,p.ey,13,'#f59e0b');
drawSeg(p.ex,p.ey,p.hx,p.hy,9,'#fbbf24');
// bucket
var ta=sA-eA,bw=16;
ctx.lineWidth=3;ctx.strokeStyle='#78350f';
ctx.beginPath();
ctx.moveTo(p.hx-bw*0.6*Math.cos(ta+0.7),p.hy-bw*0.6*Math.sin(ta+0.7));
ctx.lineTo(p.hx+12*Math.sin(ta),p.hy+12*Math.cos(ta));
ctx.lineTo(p.hx+bw*0.6*Math.cos(ta-0.7),p.hy+bw*0.6*Math.sin(ta-0.7));
ctx.stroke();
// joints
var joints=[[p.bx,p.by,6],[p.ex,p.ey,5],[p.hx,p.hy,4]];
for(var j=0;j<joints.length;j++){
ctx.beginPath();ctx.arc(joints[j][0],joints[j][1],joints[j][2],0,Math.PI*2);
ctx.fillStyle='#374151';ctx.fill();
ctx.lineWidth=1.5;ctx.strokeStyle='#9ca3af';ctx.stroke();}
// auto label
if(autoOn){
ctx.fillStyle='rgba(34,197,94,0.85)';ctx.font='bold 13px sans-serif';ctx.textAlign='left';
ctx.fillText('● AI 自动作业中',12,22);}}

function update(){
    var df=0.055;sA+=(tSA-sA)*df;eA+=(tEA-eA)*df;
    bx+=(tbX-bx)*0.06;
    var p=armPos(sA,eA);trail.push({x:p.hx,y:p.hy});
    if(trail.length>100)trail.shift();
    if(Math.abs(bx-tbX)>0.5)trackOffset=(trackOffset+0.3)%8;
    if(autoOn){
        if(autoPause>0){autoPause--;}
        else{
            var converged=Math.abs(sA-tSA)<0.03&&Math.abs(eA-tEA)<0.03&&Math.abs(bx-tbX)<2;
            if(converged){
                autoIdx=(autoIdx+1)%seq.length;
                var nx=seq[autoIdx];tSA=nx.s;tEA=nx.e;tbX=nx.bx;
                if(nx.w>0)autoPause=nx.w;}}}

function loop(){update();draw();requestAnimationFrame(loop);}

canvas.addEventListener('click',function(ev){
if(autoOn)return;
var r=canvas.getBoundingClientRect();
var x=(ev.clientX-r.left)*(W/r.width),y=(ev.clientY-r.top)*(H/r.height);
var res=ik(x,y);tSA=res.s;tEA=res.e;});

window.excToggleAuto=function(){
autoOn=!autoOn;
var btn=document.getElementById('exc-auto-btn');
var msg=document.getElementById('exc-msg');
if(autoOn){btn.textContent='⏸ 暂停';btn.className='btn btn-primary';
msg.textContent='🤖 AI 正在执行自动挖掘作业流程';
autoIdx=0;autoT=0;autoPause=0;tSA=seq[0].s;tEA=seq[0].e;tbX=seq[0].bx;trail=[];}
else{btn.textContent='▶ AI自动作业';btn.className='btn btn-green';
msg.textContent='💡 点击画布任意位置移动机械臂';}};

window.excReset=function(){
autoOn=false;autoIdx=0;autoT=0;trail=[];
sA=0.7;eA=1.0;tSA=0.7;tEA=1.0;bx=155;tbX=155;trackOffset=0;
document.getElementById('exc-auto-btn').textContent='▶ AI自动作业';
document.getElementById('exc-auto-btn').className='btn btn-green';
document.getElementById('exc-msg').textContent='💡 点击画布任意位置移动机械臂';};

loop();
})();

/* ========== Neural Network Particle Animation ========== */
(function(){
var c=document.getElementById('nn-canvas'),ctx=c.getContext('2d');
var W,H,mouseX=-999,mouseY=-999,nodes=[],N=55,CONN=110,MR=140;

function resize(){
var rect=c.parentElement.getBoundingClientRect();
W=rect.width*0.55;H=rect.height;
c.width=W;c.height=H;
nodes=[];
for(var i=0;i<N;i++){
nodes.push({
x:Math.random()*W,y:Math.random()*H,
vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,
ox:0,oy:0,
r:Math.random()*2+1.5
});}}
resize();

function draw(){
ctx.clearRect(0,0,W,H);
for(var i=0;i<N;i++){
var n=nodes[i];
n.x+=n.vx;n.y+=n.vy;
n.ox=0;n.oy=0;
// mouse attraction
var dx=mouseX-n.x,dy=mouseY-n.y;
var d=Math.sqrt(dx*dx+dy*dy);
if(d<MR&&d>1){
var f=(1-d/MR)*0.15;
n.ox=dx*f;n.oy=dy*f;}
// wrap around
if(n.x<-20)n.x=W+20;if(n.x>W+20)n.x=-20;
if(n.y<-20)n.y=H+20;if(n.y>H+20)n.y=-20;}
// draw connections
for(var i=0;i<N;i++){
var a=nodes[i],ax=a.x+a.ox,ay=a.y+a.oy;
for(var j=i+1;j<N;j++){
var b=nodes[j],bx2=b.x+b.ox,by2=b.y+b.oy;
var ddx=ax-bx2,ddy=ay-by2;
var dist=Math.sqrt(ddx*ddx+ddy*ddy);
if(dist<CONN){
var al=(1-dist/CONN)*0.45;
ctx.strokeStyle='rgba(100,180,255,'+al+')';
ctx.lineWidth=0.7;
ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx2,by2);ctx.stroke();}}}
// draw nodes
for(var i=0;i<N;i++){
var n=nodes[i],nx=n.x+n.ox,ny=n.y+n.oy;
var pulse=0.85+0.15*Math.sin(Date.now()*0.002+i);
ctx.beginPath();ctx.arc(nx,ny,n.r*pulse,0,Math.PI*2);
ctx.fillStyle='rgba(160,210,255,0.85)';ctx.fill();
// glow
ctx.beginPath();ctx.arc(nx,ny,n.r*2.5,0,Math.PI*2);
ctx.fillStyle='rgba(100,180,255,0.08)';ctx.fill();}
requestAnimationFrame(draw);}

c.parentElement.addEventListener('mousemove',function(e){
var r=c.getBoundingClientRect();
mouseX=e.clientX-r.left;mouseY=e.clientY-r.top;});
c.parentElement.addEventListener('mouseleave',function(){mouseX=-999;mouseY=-999;});
window.addEventListener('resize',function(){resize();});
draw();
})();
</script>
