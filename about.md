--- 
layout: page
permalink: /about/
title: 关于
---

## 联系方式
邮箱: wuyuechen8@gmail.com

---

## 🚀 钱学森弹道动画演示

钱学森弹道（又称为助推-滑翔弹道）是我国科学家钱学森于20世纪40年代提出的一种新型弹道理论。该弹道利用飞行器在大气层边缘的升力，实现"打水漂"式的滑翔飞行，使射程大幅超越传统弹道导弹。

<div style="text-align:center;margin:20px 0;">
    <canvas id="qxs-canvas" width="600" height="300" style="border-radius:12px;background:#0a0e1a;"></canvas>
    <div style="margin-top:12px;">
        <button id="qxs-play-btn" onclick="qxsToggle()" class="btn btn-green" style="padding:8px 24px;font-size:14px;">▶ 播放演示</button>
        <button onclick="qxsReset()" class="btn" style="padding:8px 24px;font-size:14px;">🔄 重置</button>
    </div>
    <p id="qxs-status" style="color:#9ca3af;font-size:13px;margin-top:10px;">点击播放按钮观看弹道对比</p>
</div>

### 弹道对比分析

| 特征 | 传统弹道 | 钱学森弹道 |
|------|---------|-----------|
| **飞行轨迹** | 抛物线 | 波浪形滑翔 |
| **大气层交互** | 穿透式再入 | 跳跃式滑翔 |
| **射程提升** | 受限 | 提升2-3倍 |
| **关键原理** | 仅依赖重力 | 利用升力跳跃 |
| **飞行高度** | 高抛 | 临近空间反复起伏 |

### 核心原理

1. **助推段**：飞行器被火箭助推至大气层边缘（约100公里高度）
2. **跳跃段**：利用气动升力从大气层"弹起"，进入外层空间
3. **滑翔段**：在大气层边缘滑行，再次进入大气层时被弹起
4. **再入段**：最终以较小角度再入大气层，精确打击目标

这种弹道使飞行器具备更远的射程和更强的突防能力，是现代高超音速武器的核心技术之一。

<style>
.btn{background:#374151;color:#fff;border:none;border-radius:6px;cursor:pointer;transition:all 0.2s;}
.btn:hover{opacity:0.85;}
.btn-green{background:#10b981;color:#fff;}
.btn-primary{background:#3b82f6;color:#fff;}
</style>

<script>
(function(){
var canvas=document.getElementById('qxs-canvas'),ctx=canvas.getContext('2d');
var W=canvas.width,H=canvas.height;
var earthR=H*0.65,earthY=H+earthR-30;
var atmosphereH=45;
var animOn=false,animT=0;
var qxsTrail=[],convTrail=[];

function draw(){
    ctx.clearRect(0,0,W,H);
    
    // 星空背景
    ctx.fillStyle='#0a0e1a';ctx.fillRect(0,0,W,H);
    for(var i=0;i<80;i++){
        var x=(i*137.5)%W,y=(i*73.3)%(H-80);
        var al=0.3+Math.random()*0.5;
        ctx.fillStyle='rgba(255,255,255,'+al+')';
        ctx.beginPath();ctx.arc(x,y,0.8,0,Math.PI*2);ctx.fill();
    }
    
    // 大气层光晕
    var glow=ctx.createLinearGradient(0,H-80-atmosphereH,0,H-80);
    glow.addColorStop(0,'rgba(59,130,246,0)');
    glow.addColorStop(0.5,'rgba(59,130,246,0.2)');
    glow.addColorStop(1,'rgba(59,130,246,0.4)');
    ctx.fillStyle=glow;
    ctx.beginPath();
    ctx.ellipse(W/2,H-80,W*0.65,atmosphereH,0,0,Math.PI*2);
    ctx.fill();
    
    // 大气层边界线
    ctx.strokeStyle='rgba(100,150,255,0.3)';ctx.lineWidth=1;
    ctx.beginPath();
    ctx.ellipse(W/2,H-80,W*0.65,atmosphereH,0,0,Math.PI*2);
    ctx.stroke();
    
    // 大气层标签
    ctx.font='11px sans-serif';ctx.textAlign='left';
    ctx.fillStyle='rgba(100,150,255,0.6)';
    ctx.fillText('大气层',W*0.72,H-80-atmosphereH+5);
    
    // 地球
    var earthGrad=ctx.createRadialGradient(W/2,H-30,0,W/2,H-30,earthR);
    earthGrad.addColorStop(0,'#1e40af');
    earthGrad.addColorStop(0.6,'#1d4ed8');
    earthGrad.addColorStop(1,'#1e3a8a');
    ctx.fillStyle=earthGrad;
    ctx.beginPath();
    ctx.arc(W/2,earthY,earthR,Math.PI,0);
    ctx.fill();
    
    // 地球阴影
    ctx.fillStyle='rgba(0,0,0,0.2)';
    ctx.beginPath();
    ctx.arc(W/2+30,earthY+10,earthR*0.95,Math.PI,0);
    ctx.fill();
    
    // 地球网格线
    ctx.strokeStyle='rgba(255,255,255,0.08)';ctx.lineWidth=1;
    for(var lat=-2;lat<=2;lat++){
        var y=H-30-lat*20;
        ctx.beginPath();
        ctx.ellipse(W/2,y,W*0.62*(1-Math.abs(lat)*0.15),25*(1-Math.abs(lat)*0.2),0,0,Math.PI*2);
        ctx.stroke();
    }
    
    // 发射点
    ctx.fillStyle='#f59e0b';
    ctx.beginPath();ctx.arc(80,H-32,6,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#fff';
    ctx.font='bold 12px sans-serif';ctx.textAlign='center';
    ctx.fillText('发射',80,H-38);
    
    // 传统弹道轨迹
    ctx.strokeStyle='rgba(239,68,68,0.6)';ctx.lineWidth=2.5;
    ctx.beginPath();
    for(var i=0;i<convTrail.length;i++){
        var p=convTrail[i];
        if(i===0)ctx.moveTo(p.x,p.y);
        else ctx.lineTo(p.x,p.y);
    }
    ctx.stroke();
    
    // 传统弹道箭头
    if(convTrail.length>5){
        var last=convTrail[convTrail.length-1];
        var prev=convTrail[convTrail.length-5];
        var ang=Math.atan2(last.y-prev.y,last.x-prev.x);
        ctx.beginPath();
        ctx.moveTo(last.x,last.y);
        ctx.lineTo(last.x-12*Math.cos(ang-0.3),last.y-12*Math.sin(ang-0.3));
        ctx.moveTo(last.x,last.y);
        ctx.lineTo(last.x-12*Math.cos(ang+0.3),last.y-12*Math.sin(ang+0.3));
        ctx.stroke();
    }
    
    // 传统弹道标签
    ctx.fillStyle='rgba(239,68,68,0.7)';
    ctx.fillText('传统弹道',300,100);
    ctx.font='10px sans-serif';
    ctx.fillStyle='rgba(239,68,68,0.5)';
    ctx.fillText('抛物线轨迹，射程受限',300,118);
    
    // 钱学森弹道轨迹
    ctx.strokeStyle='rgba(34,197,94,0.7)';ctx.lineWidth=2.5;
    ctx.beginPath();
    for(var i=0;i<qxsTrail.length;i++){
        var p=qxsTrail[i];
        if(i===0)ctx.moveTo(p.x,p.y);
        else ctx.lineTo(p.x,p.y);
    }
    ctx.stroke();
    
    // 钱学森弹道箭头
    if(qxsTrail.length>5){
        var last=qxsTrail[qxsTrail.length-1];
        var prev=qxsTrail[qxsTrail.length-5];
        var ang=Math.atan2(last.y-prev.y,last.x-prev.x);
        ctx.beginPath();
        ctx.moveTo(last.x,last.y);
        ctx.lineTo(last.x-12*Math.cos(ang-0.3),last.y-12*Math.sin(ang-0.3));
        ctx.moveTo(last.x,last.y);
        ctx.lineTo(last.x-12*Math.cos(ang+0.3),last.y-12*Math.sin(ang+0.3));
        ctx.stroke();
    }
    
    // 钱学森弹道标签
    ctx.fillStyle='rgba(34,197,94,0.8)';
    ctx.font='bold 13px sans-serif';
    ctx.fillText('钱学森弹道',420,130);
    ctx.font='10px sans-serif';
    ctx.fillStyle='rgba(34,197,94,0.5)';
    ctx.fillText('"打水漂"滑翔，射程提升2-3倍',420,148);
    
    // 轨迹点发光
    if(convTrail.length>0){
        var lp=convTrail[convTrail.length-1];
        ctx.fillStyle='rgba(239,68,68,0.8)';
        ctx.beginPath();ctx.arc(lp.x,lp.y,5,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='rgba(239,68,68,0.3)';
        ctx.beginPath();ctx.arc(lp.x,lp.y,10,0,Math.PI*2);ctx.fill();
    }
    if(qxsTrail.length>0){
        var lp=qxsTrail[qxsTrail.length-1];
        ctx.fillStyle='rgba(34,197,94,0.8)';
        ctx.beginPath();ctx.arc(lp.x,lp.y,5,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='rgba(34,197,94,0.3)';
        ctx.beginPath();ctx.arc(lp.x,lp.y,10,0,Math.PI*2);ctx.fill();
    }
    
    // 跳跃点标记
    for(var i=0;i<qxsTrail.length-1;i++){
        var p1=qxsTrail[i],p2=qxsTrail[i+1];
        if(p2.y<p1.y && p1.y>H-80-atmosphereH && p2.y<H-80-atmosphereH){
            ctx.fillStyle='rgba(251,191,36,0.6)';
            ctx.beginPath();ctx.arc(p2.x,p2.y,4,0,Math.PI*2);ctx.fill();
            ctx.fillStyle='rgba(251,191,36,0.8)';
            ctx.font='8px sans-serif';
            ctx.fillText('跳跃',p2.x-10,p2.y-8);
        }
    }
}

function update(){
    if(!animOn)return;
    
    animT+=0.008;
    
    // 传统弹道：抛物线
    var convMaxT=1.8;
    if(animT<convMaxT){
        var t=animT/convMaxT;
        var x=80+t*400;
        var y=H-32-120*Math.sin(t*Math.PI)+t*20;
        convTrail.push({x:x,y:y});
    }
    
    // 钱学森弹道：波浪形滑翔
    var qxsMaxT=3.5;
    if(animT<qxsMaxT){
        var t=animT/qxsMaxT;
        var baseY=H-80-atmosphereH/2;
        var wave=Math.sin(t*Math.PI*4)*atmosphereH*0.7;
        var x=80+t*520;
        var y=baseY-wave;
        qxsTrail.push({x:x,y:y});
    }
    
    if(animT>=qxsMaxT){
        animOn=false;
        document.getElementById('qxs-play-btn').textContent='▶ 播放演示';
        document.getElementById('qxs-play-btn').className='btn btn-green';
        document.getElementById('qxs-status').textContent='演示完成！钱学森弹道射程显著更远';
    }
}

function loop(){update();draw();requestAnimationFrame(loop);}

window.qxsToggle=function(){
    animOn=!animOn;
    var btn=document.getElementById('qxs-play-btn');
    var status=document.getElementById('qxs-status');
    if(animOn){
        btn.textContent='⏸ 暂停';
        btn.className='btn btn-primary';
        status.textContent='正在演示两种弹道的飞行轨迹...';
    }else{
        btn.textContent='▶ 播放演示';
        btn.className='btn btn-green';
        status.textContent='演示已暂停';
    }
};

window.qxsReset=function(){
    animOn=false;
    animT=0;
    qxsTrail=[];
    convTrail=[];
    document.getElementById('qxs-play-btn').textContent='▶ 播放演示';
    document.getElementById('qxs-play-btn').className='btn btn-green';
    document.getElementById('qxs-status').textContent='点击播放按钮观看弹道对比';
};

loop();
})();
</script>