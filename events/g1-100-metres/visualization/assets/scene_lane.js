/* ===== G1 100 metres replay (z-up, ±0.61 m corridor; DQ freeze + hazard decal) ===== */
const COL=[0x6E97C4,0xE0A43B,0xB6F24E,0xF2704E];
const LINKS=DATA.links, MESHES=DATA.hq, POL=DATA.policies, FPS=DATA.fps, NL=LINKS.length;
const PARENT=DATA.parents||{}, REST=DATA.rest||{};
const PARENT_I=LINKS.map(n=>LINKS.indexOf(PARENT[n]));
const LANE_HALF= (DATA.meta&&DATA.meta.lane_half_width_m)||0.61;
const SCORED_TIMEOUT_S=60.0;
const TORSO_COLLAPSE_Z=0.35; // display proxy when penetration traces are absent
const LANE_PITCH=2.0; // visual separation between seed replicas
const LANE=Array.from({length:POL.length},(_,i)=>(i-(POL.length-1)/2)*LANE_PITCH);
const TORSO_I=LINKS.indexOf('torso_link');
const PELVIS_I=Math.max(0, LINKS.indexOf('pelvis'));
// frame row: [t, per link: px,py,pz,qx,qy,qz,qw]
const LO=l=>1+l*7;

function finishCrossingT(p){
  // Freeze on the crossing itself, interpolated between frames. Rounding to the
  // nearest frame is 1.04 m at 13 m/s, which is why runners were stopping short
  // of the line.
  let t=p.finish;
  const o=LO(TORSO_I>=0?TORSO_I:0);
  for(let i=1;i<p.frames.length;i++){
    const a=p.frames[i-1][o]-p.startX, b=p.frames[i][o]-p.startX;
    if(a<100 && b>=100){ t=((i-1)+(100-a)/(b-a))/FPS; break; }
  }
  return t;
}

/** Website-only DQ instant from capture metadata or trajectory gate heuristics. */
function detectDq(p){
  if(p.valid!==false) return null;
  if(p.dq_time!=null && Number.isFinite(p.dq_time))
    return {t:p.dq_time, reason:p.dq_reason||'disqualified'};
  const oPel=LO(PELVIS_I), oTor=LO(TORSO_I>=0?TORSO_I:PELVIS_I);
  let tLane=null, tCollapse=null, tCross=null;
  const lastT=p.frames.length?p.frames[p.frames.length-1][0]:0;
  for(let i=0;i<p.frames.length;i++){
    const fr=p.frames[i], t=fr[0];
    const lat=Math.abs(fr[oPel+1]-p.startY);
    const dist=fr[oTor]-p.startX;
    const z=fr[oTor+2];
    if(tLane==null && lat>LANE_HALF){
      if(i>0){
        const prev=p.frames[i-1], lat0=Math.abs(prev[oPel+1]-p.startY);
        if(lat0<=LANE_HALF && lat>LANE_HALF && lat!==lat0)
          tLane=prev[0]+(LANE_HALF-lat0)/(lat-lat0)*(t-prev[0]);
        else tLane=t;
      }else tLane=t;
    }
    if(tCollapse==null && z<TORSO_COLLAPSE_Z) tCollapse=t;
    if(tCross==null && dist>=100) tCross=t;
  }
  const cands=[];
  if(tLane!=null) cands.push({t:tLane, reason:'in_lane'});
  if(tCollapse!=null) cands.push({t:tCollapse, reason:'self_collision'});
  if(tCross==null && (p.finish==null || !Number.isFinite(p.finish)))
    cands.push({t:Math.min(lastT, SCORED_TIMEOUT_S), reason:'finished'});
  if(!cands.length){
    const t=tCross!=null?tCross:(Number.isFinite(p.finish)?p.finish:lastT);
    return {t, reason:p.dq_reason||'disqualified'};
  }
  cands.sort((a,b)=>a.t-b.t);
  return cands[0];
}

POL.forEach(p=>{
  p.startY=p.frames[0][2];
  p.startX=p.frames[0][1];
  const crossT=finishCrossingT(p);
  const dq=detectDq(p);
  p.dq=dq;
  p.disqualified=!!dq;
  // Invalid seeds freeze at the first gating failure; valid seeds at the tape.
  p.freezeT=dq?dq.t:crossT;
});
const T_END=Math.max(...POL.map(p=>p.freezeT))+0.5;

// ---- three setup, z-up ----
const stage=document.getElementById('stage');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.outputEncoding=THREE.sRGBEncoding;
renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.06;
stage.appendChild(renderer.domElement);
const scene=new THREE.Scene(); const SKY=0x0e1520;
scene.background=new THREE.Color(SKY); scene.fog=new THREE.Fog(SKY,22,70);
const camera=new THREE.PerspectiveCamera(40,16/9,0.1,400); camera.up.set(0,0,1);
scene.add(new THREE.HemisphereLight(0xbcd2ff,0x20242a,0.7));
const sun=new THREE.DirectionalLight(0xfff3df,1.4); sun.position.set(4,6,12); sun.castShadow=true;
sun.shadow.mapSize.set(2048,2048); sun.shadow.bias=-0.0004;
const sc=sun.shadow.camera; sc.near=1;sc.far=60;sc.left=-10;sc.right=10;sc.top=10;sc.bottom=-10; sc.up&&sc.up.set(0,0,1);
scene.add(sun,sun.target);
const rim=new THREE.DirectionalLight(0x9ab4ff,0.45); rim.position.set(-6,-8,5); scene.add(rim);
function studioEnv(){
  const pm=new THREE.PMREMGenerator(renderer); pm.compileEquirectangularShader();
  const s2=new THREE.Scene();
  const cv=document.createElement('canvas'); cv.width=16; cv.height=256; const g=cv.getContext('2d');
  const gr=g.createLinearGradient(0,0,0,256); gr.addColorStop(0,'#cfd8e8'); gr.addColorStop(.45,'#7d8a9d'); gr.addColorStop(1,'#191d24');
  g.fillStyle=gr; g.fillRect(0,0,16,256);
  s2.add(new THREE.Mesh(new THREE.SphereGeometry(50,24,16), new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(cv),side:THREE.BackSide})));
  const soft=(x,y,z,c,i)=>{const m=new THREE.Mesh(new THREE.PlaneGeometry(9,9),new THREE.MeshBasicMaterial({color:new THREE.Color(c).multiplyScalar(i)}));m.position.set(x,y,z);m.lookAt(0,0,0);s2.add(m);};
  soft(12,10,14,0xffffff,3.2); soft(-14,6,9,0x9fb4ff,1.6); soft(2,-14,7,0xffe6c4,1.3);
  const env=pm.fromScene(s2,0.05).texture; pm.dispose(); return env;
}
scene.environment=studioEnv();

// ---- track (x = length, y = lanes, z=0 ground) ----
function trackTex(){
  const c=document.createElement('canvas'); c.width=4096;c.height=512; const g=c.getContext('2d');
  g.fillStyle='#7c3a26'; g.fillRect(0,0,c.width,c.height);
  for(let i=0;i<9000;i++){g.fillStyle=`rgba(0,0,0,${Math.random()*0.05})`;g.fillRect(Math.random()*c.width,Math.random()*c.height,2,2);}
  const X=m=>((m+6)/121)*c.width; const Yw=4.2, Y=y=>((y+Yw)/(2*Yw))*c.height;
  // Official lane corridor: ±LANE_HALF about each seed's visual centre.
  g.strokeStyle='rgba(255,255,255,.9)';g.lineWidth=5;
  LANE.forEach(l=>{[l-LANE_HALF,l+LANE_HALF].forEach(y=>{
    g.beginPath();g.moveTo(X(-6),Y(y));g.lineTo(X(115),Y(y));g.stroke();});
    // soft fill inside the legal corridor
    const y0=Y(l-LANE_HALF), y1=Y(l+LANE_HALF);
    g.fillStyle='rgba(182,242,78,0.07)';
    g.fillRect(X(-6),Math.min(y0,y1),X(115)-X(-6),Math.abs(y1-y0));
  });
  g.strokeStyle='rgba(255,255,255,.35)';g.lineWidth=2;g.setLineDash([10,10]);
  LANE.forEach(l=>{g.beginPath();g.moveTo(X(-6),Y(l));g.lineTo(X(115),Y(l));g.stroke();});
  g.setLineDash([]);
  g.fillStyle='rgba(255,255,255,.8)';g.font='bold 30px sans-serif';g.textAlign='center';
  for(let m=0;m<=100;m+=10){g.globalAlpha=.75;g.fillRect(X(m)-2,0,4,c.height);g.globalAlpha=1;LANE.forEach(l=>g.fillText(m+'',X(m),Y(l)+11));}
  // start and finish: solid white lines across the track, ~0.15 m wide, as on a
  // real track. The previous checkerboard had 0.5 m squares over 1.6 m of track
  // and read as a slab rather than a line.
  const lw=Math.max(4,Math.round(0.15/(121/c.width)));
  g.fillStyle='#fff'; g.fillRect(X(0)-lw/2,0,lw,c.height);
  g.fillRect(X(100)-lw/2,0,lw,c.height);
  g.fillStyle='rgba(0,0,0,0.30)';
  g.fillRect(X(100)+lw/2,0,2,c.height); g.fillRect(X(100)-lw/2-2,0,2,c.height);
  const t=new THREE.CanvasTexture(c);t.anisotropy=8;t.encoding=THREE.sRGBEncoding;return t;
}
const track=new THREE.Mesh(new THREE.PlaneGeometry(121,8.4),new THREE.MeshStandardMaterial({map:trackTex(),roughness:.95}));
track.position.set(54.5,0,0); track.receiveShadow=true; scene.add(track);
const finishPlane=new THREE.Mesh(new THREE.PlaneGeometry(8.4,2.0),
  new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0.07,side:THREE.DoubleSide,depthWrite:false}));
finishPlane.position.set(100,0,1.0); finishPlane.rotation.y=Math.PI/2; finishPlane.rotation.x=Math.PI/2; scene.add(finishPlane);
const infield=new THREE.Mesh(new THREE.PlaneGeometry(500,500),new THREE.MeshStandardMaterial({color:0x162129,roughness:1}));
infield.position.set(50,0,-0.02); infield.receiveShadow=true; scene.add(infield);

// ---- geometry per link (shared) ----
function b64buf(s){const bin=atob(s);const u=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer;}
const GEO={};
LINKS.forEach(n=>{
  const m=MESHES[n];
  // positions are uint16 quantized inside the link's own bbox
  const q=new Uint16Array(b64buf(m.p));
  const pos=new Float32Array(q.length);
  const lo=m.lo, sp=m.span;
  for(let i=0,v=0;i<q.length;i+=3,v++){
    pos[i]  =lo[0]+(q[i]  /65535)*sp[0];
    pos[i+1]=lo[1]+(q[i+1]/65535)*sp[1];
    pos[i+2]=lo[2]+(q[i+2]/65535)*sp[2];
  }
  const idx=m.i32?new Uint32Array(b64buf(m.f)):new Uint16Array(b64buf(m.f));
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(pos,3));
  g.setIndex(new THREE.BufferAttribute(idx,1));
  // Without normals, flatShading + DoubleSide turns the large sole/side walls of
  // the G1 foot into bright unlit planes that read as a white cross at race distance.
  g.computeVertexNormals();
  GEO[n]=g;
});
// Joint/actuator housings render dark like the real robot; big shells take the
// lane colour. Foot soles (ankle_roll) use a dark rubber material matching the
// Unitree G1; prior HQ extracts for those links were non-manifold / duplicate-faced
// and rendered as intersecting planes.
const isDark=n=>/(hip|shoulder)_(roll|yaw)_link|elbow_roll_link|^logo/.test(n);
const isFoot=n=>/ankle_roll_link$/.test(n);
const jointMat=new THREE.MeshPhysicalMaterial({color:0x2a2d33,metalness:0.72,roughness:0.44,
  clearcoat:0.35,clearcoatRoughness:0.35,envMapIntensity:1.0,side:THREE.DoubleSide,flatShading:false});
jointMat.shadowSide=THREE.FrontSide;
const footMat=new THREE.MeshPhysicalMaterial({color:0x2a2d33,metalness:0.2,roughness:0.68,
  clearcoat:0.25,clearcoatRoughness:0.45,envMapIntensity:0.85,side:THREE.DoubleSide,flatShading:false});
footMat.shadowSide=THREE.FrontSide;
function shellMat(ci){const c=new THREE.Color(COL[ci]).lerp(new THREE.Color(0xffffff),0.16);
  const m=new THREE.MeshPhysicalMaterial({color:c,metalness:0.32,roughness:0.36,
    clearcoat:0.6,clearcoatRoughness:0.22,envMapIntensity:1.05,reflectivity:0.5,side:THREE.DoubleSide,flatShading:false});
  m.shadowSide=THREE.FrontSide; return m;}
function plaque(txt,hex){
  const c=document.createElement('canvas'); c.width=512;c.height=256; const g=c.getContext('2d');
  g.fillStyle='rgba(8,12,18,0.82)'; g.beginPath();
  if(g.roundRect) g.roundRect(8,44,496,168,22); else g.rect(8,44,496,168);
  g.fill();
  g.strokeStyle=hex; g.lineWidth=7; g.stroke();
  g.fillStyle=hex; g.font='bold 118px -apple-system,Segoe UI,sans-serif'; g.textAlign='center'; g.textBaseline='middle';
  g.fillText(txt,256,132);
  const t=new THREE.CanvasTexture(c); t.anisotropy=8; t.encoding=THREE.sRGBEncoding; return t;
}
function hazardTex(reason){
  // Offscreen stripe field, then stamp DISQUALIFIED so the stripes show through the glyphs.
  const c=document.createElement('canvas'); c.width=1024; c.height=512; const g=c.getContext('2d');
  const stripes=document.createElement('canvas'); stripes.width=c.width; stripes.height=c.height;
  const sg=stripes.getContext('2d');
  sg.fillStyle='#111'; sg.fillRect(0,0,stripes.width,stripes.height);
  const sw=56; sg.fillStyle='#f0c400';
  for(let x=-stripes.height;x<stripes.width+stripes.height;x+=sw*2){
    sg.beginPath();
    sg.moveTo(x,0); sg.lineTo(x+sw,0);
    sg.lineTo(x+sw+stripes.height,stripes.height); sg.lineTo(x+stripes.height,stripes.height);
    sg.closePath(); sg.fill();
  }
  g.drawImage(stripes,0,0);
  g.fillStyle='rgba(10,10,10,0.82)';
  g.fillRect(28, c.height*0.18, c.width-56, c.height*0.64);
  // Punch letters, then refill them with the same hazard stripes.
  g.globalCompositeOperation='destination-out';
  g.font='bold 96px -apple-system,Segoe UI,sans-serif';
  g.textAlign='center'; g.textBaseline='middle';
  g.fillText('DISQUALIFIED', c.width/2, c.height*0.42);
  g.globalCompositeOperation='destination-over';
  g.drawImage(stripes,0,0);
  g.globalCompositeOperation='source-over';
  if(reason){
    g.fillStyle='#f0c400';
    g.strokeStyle='#111'; g.lineWidth=6;
    g.font='bold 34px -apple-system,Segoe UI,sans-serif';
    const label=String(reason).replace(/_/g,' ');
    g.strokeText(label, c.width/2, c.height*0.68);
    g.fillText(label, c.width/2, c.height*0.68);
  }
  const t=new THREE.CanvasTexture(c); t.anisotropy=8; t.encoding=THREE.sRGBEncoding; return t;
}
const PLAQUES=POL.map((p,ci)=>{
  const hex='#'+new THREE.Color(COL[ci]).getHexString();
  const label=p.disqualified?('DQ '+(p.freezeT!=null?p.freezeT.toFixed(2)+'s':'')): (p.finish.toFixed(2)+'s');
  const m=new THREE.Mesh(new THREE.PlaneGeometry(1.45,0.72),
    new THREE.MeshBasicMaterial({map:plaque(label.trim(),hex),transparent:true,depthWrite:false}));
  m.position.set(0,LANE[ci],0.012); m.visible=false; scene.add(m); return m;
});
const DQ_DECALS=POL.map((p,ci)=>{
  if(!p.disqualified) return null;
  const m=new THREE.Mesh(new THREE.PlaneGeometry(3.6,1.8),
    new THREE.MeshBasicMaterial({map:hazardTex(p.dq&&p.dq.reason),transparent:true,depthWrite:false}));
  m.position.set(0,LANE[ci],0.014); m.visible=false; scene.add(m); return m;
});
const ROBOTS=POL.map((p,ci)=>{
  const grp=new THREE.Group(); const shell=shellMat(ci); const nodes=[],meshes=[];
  LINKS.forEach(n=>{
    const node=new THREE.Group(); node.name=n;
    const mat=isFoot(n)?footMat:(isDark(n)?jointMat:shell);
    const mesh=new THREE.Mesh(GEO[n], mat);
    mesh.castShadow=true;node.add(mesh);nodes.push(node);meshes.push(mesh);});
  LINKS.forEach((n,l)=>{
    const pi=PARENT_I[l];
    (pi>=0?nodes[pi]:grp).add(nodes[l]);
  });
  scene.add(grp); return {grp,nodes,meshes,material:shell};
});

// ---- interpolation ----
const qA=new THREE.Quaternion(),qB=new THREE.Quaternion(),qC=new THREE.Quaternion();
function localQuat(row,l,pi,out){
  const o=LO(l),po=LO(pi);
  out.set(row[po+3],row[po+4],row[po+5],row[po+6]).normalize().conjugate();
  qC.set(row[o+3],row[o+4],row[o+5],row[o+6]).normalize();
  return out.multiply(qC).normalize();
}
function poseRobot(rb,p,t,laneY){
  const dt=1/FPS;
  const tt=Math.min(t,p.freezeT);            // hold exactly on the crossing
  let i=Math.max(0,Math.floor(tt/dt));
  const last=p.frames.length-1;
  i=Math.min(i,last-1>0?last-1:0);
  const j=Math.min(last,i+1), a=Math.min(1,Math.max(0,(tt/dt)-i));
  const A=p.frames[i],B=p.frames[j];
  const sx=p.startX, sy=p.startY;             // align start; keep true lateral
  for(let l=0;l<NL;l++){const o=LO(l);
    const node=rb.nodes[l],pi=PARENT_I[l];
    if(pi<0){
      const px=A[o]+(B[o]-A[o])*a, py=A[o+1]+(B[o+1]-A[o+1])*a, pz=A[o+2]+(B[o+2]-A[o+2])*a;
      qA.set(A[o+3],A[o+4],A[o+5],A[o+6]).normalize();
      qB.set(B[o+3],B[o+4],B[o+5],B[o+6]).normalize();
      qA.slerp(qB,a);
      // laneY separates seeds for viewing; (py-sy) is the real in-lane lateral
      node.position.set(px-sx, laneY+(py-sy), pz);
    }else{
      const r=REST[LINKS[l]];
      if(!r) throw new Error('missing rest transform for '+LINKS[l]);
      node.position.set(r[0],r[1],r[2]);
      localQuat(A,l,pi,qA); localQuat(B,l,pi,qB); qA.slerp(qB,a);
    }
    node.quaternion.copy(qA);
  }
  return (A[1]+(B[1]-A[1])*a)-sx;             // distance run from the start line
}

// ---- HUD ----
const clockEl=document.getElementById('clock');
const cards=POL.map((p,i)=>document.getElementById('lane'+i));
function hud(t,xs,dones){clockEl.textContent=t.toFixed(2);
  POL.forEach((p,i)=>{const el=cards[i]; if(!el)return; const d=Math.min(xs[i],100);
    const lat=(p.max_lateral_m!=null)?(' · body≤'+p.max_lateral_m.toFixed(2)+'m'):'';
    let status;
    if(dones[i] && p.disqualified){
      const reason=(p.dq&&p.dq.reason)?p.dq.reason.replace(/_/g,' '):'DQ';
      status='DQ '+p.freezeT.toFixed(2)+'s · '+reason+lat;
    }else if(dones[i]){
      status='FINISH '+p.finish.toFixed(2)+'s'+lat;
    }else status=d.toFixed(1)+' m';
    el.querySelector('.d').textContent=status;
    el.classList.toggle('fin',dones[i] && !p.disqualified);
    el.classList.toggle('dq',dones[i] && p.disqualified);});}

function resize(){const w=stage.clientWidth,h=Math.round(w*9/16);renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
window.addEventListener('resize',resize);
let raf=null,startWall=null,speed=1;
// playback clock that survives pausing
let playT=0, lastNow=null, playing=false;
// user camera: spherical offset around the tracked runner, dragged/zoomed
const VIEW={az:-1.30, el:0.62, dist:13.6};   // default = elevated broadcast 3/4
let validationPolicy=null,validationTargetZ=null,validationTargetXOffset=-0.6;
const camP=new THREE.Vector3(),camT=new THREE.Vector3();
(function(){
  const el=renderer.domElement; el.style.touchAction='none'; el.style.cursor='grab';
  let drag=false,px=0,py=0;
  el.addEventListener('pointerdown',e=>{drag=true;px=e.clientX;py=e.clientY;el.setPointerCapture(e.pointerId);el.style.cursor='grabbing';});
  el.addEventListener('pointermove',e=>{if(!drag)return;
    VIEW.az-=(e.clientX-px)*0.005; VIEW.el=Math.min(1.45,Math.max(0.06,VIEW.el+(e.clientY-py)*0.004));
    px=e.clientX;py=e.clientY; if(!playing)draw(playT);});
  const end=e=>{drag=false;el.style.cursor='grab';};
  el.addEventListener('pointerup',end); el.addEventListener('pointercancel',end);
  el.addEventListener('wheel',e=>{e.preventDefault();
    VIEW.dist=Math.min(48,Math.max(3.2,VIEW.dist*(1+Math.sign(e.deltaY)*0.09)));
    if(!playing)draw(playT);},{passive:false});
})();
function draw(t){
  const xs=[],dones=[];
  POL.forEach((p,i)=>{const done=t>=p.freezeT;
    // Hold the pose at finish (valid) or at the first gating failure (DQ).
    const x=poseRobot(ROBOTS[i],p,t,LANE[i]); xs[i]=x;dones[i]=done;
    const pl=PLAQUES[i];
    const dec=DQ_DECALS[i];
    if(p.disqualified){
      pl.visible=false;
      if(dec){
        dec.visible=done;
        if(done){
          dec.position.x=x;
          dec.position.y=LANE[i];
          dec.position.z=0.014;
        }
      }
    }else{
      if(dec) dec.visible=false;
      pl.visible=done;
      if(done) pl.position.x=101.6;
    }
  });
  const packX=validationPolicy===null?Math.max(...xs):xs[validationPolicy];
  const packY=validationPolicy===null?0:LANE[validationPolicy];
  camT.set(packX+validationTargetXOffset, packY, validationTargetZ===null?0.75:validationTargetZ);
  const ce=Math.cos(VIEW.el), se=Math.sin(VIEW.el);
  camP.set(camT.x+VIEW.dist*ce*Math.cos(VIEW.az),
           camT.y+VIEW.dist*ce*Math.sin(VIEW.az),
           camT.z+VIEW.dist*se);
  camera.position.lerp(camP, startWall===null?1:0.14);
  camera.lookAt(camT);
  sun.target.position.set(packX,0,0); sun.position.set(packX+2,-6,15);
  hud(t,xs,dones); renderer.render(scene,camera);
}
const btn=document.getElementById('replay');
function setBtn(){btn.innerHTML = playing ? '&#10073;&#10073; Pause' : (playT>=T_END ? '&#9654; Replay' : '&#9654; Play');}
function loop(now){
  if(lastNow===null)lastNow=now;
  playT += (now-lastNow)/1000*speed; lastNow=now; startWall=1;
  if(playT>=T_END){playT=T_END; draw(playT); playing=false; raf=null; setBtn(); return;}
  draw(playT); raf=requestAnimationFrame(loop);}
function play(){ if(playT>=T_END) playT=0; playing=true; lastNow=null;
  if(raf)cancelAnimationFrame(raf); raf=requestAnimationFrame(loop); setBtn(); }
function pause(){ playing=false; if(raf)cancelAnimationFrame(raf); raf=null; lastNow=null; setBtn(); }
btn.addEventListener('click',()=>{ playing?pause():play(); });
window.addEventListener('keydown',e=>{ if(e.code==='Space'){e.preventDefault(); playing?pause():play();} });
document.querySelectorAll('.seg button').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.seg button').forEach(x=>x.setAttribute('aria-pressed','false'));
  b.setAttribute('aria-pressed','true');speed=parseFloat(b.dataset.s);}));
window.__G1_REPLAY__={
  seek(t){
    pause(); playT=Math.min(T_END,Math.max(0,Number(t)||0));
    startWall=null; draw(playT); startWall=1; setBtn();
    return playT;
  },
  validationView({policy=null,az=VIEW.az,el=VIEW.el,dist=VIEW.dist,targetZ=0.75,targetXOffset=-0.6}={}){
    validationPolicy=Number.isInteger(policy)?Math.min(POL.length-1,Math.max(0,policy)):null;
    validationTargetZ=Number.isFinite(targetZ)?targetZ:null;
    validationTargetXOffset=Number.isFinite(targetXOffset)?targetXOffset:-0.6;
    VIEW.az=az;VIEW.el=el;VIEW.dist=dist;
    ROBOTS.forEach((r,i)=>{r.grp.visible=validationPolicy===null||i===validationPolicy;});
    startWall=null;draw(playT);startWall=1;
    return {policy:validationPolicy,az:VIEW.az,el:VIEW.el,dist:VIEW.dist,targetZ:validationTargetZ,targetXOffset:validationTargetXOffset};
  },
  inspect(policy=0){
    scene.updateMatrixWorld(true);
    const rb=ROBOTS[Math.min(POL.length-1,Math.max(0,policy))],out={};
    LINKS.forEach((n,i)=>{
      if(/(knee|ankle_(pitch|roll))_link$/.test(n)){
        out[n]={position:rb.nodes[i].getWorldPosition(new THREE.Vector3()).toArray(),
                visible:rb.meshes[i].visible};
      }
    });
    return {time:playT,fps:FPS,meta:DATA.meta,links:out};
  }
};
resize(); draw(0);
setBtn(); if(!matchMedia('(prefers-reduced-motion: reduce)').matches) setTimeout(play,500);
