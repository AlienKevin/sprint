/* ===== G1 100 metres replay (z-up, ±0.61 m corridor; reason-specific stop cues) ===== */
const COL=DATA.colors||[0x6E97C4,0xE0A43B,0xB6F24E,0xF2704E];
const LINKS=DATA.links, MESHES=DATA.hq, POL=DATA.policies, FPS=DATA.fps, NL=LINKS.length;
const PARENT=DATA.parents||{}, REST=DATA.rest||{};
const PARENT_I=LINKS.map(n=>LINKS.indexOf(PARENT[n]));
const LANE_HALF= (DATA.meta&&DATA.meta.lane_half_width_m)||0.61;
const PASSIVE_SETTLE_S=1.25;
const IS_COMPARISON=Boolean(DATA.meta&&DATA.meta.comparison);
const IS_TRAJECTORY_COMPARISON=Boolean(DATA.meta&&DATA.meta.trajectory_comparison);
// The compact Scoring illustrations reserve chrome outside the 3D viewport.
// Main race and trajectory players keep their existing 16:9 composition.
const IS_SCORING_EXAMPLE=!IS_COMPARISON&&new URLSearchParams(location.search).get('example')==='1';
const IS_SIDE_EXAMPLE=IS_SCORING_EXAMPLE&&new URLSearchParams(location.search).get('view')==='side';
const REPLAY_AUTOPLAY=new URLSearchParams(location.search).get('autoplay')!=='0';
document.documentElement.classList.toggle('replay-example',IS_SCORING_EXAMPLE);
document.documentElement.classList.toggle('trajectory-comparison',IS_TRAJECTORY_COMPARISON);
document.documentElement.classList.toggle('replay-embedded',window.parent!==window);
// Every player has a controls row in real flow on phones; display:contents
// preserves the existing desktop placement without adding another box.
{
  const controls=document.createElement('div');controls.className='replay-controls';
  const playback=document.querySelector('.ctl'),cameraControls=document.querySelector('.camera-ctl');
  playback.before(controls);controls.append(playback);if(cameraControls)controls.append(cameraControls);
}
// Standalone verifier replays retain the full eight-lane venue. The homepage
// comparison is deliberately a compact three-lane heat so its three runners
// occupy the frame instead of appearing on one edge of an empty track.
const REQUESTED_TRACK_LANES=Number(DATA.meta&&DATA.meta.track_lanes);
const TRACK_LANES=Number.isInteger(REQUESTED_TRACK_LANES)&&REQUESTED_TRACK_LANES>0
  ?REQUESTED_TRACK_LANES:8;
const TRACK_LANE_WIDTH=1.22;
const TRACK_HALF_WIDTH=TRACK_LANES*TRACK_LANE_WIDTH/2;
const MIDDLE_LANE_INDEX=Math.floor((TRACK_LANES-1)/2);
const EXPLICIT_LANES=Array.isArray(DATA.lane_indices)?DATA.lane_indices:null;
const EXPLICIT_LANE_LABELS=Array.isArray(DATA.lane_labels)?DATA.lane_labels:null;
const LANE=Array.from({length:POL.length},(_,i)=>{
  const requested=EXPLICIT_LANES&&Number.isInteger(EXPLICIT_LANES[i])?EXPLICIT_LANES[i]:MIDDLE_LANE_INDEX+i-Math.floor(POL.length/2);
  const laneIndex=Math.max(0,Math.min(TRACK_LANES-1,requested));
  return (laneIndex-(TRACK_LANES-1)/2)*TRACK_LANE_WIDTH;
});
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

/** Terminal event copied from verifier metadata. Never infer it from a pose. */
function verifierTerminal(p){
  if(p.valid!==false) return null;
  const lastT=p.frames.length?p.frames[p.frames.length-1][0]:0;
  if(Number.isFinite(p.terminal_time) && p.terminal_reason)
    return {t:p.terminal_time, reason:p.terminal_reason};
  // Neutral compatibility fallback for historical payloads. The renderer is
  // intentionally not allowed to inspect posture or lane position here.
  return {t:lastT, reason:'did_not_finish'};
}

function terminalStatus(policy,done){
  if(!policy||!done)return {kind:'',text:''};
  if(!policy.failed)return {kind:'finished',text:'FINISHED'};
  if(policy.timedOut)return {kind:'incomplete',text:'TIMEOUT'};
  return {kind:'incomplete',text:compactPolicyFailure(policy)};
}

// Explicit compact labels keep the comparison HUD readable at narrow widths.
// The clock badge and result plaques use the same outcome-only wording.
function compactPolicyFailure(policy){
  const key=String(policy.terminal?.reason||'').toLowerCase().replace(/[^a-z0-9]+/g,'_');
  return {in_lane:'LANE DRIFT',self_collision:'COLLISION',body_height:'FELL',fell:'FELL',fall:'FELL'}[key]||'DNF';
}

function policyPlaqueLabel(policy){
  if(!policy.failed)return policy.finish.toFixed(2)+'s';
  return policy.timedOut?'TIMEOUT':compactPolicyFailure(policy);
}

// Scan the complete recording, including later recoveries, before shortening a
// replay. A meaningful forward bout advances the pelvis high-water mark by at
// least 2cm at >=2cm/s; vertical/limb motion and tiny root jitter do not count.
// End exactly one second after its final advancing frame, not after a trailing
// velocity window. Official terminal poses/results remain unchanged.
const STALL_MIN_SPEED_MPS=0.02,STALL_MIN_PROGRESS_M=0.02,STALL_BOUT_GAP_S=0.1,STALL_SETTLE_S=1;
function policyPlaybackEnd(p){
  if(!p.failed||p.frames.length<2)return p.freezeT;
  const frames=p.frames,o=LO(PELVIS_I),firstT=frames[0][0];
  let furthest=frames[0][o],lastMovingT=firstT,boutProgress=0,boutStart=firstT,lastAdvanceT=null;
  for(let i=1;i<frames.length;i++){
    const frame=frames[i],dt=frame[0]-frames[i-1][0];
    const next=Math.max(furthest,frame[o]),advance=next-furthest;
    furthest=next;
    if(dt<=0)continue;
    if(advance<=1e-9)continue;
    // Millimetre-rounded 50Hz poses can repeat during a real slow crawl. Join
    // these tiny sample gaps, but measure speed over the whole moving bout.
    // This grace never extends the end timestamp: only advancing frames do.
    const newBout=lastAdvanceT===null||frame[0]-lastAdvanceT>Math.max(STALL_BOUT_GAP_S,dt)+1e-9;
    const advanceDt=newBout?dt:frame[0]-lastAdvanceT;
    if(newBout){
      boutProgress=0;boutStart=frames[i-1][0];
    }
    boutProgress+=advance;lastAdvanceT=frame[0];
    // Also require current forward speed: a fast early launch must not lend
    // its old average speed to later sub-threshold creeping. Include repeated
    // quantized samples in the elapsed time between actual advances.
    if(advance/advanceDt>=STALL_MIN_SPEED_MPS-1e-9&&
       boutProgress>=STALL_MIN_PROGRESS_M-1e-9&&
       boutProgress/(frame[0]-boutStart)>=STALL_MIN_SPEED_MPS-1e-9)lastMovingT=frame[0];
  }
  return Math.min(p.freezeT,lastMovingT+STALL_SETTLE_S);
}

POL.forEach(p=>{
  p.startY=p.frames[0][2];
  p.startX=p.frames[0][1];
  const crossT=finishCrossingT(p);
  const terminal=verifierTerminal(p);
  p.terminal=terminal;
  p.failed=!!terminal;
  p.timedOut=!!terminal&&(p.timed_out===true||['timeout','time_limit'].includes(terminal.reason));
  p.disqualified=!!terminal&&(p.disqualified===true||['in_lane','self_collision'].includes(terminal.reason));
  p.eventT=terminal?terminal.t:crossT;
  const captureEnd=p.frames.length?p.frames[p.frames.length-1][0]:p.eventT;
  const recordedTail=p.disqualified&&captureEnd>p.eventT+1/FPS;
  p.poseEndT=p.timedOut?captureEnd:(recordedTail?Math.min(captureEnd,p.eventT+PASSIVE_SETTLE_S):p.eventT);
  p.syntheticSettle=!p.timedOut&&p.disqualified&&!recordedTail;
  // The official result appears at eventT. A short, non-scoring visual tail
  // lets a disqualified runner settle instead of hanging at the gate sample.
  // Timeout runners instead use every authoritative frame still present after
  // scoring stops; eventT remains untouched and continues to drive the HUD.
  p.freezeT=p.timedOut?captureEnd:(p.syntheticSettle?p.eventT+PASSIVE_SETTLE_S:p.poseEndT);
  p.playbackEndT=policyPlaybackEnd(p);
});
// POL is the complete selected set, not merely the camera's focused runner.
// Selection/removal reloads this scene, so no removed runner keeps it playing.
const T_CAPTURE_END=Math.max(...POL.map(p=>p.freezeT));
const T_END=Math.max(...POL.map(p=>p.playbackEndT));

// ---- three setup, z-up ----
const stage=document.getElementById('stage');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.outputEncoding=THREE.sRGBEncoding;
renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.06;
stage.appendChild(renderer.domElement);
const scene=new THREE.Scene(); const SKY=0x0e1520;
scene.background=new THREE.Color(SKY);
// A comparison camera may pull back far enough to contain runners spread over
// the full 100 m.  Keep the standard replay's atmospheric falloff, but extend
// it for the race view so the track does not disappear behind fog late on.
scene.fog=new THREE.Fog(SKY,IS_COMPARISON?65:22,IS_COMPARISON?220:70);
const camera=new THREE.PerspectiveCamera(IS_COMPARISON?45:40,16/9,0.1,400); camera.up.set(0,0,1);
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
  const c=document.createElement('canvas'); c.width=4096;c.height=1024; const g=c.getContext('2d');
  const TRACK_MIN=-8,TRACK_MAX=108,TRACK_LENGTH=TRACK_MAX-TRACK_MIN;
  const X=m=>((m-TRACK_MIN)/TRACK_LENGTH)*c.width;
  const Y=y=>((y+TRACK_HALF_WIDTH)/(TRACK_HALF_WIDTH*2))*c.height;
  g.fillStyle='#b64936'; g.fillRect(0,0,c.width,c.height);
  // Fine rubber granule texture, kept subtle so the markings stay crisp.
  for(let i=0;i<16000;i++){
    const alpha=.015+Math.random()*.025;
    g.fillStyle=`rgba(${Math.random()>.5?'255,214,185':'78,24,18'},${alpha})`;
    const size=1+Math.random()*2;
    g.fillRect(Math.random()*c.width,Math.random()*c.height,size,size);
  }
  // Regulation-width lanes with continuous 50 mm white separators.
  g.strokeStyle='rgba(255,255,255,.96)';
  g.lineWidth=Math.max(4,Math.round(.05/(TRACK_LANE_WIDTH*TRACK_LANES)*c.height));
  for(let lane=0;lane<=TRACK_LANES;lane++){
    const y=-TRACK_HALF_WIDTH+lane*TRACK_LANE_WIDTH;
    g.beginPath();g.moveTo(X(TRACK_MIN),Y(y));g.lineTo(X(TRACK_MAX),Y(y));g.stroke();
  }
  // Preserve the established eight-lane replay texture. The much narrower
  // comparison track uses physical decals below so its numerals do not inherit
  // this canvas's highly anisotropic world-space scaling.
  if(!IS_COMPARISON&&!IS_SCORING_EXAMPLE){
    g.fillStyle='rgba(255,255,255,.96)';
    g.font='800 74px Arial, sans-serif';
    g.textAlign='center';g.textBaseline='middle';
    for(let lane=0;lane<TRACK_LANES;lane++){
      const center=-TRACK_HALF_WIDTH+(lane+.5)*TRACK_LANE_WIDTH;
      g.fillText(String(lane+1),X(-2.2),Y(center));
    }
  }
  // Common start and finish lines for the 100 m straight.
  const lineWidth=Math.max(5,Math.round(.05/TRACK_LENGTH*c.width));
  g.fillStyle='#fff';
  g.fillRect(X(0)-lineWidth/2,0,lineWidth,c.height);
  g.fillRect(X(100)-lineWidth/2,0,lineWidth,c.height);
  // Small 10 m edge ticks aid distance reading without striping across lanes.
  g.fillStyle='rgba(255,255,255,.82)';
  for(let m=10;m<100;m+=10){
    const tick=Math.max(2,Math.round(.035/TRACK_LENGTH*c.width));
    const inset=Math.round(c.height*.018);
    g.fillRect(X(m)-tick/2,0,tick,inset);
    g.fillRect(X(m)-tick/2,c.height-inset,tick,inset);
  }
  const t=new THREE.CanvasTexture(c);t.anisotropy=8;t.encoding=THREE.sRGBEncoding;return t;
}
const track=new THREE.Mesh(new THREE.PlaneGeometry(116,TRACK_LANES*TRACK_LANE_WIDTH),new THREE.MeshStandardMaterial({map:trackTex(),roughness:.93}));
track.position.set(50,0,0); track.receiveShadow=true; scene.add(track);
function laneNumberTexture(label,color='rgba(255,255,255,.97)'){
  const c=document.createElement('canvas');c.width=256;c.height=512;const g=c.getContext('2d');
  g.clearRect(0,0,c.width,c.height);g.fillStyle=color;
  g.font='900 390px Arial, sans-serif';g.textAlign='center';g.textBaseline='middle';
  g.fillText(String(label),c.width/2,c.height/2+8);
  const texture=new THREE.CanvasTexture(c);texture.anisotropy=8;texture.encoding=THREE.sRGBEncoding;return texture;
}
if(IS_COMPARISON){
  const geometry=new THREE.PlaneGeometry(.62,1.18);
  for(let lane=0;lane<TRACK_LANES;lane++){
    const label=EXPLICIT_LANE_LABELS?EXPLICIT_LANE_LABELS[lane]:lane+1;
    if(label===null||label===undefined||label==='')continue;
    const material=new THREE.MeshBasicMaterial({map:laneNumberTexture(label),transparent:true,alphaTest:.08,depthWrite:false,side:THREE.DoubleSide});
    const numeral=new THREE.Mesh(geometry,material);
    // Match the original texture orientation exactly: the canvas horizontal
    // axis follows track X and its tall axis crosses the lane on track Y. The
    // independent .62 m by 1.18 m plane fixes the old texture compression
    // without turning or mirroring the familiar start-line labels.
    numeral.position.set(-3.2,-TRACK_HALF_WIDTH+(lane+.5)*TRACK_LANE_WIDTH,.008);
    numeral.renderOrder=2;scene.add(numeral);
  }
}
const finishPlane=new THREE.Mesh(new THREE.PlaneGeometry(TRACK_LANES*TRACK_LANE_WIDTH,2.0),
  new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0.07,side:THREE.DoubleSide,depthWrite:false}));
finishPlane.position.set(100,0,1.0); finishPlane.rotation.y=Math.PI/2; finishPlane.rotation.x=Math.PI/2; scene.add(finishPlane);
const infield=new THREE.Mesh(new THREE.PlaneGeometry(500,500),new THREE.MeshStandardMaterial({color:0x162129,roughness:1}));
infield.position.set(50,0,-0.02); infield.receiveShadow=true; scene.add(infield);

// ---- geometry per link (shared) ----
function b64buf(s){const bin=atob(s);const u=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer;}
// Measured front envelope of the G1 torso through the upper-chest logo area.
// Each row stores z, centre-front x, and the centre-to-edge falloff. Interpolate
// it in both axes so identity artwork follows the torso rather than floating on
// a one-dimensional cylindrical bow.
const CHEST_PROFILE=[
  [.168,.083,.037],[.183,.086,.035],[.197,.085,.032],[.212,.084,.027],
  [.226,.081,.026],[.241,.079,.024],[.255,.076,.021],[.270,.072,.021],[.284,.067,.020]
];
function chestProfileAt(z){
  if(z<=CHEST_PROFILE[0][0])return CHEST_PROFILE[0];
  if(z>=CHEST_PROFILE[CHEST_PROFILE.length-1][0])return CHEST_PROFILE[CHEST_PROFILE.length-1];
  for(let i=1;i<CHEST_PROFILE.length;i++)if(z<=CHEST_PROFILE[i][0]){
    const a=CHEST_PROFILE[i-1],b=CHEST_PROFILE[i],t=(z-a[0])/(b[0]-a[0]);
    return [z,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];
  }
}
function torsoChestSurfaceX(y,z){
  const p=chestProfileAt(z),lateral=Math.min(1,Math.abs(y)/.114);
  return p[1]-p[2]*lateral*lateral;
}
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
  g.computeBoundingBox();
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
function shellMat(ci){const c=new THREE.Color(COL[ci]);
  const emphasized=IS_TRAJECTORY_COMPARISON&&POL[ci]?.emphasized;
  if(!emphasized)c.lerp(new THREE.Color(0xffffff),0.16);
  const m=new THREE.MeshPhysicalMaterial({color:c,emissive:emphasized?c:0x000000,emissiveIntensity:emphasized?0.12:0,
    metalness:emphasized?0.04:0.32,roughness:emphasized?0.62:0.36,
    clearcoat:emphasized?0.08:0.6,clearcoatRoughness:0.22,envMapIntensity:emphasized?0.25:1.05,reflectivity:0.5,side:THREE.DoubleSide,flatShading:false});
  m.shadowSide=THREE.FrontSide; return m;}
function plaque(txt,hex){
  const c=document.createElement('canvas'); c.width=512;c.height=256; const g=c.getContext('2d');
  g.fillStyle='rgba(8,12,18,0.82)'; g.beginPath();
  if(g.roundRect) g.roundRect(8,44,496,168,22); else g.rect(8,44,496,168);
  g.fill();
  g.strokeStyle=hex; g.lineWidth=7; g.stroke();
  g.fillStyle=hex;let fontSize=118;g.font=`bold ${fontSize}px -apple-system,Segoe UI,sans-serif`;
  while(g.measureText(txt).width>456&&fontSize>48){fontSize-=2;g.font=`bold ${fontSize}px -apple-system,Segoe UI,sans-serif`;}
  g.textAlign='center'; g.textBaseline='middle';
  g.fillText(txt,256,132);
  const t=new THREE.CanvasTexture(c); t.anisotropy=8; t.encoding=THREE.sRGBEncoding;t.userData={label:txt,font:g.font};return t;
}
const PLAQUES=POL.map((p,ci)=>{
  const hex='#'+new THREE.Color(COL[ci]).getHexString();
  const label=policyPlaqueLabel(p);
  const m=new THREE.Mesh(new THREE.PlaneGeometry(1.45,0.72),
    new THREE.MeshBasicMaterial({map:plaque(label.trim(),hex),transparent:true,depthWrite:false}));
  m.userData.resultLabel=label;
  m.position.set(p.failed?failureFrame(p)[1]-p.startX+1.6:101.6,LANE[ci],0.012);m.visible=false;scene.add(m);return m;
});

// Two curved identity elements follow the torso link instead of facing the
// camera: a standalone company mark above, and a model-name-only race bib below.
// The Unitree G1 torso is about 25.6 cm wide and bows forward by roughly 3.3 cm
// from centre to edge, so both meshes are curved across Y.
function drawIdentityLogo(g,identity,texture){
  if(!identity.logo)return;
  const image=new Image();
  image.onload=()=>{
    const maxW=480,maxH=224,scale=Math.min(maxW/image.naturalWidth,maxH/image.naturalHeight);
    const width=image.naturalWidth*scale,height=image.naturalHeight*scale;
    g.drawImage(image,256-width/2,124-height/2,width,height);
    texture.needsUpdate=true;
  };
  image.src=identity.logo;
}
function logoTexture(identity){
  const c=document.createElement('canvas');c.width=512;c.height=248;const g=c.getContext('2d');
  const t=new THREE.CanvasTexture(c);t.anisotropy=16;t.encoding=THREE.sRGBEncoding;
  drawIdentityLogo(g,identity,t);return t;
}
function bibTexture(identity,hex){
  const c=document.createElement('canvas');c.width=1024;c.height=420;const g=c.getContext('2d');
  g.fillStyle='#f7f5ef';g.fillRect(0,0,c.width,c.height);
  const text=identity.model||identity.company||'Runner';
  let lines=[text],fontSize=126;
  g.font=`800 ${fontSize}px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif`;
  if(g.measureText(text).width>880){
    const words=text.split(/\s+/);let best=null;
    for(let i=1;i<words.length;i++){
      const candidate=[words.slice(0,i).join(' '),words.slice(i).join(' ')];
      const width=Math.max(...candidate.map(line=>g.measureText(line).width));
      if(!best||width<best.width)best={lines:candidate,width};
    }
    if(best)lines=best.lines;
  }
  do{g.font=`800 ${fontSize}px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif`;fontSize-=4}
  while(Math.max(...lines.map(line=>g.measureText(line).width))>880&&fontSize>52);
  g.fillStyle='#000';g.textAlign='center';g.textBaseline='middle';
  const lineHeight=Math.round((fontSize+4)*.88),startY=210-(lines.length-1)*lineHeight/2;
  lines.forEach((line,index)=>g.fillText(line,526,startY+index*lineHeight));
  g.strokeStyle='rgba(17,24,32,.18)';g.lineWidth=7;g.strokeRect(3.5,3.5,c.width-7,c.height-7);
  const t=new THREE.CanvasTexture(c);t.anisotropy=16;t.encoding=THREE.sRGBEncoding;
  return t;
}
function policyNumberTexture(policy,identity){
  const c=document.createElement('canvas');c.width=768;c.height=512;const g=c.getContext('2d');
  g.fillStyle='#fff';g.fillRect(0,0,c.width,c.height);
  const number=String(policy.policy_number??policy.lane_number??'');
  let size=number.length>1?300:370;
  g.font=`900 ${size}px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif`;
  while(g.measureText(number).width>650&&size>180){size-=10;g.font=`900 ${size}px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif`;}
  g.fillStyle='#000';g.textAlign='center';g.textBaseline='middle';g.fillText(number,c.width/2,205);
  const model=identity?.model||identity?.company||'';
  if(model){
    let modelSize=52;g.font=`800 ${modelSize}px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif`;
    while(g.measureText(model).width>660&&modelSize>28){modelSize-=2;g.font=`800 ${modelSize}px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif`;}
    g.fillText(model,c.width/2,449);
  }
  const t=new THREE.CanvasTexture(c);t.anisotropy=16;t.encoding=THREE.sRGBEncoding;return t;
}
function curvedChestPlateGeometry(width,height,centerZ,frontX,curve,columns){
  const pos=[],uv=[],idx=[];
  for(let row=0;row<2;row++)for(let col=0;col<=columns;col++){
    const u=col/columns,v=row,y=(u-.5)*width,z=centerZ+(v-.5)*height;
    const normalized=y/(width*.5);pos.push(frontX-curve*normalized*normalized,y,z);uv.push(u,v);
  }
  for(let col=0;col<columns;col++){const a=col,b=col+1,c=columns+1+col,d=c+1;idx.push(a,b,c,b,d,c)}
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));geometry.setIndex(idx);geometry.computeVertexNormals();return geometry;
}
function contouredChestGeometry(width,height,centerZ,offset,columns,rows){
  const pos=[],uv=[],idx=[],stride=columns+1;
  for(let row=0;row<=rows;row++)for(let col=0;col<=columns;col++){
    const u=col/columns,v=row/rows,y=(u-.5)*width,z=centerZ+(v-.5)*height;
    pos.push(torsoChestSurfaceX(y,z)+offset,y,z);uv.push(u,v);
  }
  for(let row=0;row<rows;row++)for(let col=0;col<columns;col++){
    const a=row*stride+col,b=a+1,c=a+stride,d=c+1;idx.push(a,b,c,b,d,c);
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));geometry.setIndex(idx);geometry.computeVertexNormals();return geometry;
}
// The standalone mark now spans nearly the full 25.6 cm torso width and is
// centred over the molded Unitree chest branding. Lower and shorten the model
// bib so both elements remain distinct, with enough crown to clear the raised
// shell details rather than clipping through them in motion.
const CHEST_BIB_GEO=curvedChestPlateGeometry(.228,.076,.117,.103,.040,10);
// A shell-material cover sits just beyond the molded Unitree relief. It is not
// a banner: it follows the measured torso surface and visually merges with the
// robot body. The transparent logo floats 1 mm above it. OpenAI's square mark
// gets a 20% larger transparent carrier so its artwork grows without stretching.
const CHEST_COVER_GEO=contouredChestGeometry(.234,.122,.226,.011,12,6);
const CHEST_LOGO_GEO=contouredChestGeometry(.228,.116,.226,.012,12,6);
const CHEST_OPENAI_LOGO_GEO=contouredChestGeometry(.274,.139,.226,.012,14,7);
const CHEST_POLICY_BIB_GEO=contouredChestGeometry(.242,.158,.205,.013,14,8);
function chestBib(identity,hex){
  if(!identity)return null;
  const mesh=new THREE.Mesh(CHEST_BIB_GEO,new THREE.MeshBasicMaterial({map:bibTexture(identity,hex),side:THREE.DoubleSide}));
  mesh.name='runner_chest_bib';mesh.renderOrder=3;return mesh;
}
function chestCover(material){
  const mesh=new THREE.Mesh(CHEST_COVER_GEO,material);
  mesh.name='runner_chest_cover';mesh.renderOrder=2;return mesh;
}
function chestLogo(identity){
  if(!identity?.logo)return null;
  const geometry=identity.brand==='openai'?CHEST_OPENAI_LOGO_GEO:CHEST_LOGO_GEO;
  const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({
    map:logoTexture(identity),side:THREE.DoubleSide,transparent:true,alphaTest:.02,depthWrite:false
  }));
  mesh.name='runner_chest_logo';mesh.renderOrder=4;return mesh;
}
function chestPolicyNumber(policy,identity){
  const material=new THREE.MeshBasicMaterial({
    color:0xffffff,map:policyNumberTexture(policy,identity),side:THREE.DoubleSide
  });
  // Policy bibs are identity labels, not reflective robot bodywork. Keep white
  // pixels truly white regardless of exposure, environment map, or tone mapping.
  material.toneMapped=false;
  const mesh=new THREE.Mesh(CHEST_POLICY_BIB_GEO,material);
  mesh.name='runner_chest_policy_number';mesh.renderOrder=4;return mesh;
}
const ROBOTS=POL.map((p,ci)=>{
  const grp=new THREE.Group(); const shell=shellMat(ci); const nodes=[],meshes=[];
  grp.userData.policyIndex=ci;
  LINKS.forEach(n=>{
    const node=new THREE.Group(); node.name=n;
    const mat=isFoot(n)?footMat:(isDark(n)?jointMat:shell);
    const mesh=new THREE.Mesh(GEO[n], mat);
    mesh.castShadow=true;node.add(mesh);
    if(n==='torso_link'){
      const identity=p.identity||null,hex=p.color||('#'+new THREE.Color(COL[ci]).getHexString());
      const cover=chestCover(shell),logo=IS_TRAJECTORY_COMPARISON?null:chestLogo(identity),
        bib=IS_TRAJECTORY_COMPARISON?chestPolicyNumber(p,identity):chestBib(identity,hex);
      node.add(cover);
      if(logo)node.add(logo);
      if(bib)node.add(bib);
    }
    grp.add(node);nodes.push(node);meshes.push(mesh);});
  scene.add(grp); return {grp,nodes,meshes,material:shell};
});

function failureFrame(p){
  const index=Math.min(p.frames.length-1,Math.max(0,Math.round(p.eventT*FPS)));
  return p.frames[index];
}
function capturedPoint(p,ci,frame,linkIndex){
  const o=LO(linkIndex);
  return new THREE.Vector3(
    frame[o]-p.startX,
    frame[o+1]-p.startY+LANE[ci],
    frame[o+2]
  );
}
function likelySelfCollisionPoint(p,ci){
  const frame=failureFrame(p);
  const candidates=LINKS.map((name,index)=>({name,index}))
    .filter(({name})=>/(knee|ankle|palm|elbow|torso|head)/.test(name));
  let best=null;
  for(let a=0;a<candidates.length;a++) for(let b=a+1;b<candidates.length;b++){
    const left=candidates[a],right=candidates[b];
    if(PARENT[left.name]===right.name||PARENT[right.name]===left.name) continue;
    const leftSide=left.name.startsWith('left_')?'left':left.name.startsWith('right_')?'right':'core';
    const rightSide=right.name.startsWith('left_')?'left':right.name.startsWith('right_')?'right':'core';
    if(leftSide!=='core'&&leftSide===rightSide) continue;
    const pa=capturedPoint(p,ci,frame,left.index),pb=capturedPoint(p,ci,frame,right.index);
    const distance=pa.distanceToSquared(pb);
    if(!best||distance<best.distance) best={distance,point:pa.add(pb).multiplyScalar(.5)};
  }
  return best?.point||capturedPoint(p,ci,frame,TORSO_I>=0?TORSO_I:PELVIS_I);
}
function laneFailureMarker(p,ci){
  const frame=failureFrame(p),pelvis=LO(PELVIS_I);
  const side=(frame[pelvis+1]-p.startY)>=0?1:-1;
  const group=new THREE.Group();
  const glowMaterial=new THREE.MeshBasicMaterial({color:0xffd166,transparent:true,opacity:.3,depthWrite:false,side:THREE.DoubleSide,blending:THREE.AdditiveBlending});
  const lineMaterial=new THREE.MeshBasicMaterial({color:0xffc928,transparent:true,opacity:.98,depthWrite:false});
  // Make the crossed lane edge an upright yellow plane. It runs along the
  // lane boundary (track X) and rises vertically (track Z); the previous cue
  // was a flat transverse patch on the track and therefore read as rotated the
  // wrong way from the broadcast camera.
  const wash=new THREE.Mesh(new THREE.PlaneGeometry(4.8,.62),glowMaterial);
  wash.rotation.x=Math.PI/2;wash.position.set(0,0,.31);wash.renderOrder=7;group.add(wash);
  const line=new THREE.Mesh(new THREE.BoxGeometry(4.8,.045,.065),lineMaterial);
  line.position.set(0,0,.033);line.renderOrder=8;group.add(line);
  group.position.set(0,LANE[ci]+side*LANE_HALF,0);group.visible=false;scene.add(group);
  return {kind:'lane',group,side};
}
function selfCollisionMarker(p,ci){
  const group=new THREE.Group(),point=likelySelfCollisionPoint(p,ci);
  const core=new THREE.Mesh(new THREE.SphereGeometry(.105,18,12),
    new THREE.MeshBasicMaterial({color:0xff3b30,depthTest:false,depthWrite:false}));
  core.renderOrder=12;group.add(core);
  const aura=new THREE.Mesh(new THREE.SphereGeometry(.25,18,12),
    new THREE.MeshBasicMaterial({color:0xffa62b,transparent:true,opacity:.28,depthTest:false,depthWrite:false,blending:THREE.AdditiveBlending}));
  aura.renderOrder=11;group.add(aura);
  const ring=new THREE.Mesh(new THREE.TorusGeometry(.19,.024,10,28),
    new THREE.MeshBasicMaterial({color:0xffd166,transparent:true,opacity:.95,depthTest:false,depthWrite:false}));
  ring.rotation.x=Math.PI/2;ring.renderOrder=13;group.add(ring);
  const light=new THREE.PointLight(0xff3b30,1.7,2.4);group.add(light);
  group.position.copy(point);group.visible=false;scene.add(group);
  return {kind:'self-collision',group,point};
}
const FAILURE_MARKERS=POL.map((p,ci)=>{
  if(!p.disqualified) return null;
  const reason=p.terminal?.reason;
  if(reason==='in_lane') return laneFailureMarker(p,ci);
  if(reason==='self_collision') return selfCollisionMarker(p,ci);
  return null;
});

// ---- authoritative captured-state playback ----
const qA=new THREE.Quaternion(),qB=new THREE.Quaternion();
function poseRobot(rb,p,t,laneY){
  const tt=Math.min(t,p.poseEndT);
  const last=p.frames.length-1;
  const frame=Math.min(last,Math.max(0,tt*FPS));
  const i=Math.floor(frame),j=Math.min(last,i+1),alpha=j===i?0:frame-i;
  const A=p.frames[i],B=p.frames[j];
  const sx=p.startX, sy=p.startY;             // align start; keep true lateral
  for(let l=0;l<NL;l++){const o=LO(l);
    const node=rb.nodes[l];
    // Captures contain authoritative world-space link transforms. Interpolate
    // only between adjacent samples so slow playback remains fluid without
    // rebuilding the robot through the approximate kinematic hierarchy.
    node.position.set(
      A[o]+(B[o]-A[o])*alpha-sx,
      laneY+(A[o+1]+(B[o+1]-A[o+1])*alpha-sy),
      A[o+2]+(B[o+2]-A[o+2])*alpha
    );
    qA.set(A[o+3],A[o+4],A[o+5],A[o+6]).normalize();
    qB.set(B[o+3],B[o+4],B[o+5],B[o+6]).normalize();
    node.quaternion.copy(qA).slerp(qB,alpha);
  }
  return A[1]+(B[1]-A[1])*alpha-sx;           // distance run from the start line
}

// A replay can sample both sides of the current instant. A centred camera-only
// filter removes gait shake without the speed-dependent lag of a causal spring.
// Robot transforms still use only the authoritative adjacent samples above.
function cameraRootX(p,t){
  const rootAt=at=>{
    const frame=Math.min(p.frames.length-1,Math.max(0,Math.min(at,p.poseEndT)*FPS));
    const i=Math.floor(frame),j=Math.min(p.frames.length-1,i+1),alpha=frame-i;
    return p.frames[i][1]+(p.frames[j][1]-p.frames[i][1])*alpha-p.startX;
  };
  const raw=rootAt(t);
  const filtered=(rootAt(t-.10)+2*rootAt(t-.05)+3*raw+2*rootAt(t+.05)+rootAt(t+.10))/9;
  // Keep starts, sudden pulses and terminal stops close to the visible runner.
  return Math.max(raw-.18,Math.min(raw+.18,filtered));
}

function cameraPostureTargetZ(p,t,uprightTargetZ){
  const rootHeightAt=at=>{
    const frame=Math.min(p.frames.length-1,Math.max(0,Math.min(at,p.poseEndT)*FPS));
    const i=Math.floor(frame),j=Math.min(p.frames.length-1,i+1),alpha=frame-i,o=LO(PELVIS_I)+2;
    return p.frames[i][o]+(p.frames[j][o]-p.frames[i][o])*alpha;
  };
  const height=(rootHeightAt(t-.20)+2*rootHeightAt(t-.10)+3*rootHeightAt(t)+2*rootHeightAt(t+.10)+rootHeightAt(t+.20))/9;
  // Raised hands/feet must not lift the camera while a runner crawls. Use a
  // stable low-posture anchor, with a smooth transition based on trunk height,
  // not the changing extremities of the whole-body bounding box. Upright
  // runners retain their existing framing exactly; this is not model-specific.
  const upright=Math.max(0,Math.min(1,(height-.38)/.18));
  const blend=upright*upright*(3-2*upright);
  return .18+(uprightTargetZ-.18)*blend;
}

// Historical captures stop on the exact DQ sample, so they cannot contain a
// passive physics tail. For those files only, settle the captured rigid pose
// onto the track with a clearly display-only gravity fall. New captures with
// recorded post-terminal states take the authoritative branch above instead.
const settleAxis=new THREE.Vector3(),settleQ=new THREE.Quaternion(),settlePivot=new THREE.Vector3();
const groundCorner=new THREE.Vector3();
function groundRobotToTrack(rb,clearance=.006){
  rb.grp.updateMatrixWorld(true);
  let minZ=Infinity;
  for(const mesh of rb.meshes){
    const box=mesh.geometry.boundingBox;if(!box)continue;
    for(let xi=0;xi<2;xi++)for(let yi=0;yi<2;yi++)for(let zi=0;zi<2;zi++){
      groundCorner.set(
        xi?box.max.x:box.min.x,yi?box.max.y:box.min.y,zi?box.max.z:box.min.z
      ).applyMatrix4(mesh.matrixWorld);
      minZ=Math.min(minZ,groundCorner.z);
    }
  }
  if(Number.isFinite(minZ)){
    rb.grp.position.z+=clearance-minZ;
    rb.grp.updateMatrixWorld(true);
  }
}
function settleRobot(rb,p,t,laneY,x){
  rb.grp.position.set(0,0,0);rb.grp.quaternion.identity();
  if(!p.syntheticSettle||t<=p.eventT) return;
  const u=Math.min(1,Math.max(0,(t-p.eventT)/PASSIVE_SETTLE_S));
  const ease=1-Math.pow(1-u,3);
  const frame=failureFrame(p),tor=LO(TORSO_I>=0?TORSO_I:PELVIS_I),pel=LO(PELVIS_I);
  let dx=frame[tor]-frame[pel],dy=frame[tor+1]-frame[pel+1];
  const mag=Math.hypot(dx,dy);
  if(mag<.08){dx=p.terminal.reason==='in_lane'?0:1;dy=p.terminal.reason==='in_lane'?(frame[pel+1]>=p.startY?1:-1):0;}
  else{dx/=mag;dy/=mag;}
  settleAxis.set(-dy,dx,0).normalize();
  const upright=Math.max(0,Math.min(1,(frame[tor+2]-.42)/.75));
  settleQ.setFromAxisAngle(settleAxis,1.28*upright*ease);
  settlePivot.set(x,laneY,.04);
  rb.grp.quaternion.copy(settleQ);
  rb.grp.position.copy(settlePivot).sub(settlePivot.clone().applyQuaternion(settleQ));
  groundRobotToTrack(rb);
}

// ---- HUD ----
const clockEl=document.getElementById('clock');
const clockStatusEl=document.getElementById('clock-status');
const cards=POL.map((p,i)=>document.getElementById('lane'+i));
function orderModelHudCards(){
  if(!IS_COMPARISON||IS_TRAJECTORY_COMPARISON)return;
  const lanes=document.querySelector('.lanes');if(!lanes)return;
  const rank=p=>{
    const model=String(p.identity?.model||p.label||'').toLowerCase();
    return model.includes('deepseek')?0:model.includes('luna')?1:model.includes('glm')?2:3;
  };
  // Reorder existing DOM nodes only: physical lanes, policy indices, camera
  // targets, highlights, and the cards array keep their original identities.
  POL.map((p,i)=>({i,rank:rank(p)})).sort((a,b)=>a.rank-b.rank||a.i-b.i)
    .forEach(({i})=>{if(cards[i])lanes.append(cards[i]);});
}
orderModelHudCards();
const singlePolicy=POL.length===1;
if(singlePolicy&&!IS_TRAJECTORY_COMPARISON){const lanes=document.querySelector('.lanes');if(lanes)lanes.style.display='none';}
function hud(t,xs,dones,focusedPolicy){
  const primary=POL[0];
  const primaryFinished=singlePolicy && dones[0] && !primary.failed;
  const primaryFailed=singlePolicy && dones[0] && primary.failed;
  const clockTime=primaryFinished&&Number.isFinite(primary.finish)?primary.finish:
    (primaryFailed&&Number.isFinite(primary.eventT)?primary.eventT:t);
  clockEl.textContent=clockTime.toFixed(2);
  if(clockStatusEl){
    const index=singlePolicy?0:focusedPolicy;
    const status=terminalStatus(index==null?null:POL[index],index!=null&&dones[index]);
    clockStatusEl.textContent=status.text;
    clockStatusEl.dataset.kind=status.kind;
  }
  POL.forEach((p,i)=>{const el=cards[i]; if(!el)return; const d=Math.min(xs[i],100);
    if(p.timedOut){
      const timeEl=el.querySelector('.tm');
      if(timeEl)timeEl.textContent=Math.min(Math.max(0,t),p.eventT).toFixed(2)+'s';
    }
    let status;
    if(dones[i] && p.timedOut){
      status='TIMEOUT';
    }else if(dones[i] && p.failed){
      status=compactPolicyFailure(p);
    }else if(dones[i]){
      status='FINISHED';
    }else status=d.toFixed(1)+' m';
    el.querySelector('.d').textContent=status;
    el.classList.toggle('fin',dones[i] && !p.failed);
    el.classList.toggle('dq',dones[i] && p.failed);});}

function replayMobileLayout(){
  let width=window.innerWidth;
  try{if(window.parent!==window)width=window.parent.innerWidth;}catch{}
  return width<=720;
}
function comparisonMobileLayout(){
  if(!IS_COMPARISON)return false;
  return replayMobileLayout();
}
function resize(){document.documentElement.classList.toggle('replay-mobile',replayMobileLayout());const w=Math.max(1,stage.clientWidth),h=Math.max(1,IS_SCORING_EXAMPLE?stage.clientHeight:Math.round(w*9/16));renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
let raf=null,startWall=null,speed=1;
// playback clock that survives pausing
let playT=0, lastNow=null, playing=false;
// user camera: spherical offset around the tracked runner, dragged/zoomed
const FOLLOW_VIEW={az:-0.25,el:0.13,dist:2.541,targetZ:0.63,targetXOffset:0};
// The comparison keeps a wide start-line view available for explicit validation,
// then follows the fastest active runner with FOLLOW_VIEW during playback.
// Standalone and trajectory comparisons use that same close composition.
const DEFAULT_VIEW=IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON
  ?{az:-0.82,el:0.34,dist:8.6}:{...FOLLOW_VIEW,az:IS_SIDE_EXAMPLE?-Math.PI/2:FOLLOW_VIEW.az};
const VIEW={...DEFAULT_VIEW};
let validationPolicy=null,validationTargetZ=null,validationTargetXOffset=-0.6;
function defaultFollowPolicy(){
  if(!IS_TRAJECTORY_COMPARISON)return null;
  const emphasized=POL.findIndex(policy=>policy.emphasized);
  return emphasized>=0?emphasized:Math.max(0,POL.length-1);
}
let DEFAULT_FOLLOW_POLICY=defaultFollowPolicy();
let userFollowPolicy=DEFAULT_FOLLOW_POLICY;
const AUTO_FOLLOW_ORDER=POL.map((p,i)=>i).sort((a,b)=>
  Number(POL[b].effective_speed_mps||0)-Number(POL[a].effective_speed_mps||0)||a-b
);
let currentAutoFollowPolicy=IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON?(AUTO_FOLLOW_ORDER[0]??null):null;
const camP=new THREE.Vector3(),camT=new THREE.Vector3();
const CAMERA_FOLLOW={x:null,y:null,z:null,bodyOffsetX:null,span:null,t:null};
let comparisonAutoCamera=IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON;
// Policy selection changes the destination immediately, but camera travel is
// wall-clock based: paused and 0.1x replays still take half a second to pan.
const CAMERA_SWITCH_MS=500;
let cameraSwitch=null,cameraSwitchRaf=null;
function cancelCameraSwitch(){
  cameraSwitch=null;
  if(cameraSwitchRaf!==null)cancelAnimationFrame(cameraSwitchRaf);
  cameraSwitchRaf=null;
}
function scheduleCameraSwitch(){
  if(!cameraSwitch||cameraSwitchRaf!==null)return;
  cameraSwitchRaf=requestAnimationFrame(()=>{
    cameraSwitchRaf=null;
    // Playing replays already draw in their playback loop. While paused this
    // extra frame updates only the camera, never the simulation clock.
    if(!playing)draw(playT);
    if(cameraSwitch)scheduleCameraSwitch();
  });
}
function startCameraSwitch(){
  cancelCameraSwitch();
  if(!IS_TRAJECTORY_COMPARISON||matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  // Capture the currently displayed pose, including an interrupted pan.
  cameraSwitch={started:performance.now(),position:camera.position.clone(),target:camT.clone()};
  scheduleCameraSwitch();
}
function applyCameraSwitch(now=performance.now()){
  if(!cameraSwitch)return;
  const progress=Math.min(1,Math.max(0,(now-cameraSwitch.started)/CAMERA_SWITCH_MS));
  const alpha=progress*progress*(3-2*progress);
  camP.lerpVectors(cameraSwitch.position,camP,alpha);
  camT.lerpVectors(cameraSwitch.target,camT,alpha);
  if(progress>=1)cancelCameraSwitch();
}
const cameraModeEl=document.getElementById('camera-mode');
const cameraResetBtn=document.getElementById('camera-reset');
function resetFollowDamping(){CAMERA_FOLLOW.x=null;CAMERA_FOLLOW.y=null;CAMERA_FOLLOW.z=null;CAMERA_FOLLOW.bodyOffsetX=null;CAMERA_FOLLOW.span=null;CAMERA_FOLLOW.t=null;}
function beginManualCamera(){
  cancelCameraSwitch();
  if(IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON&&comparisonAutoCamera&&userFollowPolicy===null)
    Object.assign(VIEW,{az:FOLLOW_VIEW.az,el:FOLLOW_VIEW.el,dist:FOLLOW_VIEW.dist});
  comparisonAutoCamera=false;
}
function orbitCamera(deltaAz,deltaEl=0){
  beginManualCamera();VIEW.az+=Number(deltaAz)||0;
  VIEW.el=Math.min(1.45,Math.max(0.06,VIEW.el+(Number(deltaEl)||0)));
  if(!playing)draw(playT);return {az:VIEW.az,el:VIEW.el,dist:VIEW.dist};
}
function policyCameraLabel(index){const p=POL[index];return p?.label||`Lane ${Number(p?.lane_number||index+1)}`;}
function followViewportWidth(){
  if(IS_TRAJECTORY_COMPARISON&&window.parent!==window){
    try{if(window.parent.innerWidth>0)return window.parent.innerWidth;}catch{}
  }
  return stage.clientWidth;
}
function emphasizePolicy(index){
  if(!IS_TRAJECTORY_COMPARISON)return;
  DEFAULT_FOLLOW_POLICY=index;
  POL.forEach((policy,i)=>{
    policy.emphasized=i===index;
    policy.color=policy.emphasized?(policy.model_color||policy.identity?.color||'#6E97C4'):'#FFFFFF';
    COL[i]=Number.parseInt(policy.color.slice(1),16);
    const material=shellMat(i);ROBOTS[i].material.copy(material);material.dispose();
    const card=cards[i];if(!card)return;
    card.classList.toggle('emphasized',policy.emphasized);
    card.style.setProperty('--emphasis',policy.color);
    card.querySelector('.sw').style.background=policy.color;
    card.querySelector('.tm').style.color=policy.color;
  });
  window.dispatchEvent(new CustomEvent('g1:policy-focused',{detail:{captureId:POL[index].capture_id}}));
}
function updateCameraControls(){
  const followedPolicy=userFollowPolicy??(IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON?currentAutoFollowPolicy:null);
  cards.forEach((card,index)=>{
    if(!card)return;
    const followed=index===followedPolicy;
    card.setAttribute('aria-pressed',String(followed));
    if(followed)card.setAttribute('aria-current','true');else card.removeAttribute('aria-current');
    const followControl=card.querySelector('.policy-follow');
    if(followControl){followControl.setAttribute('aria-pressed',String(followed));if(followed)followControl.setAttribute('aria-current','true');else followControl.removeAttribute('aria-current');}
  });
  if(cameraResetBtn)cameraResetBtn.textContent='Reset';
  if(cameraModeEl){cameraModeEl.textContent='';cameraModeEl.hidden=true;}
}
function followPolicy(index){
  const next=Number(index);
  if(!Number.isInteger(next)||next<0||next>=POL.length)return false;
  if(next!==userFollowPolicy)startCameraSwitch();
  userFollowPolicy=next;comparisonAutoCamera=false;
  emphasizePolicy(next);
  if(IS_COMPARISON)Object.assign(VIEW,{az:FOLLOW_VIEW.az,el:FOLLOW_VIEW.el,dist:FOLLOW_VIEW.dist});
  resetFollowDamping();updateCameraControls();
  if(!playing)draw(playT);return true;
}
function resetCamera(){
  cancelCameraSwitch();
  userFollowPolicy=DEFAULT_FOLLOW_POLICY;comparisonAutoCamera=IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON;Object.assign(VIEW,DEFAULT_VIEW);
  resetFollowDamping();updateCameraControls();if(!playing)draw(playT);
  return {follow:userFollowPolicy,auto:comparisonAutoCamera,view:{...VIEW}};
}
function zoomCamera(factor){
  beginManualCamera();VIEW.dist=Math.min(48,Math.max(1.6,VIEW.dist*factor));
  if(!playing)draw(playT);return VIEW.dist;
}
(function(){
  const el=renderer.domElement; el.style.touchAction='none'; el.style.cursor='grab';
  const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();
  let drag=false,moved=false,px=0,py=0,startX=0,startY=0;
  function pickedPolicy(e){
    const rect=el.getBoundingClientRect();
    pointer.set(((e.clientX-rect.left)/rect.width)*2-1,-((e.clientY-rect.top)/rect.height)*2+1);
    raycaster.setFromCamera(pointer,camera);
    const hit=raycaster.intersectObjects(ROBOTS.map(robot=>robot.grp),true)[0];
    let object=hit?.object;
    while(object&&!Number.isInteger(object.userData?.policyIndex))object=object.parent;
    return Number.isInteger(object?.userData?.policyIndex)?object.userData.policyIndex:null;
  }
  el.addEventListener('pointerdown',e=>{drag=true;moved=false;px=startX=e.clientX;py=startY=e.clientY;el.setPointerCapture(e.pointerId);el.style.cursor='grabbing';});
  el.addEventListener('pointermove',e=>{if(!drag)return;
    if(Math.hypot(e.clientX-startX,e.clientY-startY)>5)moved=true;
    if(!moved)return;
    beginManualCamera();
    VIEW.az-=(e.clientX-px)*0.005; VIEW.el=Math.min(1.45,Math.max(0.06,VIEW.el+(e.clientY-py)*0.004));
    px=e.clientX;py=e.clientY; if(!playing)draw(playT);});
  const end=e=>{if(drag&&!moved&&IS_COMPARISON){const policy=pickedPolicy(e);if(policy!==null)followPolicy(policy);}drag=false;el.style.cursor='grab';};
  el.addEventListener('pointerup',end); el.addEventListener('pointercancel',end);
  el.addEventListener('wheel',e=>{e.preventDefault();zoomCamera(Math.exp(e.deltaY*.001));},{passive:false});
})();
function draw(t){
  const xs=[],dones=[],visualDones=[];
  POL.forEach((p,i)=>{const done=t>=p.eventT;
    // Scoring ends at eventT. Recorded or display-only settling may continue
    // afterward, but cannot change the verifier result shown in the HUD.
    const x=poseRobot(ROBOTS[i],p,t,LANE[i]); xs[i]=x;dones[i]=done;
    visualDones[i]=p.timedOut?t>=p.freezeT:done;
    settleRobot(ROBOTS[i],p,t,LANE[i],x);
    const pl=PLAQUES[i];
    const failure=FAILURE_MARKERS[i];
    if(p.failed){
      pl.visible=done;
      if(failure){
        failure.group.visible=done;
        if(done&&failure.kind==='lane') failure.group.position.x=x;
        if(done&&failure.kind==='self-collision'){
          const pulse=1+.08*Math.sin(t*12);
          failure.group.scale.setScalar(pulse);
        }
      }
    }else{
      if(failure) failure.group.visible=false;
      pl.visible=done;
      if(done) pl.position.x=101.6;
    }
  });
  const packMax=Math.max(...xs);
  if(IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON&&validationPolicy===null&&userFollowPolicy===null){
    const next=AUTO_FOLLOW_ORDER.find(i=>!visualDones[i]);
    if(next!==undefined&&next!==currentAutoFollowPolicy){
      currentAutoFollowPolicy=next;resetFollowDamping();updateCameraControls();
    }
  }
  let comparisonFocusIndices=xs.map((_,i)=>i);
  const focusedPolicy=validationPolicy??userFollowPolicy??(IS_TRAJECTORY_COMPARISON?null:currentAutoFollowPolicy);
  const runnerPolicyIndex=focusedPolicy??(!IS_COMPARISON?0:null);
  const runnerFollowComposition=validationPolicy===null&&runnerPolicyIndex!==null;
  const automaticRunnerFollow=IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON&&comparisonAutoCamera&&validationPolicy===null&&userFollowPolicy===null&&focusedPolicy!==null;
  if(IS_TRAJECTORY_COMPARISON&&focusedPolicy===null){
    // Keep an all-lanes fallback for explicitly unfocused comparison contexts.
    // Normal trajectory playback focuses its currently emphasized policy.
    comparisonFocusIndices=xs.map((_,i)=>i);
  }else if(IS_COMPARISON&&focusedPolicy===null){
    // Broadcast framing follows the leading *active* pack. Once those runners
    // finish or are disqualified, they leave the camera calculation and the
    // view returns to the quickest runner still on the course.
    const activeIndices=xs.map((_,i)=>i).filter(i=>!visualDones[i]);
    if(activeIndices.length){
      const activeLeader=Math.max(...activeIndices.map(i=>xs[i]));
      comparisonFocusIndices=activeIndices.filter(i=>xs[i]>=activeLeader-8);
    }else if(CAMERA_FOLLOW.x!==null){
      // Hold the last live-race composition after the final terminal event.
      comparisonFocusIndices=[];
    }
  }
  const comparisonFocusXs=comparisonFocusIndices.map(i=>xs[i]);
  const focusMin=Math.min(...comparisonFocusXs),focusMax=Math.max(...comparisonFocusXs);
  const holdingFinishedComparison=IS_COMPARISON&&focusedPolicy===null&&!comparisonFocusIndices.length;
  const rawPackX=holdingFinishedComparison?CAMERA_FOLLOW.x:
    (focusedPolicy===null?(IS_COMPARISON?(focusMin+focusMax)/2:packMax):xs[focusedPolicy]);
  const rawPackY=holdingFinishedComparison?CAMERA_FOLLOW.y:
    (focusedPolicy===null&&IS_COMPARISON
      ?comparisonFocusIndices.reduce((sum,i)=>sum+LANE[i],0)/comparisonFocusIndices.length
      :(focusedPolicy===null?0:LANE[focusedPolicy]));
  const occupiedLaneSpan=IS_TRAJECTORY_COMPARISON&&LANE.length
    ?Math.max(...LANE)-Math.min(...LANE):0;
  const rawPackSpan=holdingFinishedComparison?CAMERA_FOLLOW.span:
    (focusedPolicy===null&&IS_COMPARISON?Math.max(focusMax-focusMin,occupiedLaneSpan):0);
  // Overview cameras can ease across the pack. A close follow instead uses the
  // centred recorded-root filter: never apply a trailing spring to translation.
  const followDt=CAMERA_FOLLOW.t===null?0:t-CAMERA_FOLLOW.t;
  if(CAMERA_FOLLOW.x===null||followDt<0||followDt>.35){
    CAMERA_FOLLOW.x=rawPackX;CAMERA_FOLLOW.y=rawPackY;CAMERA_FOLLOW.span=rawPackSpan;
  }else{
    const followAlpha=1-Math.exp(-Math.max(0,followDt)*6);
    if(Math.abs(rawPackX-CAMERA_FOLLOW.x)>.018)
      CAMERA_FOLLOW.x+=(rawPackX-CAMERA_FOLLOW.x)*followAlpha;
    // Focused runners stay anchored to the physical lane centre. Never copy
    // gait or limb sway into the camera's lateral target.
    CAMERA_FOLLOW.y=rawPackY;
    CAMERA_FOLLOW.span+=(rawPackSpan-CAMERA_FOLLOW.span)*followAlpha;
  }
  if(runnerFollowComposition)CAMERA_FOLLOW.x=cameraRootX(POL[runnerPolicyIndex],t);
  CAMERA_FOLLOW.t=t;
  const packX=CAMERA_FOLLOW.x;
  const packY=CAMERA_FOLLOW.y;
  const targetXOffset=validationPolicy!==null?validationTargetXOffset:
    (runnerFollowComposition?FOLLOW_VIEW.targetXOffset:validationTargetXOffset);
  let targetZ=validationPolicy!==null?(validationTargetZ===null?0.75:validationTargetZ):
    (runnerFollowComposition?FOLLOW_VIEW.targetZ:(validationTargetZ===null?0.75:validationTargetZ));
  let desiredBodyOffsetX=targetXOffset,desiredTargetZ=targetZ;
  if(runnerFollowComposition){
    // Keep a fallen or crawling runner in frame without changing the close,
    // low-angle standing composition.  A fixed torso-height target otherwise
    // pushes a horizontal body through the bottom edge of the viewport.
    const bounds=new THREE.Box3().setFromObject(ROBOTS[runnerPolicyIndex].grp);
    if(Number.isFinite(bounds.min.z)&&Number.isFinite(bounds.max.z)){
      const center=bounds.getCenter(new THREE.Vector3());
      desiredTargetZ=Math.min(FOLLOW_VIEW.targetZ,Math.max(.10,center.z-.10));
      // Once the robot is horizontal, center its whole physical envelope
      // rather than its pelvis so neither head nor feet fall under the toolbar.
      if(bounds.max.z<1.25)desiredBodyOffsetX=center.x-xs[runnerPolicyIndex];
    }
    desiredTargetZ=cameraPostureTargetZ(POL[runnerPolicyIndex],t,desiredTargetZ);
  }
  const compactFollowScale=IS_SCORING_EXAMPLE?1:(runnerFollowComposition&&followViewportWidth()<560
    ?1.9:1);
  if(compactFollowScale>1||IS_SCORING_EXAMPLE)desiredTargetZ+=.06;
  const snapFollow=CAMERA_FOLLOW.z===null||followDt<0||followDt>.35;
  if(snapFollow){CAMERA_FOLLOW.z=desiredTargetZ;CAMERA_FOLLOW.bodyOffsetX=desiredBodyOffsetX;}
  else{
    const bodyAlpha=1-Math.exp(-Math.max(0,followDt)*5);
    if(Math.abs(desiredTargetZ-CAMERA_FOLLOW.z)>.012)
      CAMERA_FOLLOW.z+=(desiredTargetZ-CAMERA_FOLLOW.z)*bodyAlpha;
    // Smooth posture relative to the moving root, not its absolute position;
    // filtering absolute X here used to add a second several-metre live lag.
    if(Math.abs(desiredBodyOffsetX-CAMERA_FOLLOW.bodyOffsetX)>.018)
      CAMERA_FOLLOW.bodyOffsetX+=(desiredBodyOffsetX-CAMERA_FOLLOW.bodyOffsetX)*bodyAlpha;
  }
  targetZ=CAMERA_FOLLOW.z;
  camT.set(packX+CAMERA_FOLLOW.bodyOffsetX,packY,targetZ);
  const activeView=automaticRunnerFollow?FOLLOW_VIEW:VIEW;
  const viewDist=(automaticRunnerFollow?FOLLOW_VIEW.dist:(comparisonAutoCamera
    ?Math.max(DEFAULT_VIEW.dist,Math.min(72,DEFAULT_VIEW.dist+CAMERA_FOLLOW.span*.67))
    :VIEW.dist))*compactFollowScale;
  const ce=Math.cos(activeView.el), se=Math.sin(activeView.el);
  camP.set(camT.x+viewDist*ce*Math.cos(activeView.az),
           camT.y+viewDist*ce*Math.sin(activeView.az),
           camT.z+viewDist*se);
  // Keep the selected broadcast angle rigid relative to the damped follow
  // target. The viewer can still orbit or zoom explicitly.
  applyCameraSwitch();
  camera.position.copy(camP);
  camera.lookAt(camT);
  sun.target.position.set(packX,0,0); sun.position.set(packX+2,-6,15);
  hud(t,xs,dones,runnerPolicyIndex); renderer.render(scene,camera);
}
const btn=document.getElementById('replay');
function setBtn(){btn.innerHTML = playing ? '&#10073;&#10073; Pause' : (playT>=T_END ? '&#9654; Replay' : '&#9654; Play');}
function loop(now){
  if(lastNow===null)lastNow=now;
  playT += (now-lastNow)/1000*speed; lastNow=now; startWall=1;
  if(playT>=T_END){playT=T_END; draw(playT); playing=false; raf=null; setBtn(); return;}
  draw(playT); raf=requestAnimationFrame(loop);}
function play(){ if(playT>=T_END) playT=0; playing=true; lastNow=null;startWall=null;
  if(raf)cancelAnimationFrame(raf); raf=requestAnimationFrame(loop); setBtn(); }
function pause(){ playing=false; if(raf)cancelAnimationFrame(raf); raf=null; lastNow=null; setBtn(); }
btn.addEventListener('click',()=>{ playing?pause():play(); });
window.addEventListener('keydown',e=>{if(e.code==='Space'&&!e.target?.closest?.('button,[role=button],a,input,select,textarea')){e.preventDefault();playing?pause():play();}});
document.querySelectorAll('.seg button').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.seg button').forEach(x=>x.setAttribute('aria-pressed','false'));
  b.setAttribute('aria-pressed','true');speed=parseFloat(b.dataset.s);}));
cards.forEach((card,index)=>{
  if(!card)return;
  card.addEventListener('click',()=>followPolicy(index));
  if(!card.querySelector('.policy-follow'))card.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();followPolicy(index);}});
  card.querySelector('.policy-remove')?.addEventListener('click',event=>{
    event.preventDefault();event.stopPropagation();
    window.dispatchEvent(new CustomEvent('g1:policy-remove',{detail:{captureId:POL[index].capture_id}}));
  });
});
document.getElementById('camera-zoom-in')?.addEventListener('click',()=>zoomCamera(1/1.22));
document.getElementById('camera-zoom-out')?.addEventListener('click',()=>zoomCamera(1.22));
cameraResetBtn?.addEventListener('click',resetCamera);
window.__G1_REPLAY__={
  playback(){return {time:playT,playing,end:T_END,recordedEnd:T_CAPTURE_END,policyEnds:POL.map(p=>p.playbackEndT)};},
  results(){return PLAQUES.map((pl,index)=>({policy:index,label:pl.userData.resultLabel,visible:pl.visible,position:pl.position.toArray(),font:pl.material.map.userData?.font}));},
  seek(t){
    pause(); playT=Math.min(T_END,Math.max(0,Number(t)||0));
    resetFollowDamping();startWall=null; draw(playT); startWall=1; setBtn();
    return playT;
  },
  validationView({policy=null,az=VIEW.az,el=VIEW.el,dist=VIEW.dist,targetZ=0.75,targetXOffset=-0.6}={}){
    validationPolicy=Number.isInteger(policy)?Math.min(POL.length-1,Math.max(0,policy)):null;
    validationTargetZ=Number.isFinite(targetZ)?targetZ:null;
    validationTargetXOffset=Number.isFinite(targetXOffset)?targetXOffset:-0.6;
    VIEW.az=az;VIEW.el=el;VIEW.dist=dist;
    ROBOTS.forEach((r,i)=>{r.grp.visible=validationPolicy===null||i===validationPolicy;});
    resetFollowDamping();updateCameraControls();startWall=null;draw(playT);startWall=1;
    return {policy:validationPolicy,az:VIEW.az,el:VIEW.el,dist:VIEW.dist,targetZ:validationTargetZ,targetXOffset:validationTargetXOffset};
  },
  follow(policy){return followPolicy(policy);},
  orbit(deltaAz,deltaEl=0){return orbitCamera(deltaAz,deltaEl);},
  zoom(factor){return zoomCamera(factor);},
  resetCamera,
  camera(){return {follow:userFollowPolicy,auto:comparisonAutoCamera,autoFollow:currentAutoFollowPolicy,
    transitioning:cameraSwitch!==null,
    responsiveScale:IS_SCORING_EXAMPLE?1:(followViewportWidth()<560?1.9:1),
    viewport:{width:stage.clientWidth,height:stage.clientHeight,aspect:camera.aspect},
    view:{...(IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON&&userFollowPolicy===null?FOLLOW_VIEW:VIEW)},
    position:camera.position.toArray(),target:camT.toArray()};},
  framing(policy=0){
    scene.updateMatrixWorld(true);camera.updateMatrixWorld(true);
    const index=Math.min(POL.length-1,Math.max(0,policy));
    const bounds=new THREE.Box3().setFromObject(ROBOTS[index].grp),screen={minX:Infinity,maxX:-Infinity,minY:Infinity,maxY:-Infinity};
    for(const x of [bounds.min.x,bounds.max.x])for(const y of [bounds.min.y,bounds.max.y])for(const z of [bounds.min.z,bounds.max.z]){
      const point=new THREE.Vector3(x,y,z).project(camera);
      screen.minX=Math.min(screen.minX,point.x);screen.maxX=Math.max(screen.maxX,point.x);
      screen.minY=Math.min(screen.minY,point.y);screen.maxY=Math.max(screen.maxY,point.y);
    }
    const ground=new THREE.Vector3(camT.x,camT.y,0).project(camera);
    return {time:playT,rootX:cameraRootX(POL[index],playT),target:camT.toArray(),groundY:ground.y,screen};
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
    return {time:playT,fps:FPS,meta:DATA.meta,color:'#'+rb.material.color.getHexString(),links:out};
  }
};
window.addEventListener('resize',()=>{resize();if(!playing)draw(playT);});
resize(); draw(0);
if(!IS_SCORING_EXAMPLE&&window.parent!==window){
  const stageWrap=document.querySelector('.stagewrap');let lastLayout='';
  const publishLayout=()=>{
    const mobile=replayMobileLayout(),height=Math.ceil(stageWrap.getBoundingClientRect().height);
    const key=`${mobile}:${height}`;if(key===lastLayout)return;lastLayout=key;
    parent.postMessage({type:'g1:replay-layout',mobile,height,replayGeneration:new URLSearchParams(location.search).get('replayGeneration')||''},location.origin);
  };
  new ResizeObserver(publishLayout).observe(stageWrap);
  window.addEventListener('resize',publishLayout);publishLayout();
}
setBtn();updateCameraControls();if(REPLAY_AUTOPLAY&&!DATA.meta?.start_paused&&!matchMedia('(prefers-reduced-motion: reduce)').matches) setTimeout(play,500);
