// Read-only local replay validation. See replay_performance_README.md.
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import {brotliCompressSync, constants} from 'node:zlib';
const root=path.resolve(process.env.REPLAY_WEB_ROOT||new URL('../../web/',import.meta.url).pathname), snapshot=process.argv[3]||'', port=Number(process.argv[2]||59454), cache=new Map();
const mime={'.html':'text/html; charset=utf-8','.js':'application/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json','.png':'image/png','.jpg':'image/jpeg','.woff2':'font/woff2','.svg':'image/svg+xml'};
http.createServer(async(req,res)=>{
 try{
  const pathname=decodeURIComponent(new URL(req.url,'http://local').pathname);
  if(pathname==='/__fixture_info'){res.setHeader('Content-Type','application/json');return res.end(JSON.stringify({root,encoding:'br quality5',config:'web/vercel.json',compressedEntries:cache.size}));}
  let file=path.resolve(root,'.'+pathname);if(!file.startsWith(root+'/')&&file!==root)throw Error('outside root');
  if(pathname==='/')file=path.join(root,'index.html');
  let stat;try{stat=await fs.stat(file);}catch{file+='.html';stat=await fs.stat(file);}
  if(!stat.isFile())throw Error('not file');
  if(snapshot){const frozen=path.join(snapshot,path.relative(root,file));try{const frozenStat=await fs.stat(frozen);file=frozen;stat=frozenStat;}catch{}}
  const conf=JSON.parse(await fs.readFile(path.join(snapshot||root,'vercel.json'),'utf8'));let cc='public, max-age=60';
  for(const rule of conf.headers||[]){if(new RegExp('^'+rule.source+'$').test(pathname)){for(const h of rule.headers||[])if(h.key.toLowerCase()==='cache-control')cc=h.value;}}
  const compress=/\bbr\b/.test(req.headers['accept-encoding']||'')&&/\.(html|js|css|json|svg)$/.test(file);
  const key=file+':'+stat.mtimeMs+':'+compress;
  let body=cache.get(key);if(!body){body=await fs.readFile(file);if(compress)body=brotliCompressSync(body,{params:{[constants.BROTLI_PARAM_QUALITY]:5}});cache.set(key,body);}
  res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.setHeader('Cache-Control',cc);res.setHeader('Content-Length',body.length);res.setHeader('Vary','Accept-Encoding');if(compress)res.setHeader('Content-Encoding','br');
  res.end(req.method==='HEAD'?undefined:body);
 }catch(err){res.statusCode=404;res.end('Not found');}
}).listen(port,'127.0.0.1',()=>console.log('Replay fixture http://127.0.0.1:'+port));
