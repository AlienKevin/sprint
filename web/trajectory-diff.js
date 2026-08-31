/* Read-only Edit/MultiEdit presentation. jsdiff is vendored locally; source text
   is never interpreted as HTML or used to apply an edit. */
(() => {
  'use strict';
  const limits = Object.freeze({characters:100000,lines:3000,rows:1500,editLength:1200,lineMs:25,inlineMs:12,inlinePairs:80,inlineCharacters:1000});
  const pending = new WeakMap();
  const now = () => performance.now();
  const splitLines = value => value.match(/[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$/g) || [];
  const lineEnding = value => value.endsWith('\r\n') ? '\r\n' : /[\r\n]$/.test(value) ? value.at(-1) : '';
  const lineBody = value => value.slice(0,value.length-lineEnding(value).length);
  const text = (tag,className,value) => {const node=document.createElement(tag);if(className)node.className=className;if(value!==undefined)node.textContent=value;return node};
  function label(node,source,params={}) {
    node.dataset.trajectoryI18n=source;node.dataset.trajectoryParams=JSON.stringify(params);
    node.textContent=window.SiteI18n?.t(`trajectory.${source}`,params,source)??source.replace(/\{(\w+)\}/g,(match,key)=>params[key]??match);
    return node;
  }
  function inlinePair(oldRow,newRow,deadline) {
    const before=lineBody(oldRow.raw),after=lineBody(newRow.raw);
    if(before.length+after.length>limits.inlineCharacters||now()>=deadline)return;
    const changes=window.Diff.diffChars(before,after,{timeout:Math.max(1,Math.floor(deadline-now())),maxEditLength:256});
    if(!changes)return;
    // Whole unrelated lines already have a background; stronger highlights are
    // reserved for replacements that retain some meaningful surrounding text.
    if(changes.some(part=>!part.added&&!part.removed&&part.value.trim())) {
      oldRow.segments=changes.filter(part=>!part.added).map(part=>({text:part.value,changed:!!part.removed}));
      newRow.segments=changes.filter(part=>!part.removed).map(part=>({text:part.value,changed:!!part.added}));
    }
    if(lineEnding(oldRow.raw)!==lineEnding(newRow.raw)){oldRow.showEnding=true;newRow.showEnding=true;}
  }
  function compute(before,after) {
    if(typeof before!=='string'||typeof after!=='string')return {status:'unavailable',rows:[]};
    if(before===after)return {status:'unchanged',rows:[],added:0,removed:0};
    if(before.length+after.length>limits.characters)return {status:'limited',rows:[]};
    const oldLines=splitLines(before),newLines=splitLines(after);
    if(oldLines.length+newLines.length>limits.lines)return {status:'limited',rows:[]};
    if(!window.Diff?.diffArrays||!window.Diff?.diffChars)return {status:'unavailable',rows:[]};
    const changes=window.Diff.diffArrays(oldLines,newLines,{timeout:limits.lineMs,maxEditLength:limits.editLength});
    if(!changes)return {status:'limited',rows:[]};
    const rows=[];let oldNumber=1,newNumber=1,added=0,removed=0;
    for(const change of changes){
      const kind=change.added?'add':change.removed?'remove':'context';
      for(const raw of change.value){
        rows.push({kind,raw,oldNumber:change.added?null:oldNumber++,newNumber:change.removed?null:newNumber++});
        if(change.added)added++;if(change.removed)removed++;
      }
    }
    if(rows.length>limits.rows)return {status:'limited',rows:[]};
    const deadline=now()+limits.inlineMs;let pairs=0;
    for(let index=0;index<rows.length&&now()<deadline&&pairs<limits.inlinePairs;index++){
      if(rows[index].kind!=='remove')continue;
      const removedRows=[];while(rows[index]?.kind==='remove')removedRows.push(rows[index++]);
      const addedRows=[];while(rows[index]?.kind==='add')addedRows.push(rows[index++]);
      for(let pair=0;pair<Math.min(removedRows.length,addedRows.length)&&pairs<limits.inlinePairs;pair++,pairs++)inlinePair(removedRows[pair],addedRows[pair],deadline);
      index--;
    }
    return {status:'ready',rows,added,removed};
  }
  function rowElement(row) {
    const line=text('div',`diff-line diff-${row.kind}`);
    // These are positions in the supplied replacement fragments, not full-file
    // line numbers: Edit arguments do not carry a reliable full-file offset.
    const oldNumber=text('span','diff-line-number diff-old-number',row.oldNumber??'');
    const newNumber=text('span','diff-line-number diff-new-number',row.newNumber??'');
    const sign=text('span','diff-sign',row.kind==='add'?'+':row.kind==='remove'?'−':' ');
    oldNumber.setAttribute('aria-hidden','true');newNumber.setAttribute('aria-hidden','true');
    const code=text('code','diff-code');
    if(row.segments)for(const segment of row.segments)code.append(text('span',segment.changed?'diff-inline-change':'',segment.text));
    else code.textContent=lineBody(row.raw);
    if(row.kind!=='context'&&(!lineEnding(row.raw)||row.showEnding)) {
      const ending=lineEnding(row.raw),caption=ending==='\r\n'?'CRLF':ending==='\n'?'LF':ending==='\r'?'CR':'No newline';
      code.append(label(text('span','diff-line-ending'),caption));
    }
    line.append(oldNumber,newNumber,sign,code);return line;
  }
  function appendRows(body,rows) {
    for(let index=0;index<rows.length;){
      if(rows[index].kind!=='context'){body.append(rowElement(rows[index++]));continue;}
      const start=index;while(rows[index]?.kind==='context')index++;
      const count=index-start,keepBefore=start===0?0:3,keepAfter=index===rows.length?0:3;
      if(count<=keepBefore+keepAfter+2){for(let cursor=start;cursor<index;cursor++)body.append(rowElement(rows[cursor]));continue;}
      for(let cursor=start;cursor<start+keepBefore;cursor++)body.append(rowElement(rows[cursor]));
      const hidden=rows.slice(start+keepBefore,index-keepAfter),gap=text('div','diff-context-gap');
      const button=label(text('button','diff-context-toggle'),'{count} unchanged lines',{count:hidden.length});
      button.type='button';button.setAttribute('aria-expanded','false');
      button.addEventListener('click',event=>{event.stopPropagation();const fragment=document.createDocumentFragment();for(const row of hidden)fragment.append(rowElement(row));gap.replaceWith(fragment);});
      gap.append(button);body.append(gap);
      for(let cursor=index-keepAfter;cursor<index;cursor++)body.append(rowElement(rows[cursor]));
    }
  }
  function render(node) {
    const source=pending.get(node);if(!source)return;
    pending.delete(node);observer?.unobserve(node);
    let result;try{result=compute(source.before,source.after)}catch{result={status:'unavailable',rows:[]}}
    node.replaceChildren();node.dataset.diffState=result.status;
    if(result.status!=='ready') {
      const message=result.status==='unchanged'?'No changes':result.status==='limited'?'This edit is too large for an inline diff. View the raw arguments below.':'Diff unavailable. View the raw arguments below.';
      node.append(label(text('p','diff-message'),message));return;
    }
    const heading=text('div','diff-heading'),name=label(text('span','diff-title'),'Changes');
    const stats=text('span','diff-stats');stats.append(text('span','diff-added-count',`+${result.added}`),text('span','diff-removed-count',`−${result.removed}`));
    heading.append(name,stats);
    const note=label(text('p','diff-note'),'Line numbers refer to the edited excerpt.');
    const body=text('div','diff-lines');appendRows(body,result.rows);
    node.append(heading,note,body);
  }
  function renderWithin(container) {
    for(const node of container.querySelectorAll('.trajectory-diff')){
      // An open turn can contain manually closed tools. Do not diff those until
      // they are expanded or scrolled into view.
      if(!node.closest('details:not([open])'))render(node);
    }
  }
  const observer=typeof IntersectionObserver==='function'?new IntersectionObserver(entries=>{for(const entry of entries){if(entry.target.isConnected===false){pending.delete(entry.target);observer.unobserve(entry.target);}else if(entry.isIntersecting&&!entry.target.closest('details:not([open])'))render(entry.target);}},{rootMargin:'160px'}):null;
  function create(before,after) {
    const node=text('section','trajectory-diff');node.dataset.diffState='pending';
    node.append(label(text('p','diff-message'),'Loading diff…'));pending.set(node,{before,after});
    if(observer)observer.observe(node);
    // Native details emits toggle after attachment; this also covers browsers
    // without IntersectionObserver. A detached node never needs eager diffing.
    return node;
  }
  document.addEventListener('toggle',event=>{if(event.target.open)renderWithin(event.target)},true);
  window.TrajectoryDiff=Object.freeze({create,renderWithin,compute,limits});
})();
