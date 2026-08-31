"""Citation metadata, accessible feedback, and mobile-safe presentation."""

import json
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "web/index.html").read_text()
APP = (ROOT / "web/app.js").read_text()
BIBTEX = """@misc{agents100m2026,
  author = {Xiang Li},
  title = {Agents' 100m},
  year = {2026},
  url = {https://agents100m.com/}
}"""


def test_citation_is_last_section_with_verified_metadata_and_accessible_copy() -> None:
    citation = re.search(r'<section[^>]+id="citation"[^>]*>([\s\S]*?)</section>', PAGE).group(1)
    assert PAGE.index('id="inspiration"') < PAGE.index('id="citation"') < PAGE.index('</main>') < PAGE.index('<footer>')
    assert '<h2 id="citation-heading">Citation</h2>' in citation
    assert re.search(r'<code id="citation-bibtex">([\s\S]*?)</code>', citation).group(1) == BIBTEX
    assert 'author = {Xiang Li}' in citation and 'arxiv' not in citation.lower()
    assert 'url = {https://agents100m.com/}' in citation
    assert 'type="button" aria-label="Copy BibTeX citation"' in citation
    assert 'role="status" aria-live="polite" aria-atomic="true"' in citation
    assert citation.index('</code></pre>') < citation.index('class="citation-actions"') < citation.index('id="citation-copy"') < citation.index('id="citation-copy-status"')
    assert "  initCitation();" in APP


def test_citation_wraps_long_lines_without_mobile_horizontal_overflow() -> None:
    css = (ROOT / "web/styles.css").read_text()
    pre = re.search(r'\.citation-box pre\s*\{([^}]+)\}', css).group(1)
    for rule in ['max-width: 100%', 'white-space: pre-wrap', 'overflow-wrap: anywhere', 'var(--mono)', 'color: var(--text)']:
        assert rule in pre
    assert '.citation-box code { font: inherit; }' in css
    assert '#citation-copy:focus-visible' in css
    actions = re.search(r'\.citation-actions\s*\{([^}]+)\}', css).group(1)
    assert 'padding: 0 16px 16px' in actions
    assert 'justify-content: flex-end' not in actions


@pytest.mark.parametrize("clipboard", ["success", "reject", "unavailable"])
def test_citation_copy_reports_only_actual_success(clipboard: str) -> None:
    function = APP[APP.index('  function initCitation(){'):APP.index('  initCitation();')]
    script = "const assert=require('node:assert/strict');\n"
    script += f"const source={json.dumps(BIBTEX)},mode={json.dumps(clipboard)};\n"
    script += "const window={};\n" + next(line for line in APP.splitlines() if line.strip().startswith("const t=")) + "\n"
    script += """
let listener,complete,copied;
const button={disabled:false,textContent:'Copy',addEventListener(type,fn){assert.equal(type,'click');listener=fn}};
const code={textContent:source},status={textContent:''};
const nodes={'#citation-copy':button,'#citation-bibtex':code,'#citation-copy-status':status};
const $=selector=>nodes[selector];
const navigator=mode==='unavailable'?{}:{clipboard:{writeText(text){copied=text;return new Promise((resolve,reject)=>{complete=()=>mode==='reject'?reject(new Error('denied')):resolve()})}}};
""" + function + """
(async()=>{
  initCitation();const promise=listener();
  if(mode!=='unavailable'){
    assert.equal(button.disabled,true);assert.equal(button.textContent,'Copying…');
    assert.equal(status.textContent,'');assert.equal(copied,source);
    complete();
  }
  await promise;assert.equal(button.disabled,false);
  if(mode==='success'){
    assert.equal(button.textContent,'Copied');assert.equal(status.textContent,'Citation copied to clipboard.');
  }else{
    assert.equal(button.textContent,'Copy');assert.match(status.textContent,/copy it manually/);
    assert.doesNotMatch(status.textContent,/Citation copied/);
  }
  // Other pages without this optional section remain safe.
  delete nodes['#citation-copy'];assert.doesNotThrow(initCitation);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
