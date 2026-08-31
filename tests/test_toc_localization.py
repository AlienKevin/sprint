"""Authored outlines may be localized; trace evidence and navigation IDs may not."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def run_node(body: str) -> None:
    setup = """
const assert = require('node:assert/strict');
const fs = require('node:fs'), vm = require('node:vm');
global.window = {SiteI18n: {language: 'zh-CN'}};
vm.runInThisContext(fs.readFileSync('web/locales/toc.js', 'utf8'));
const api = window.SiteTOCI18n;
const batch = JSON.parse(fs.readFileSync('web/data/batches/current.json', 'utf8'));
const originals = batch.arms.map(arm => JSON.parse(fs.readFileSync(`web/data/trajectories/${arm.run_id}.outline.json`, 'utf8')));
"""
    result = subprocess.run(["node", "-e", setup + body], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_all_current_outlines_and_chapters_have_pinned_chinese_translations() -> None:
    run_node("""
const coverage = api.coverage();
assert.equal(coverage.length, 15);
assert.equal(coverage.reduce((sum, row) => sum + row.chapters, 0), 195);
assert.deepEqual(coverage.map(row => row.run_id).sort(), originals.map(row => row.run_id).sort());
for (const original of originals) {
  const row = coverage.find(row => row.run_id === original.run_id);
  assert.equal(row.source_fingerprint, original.trajectory.source_fingerprint);
  assert.equal(row.chapters, original.chapters.length);
  const translated = api.translate(original);
  assert.notEqual(translated, original, original.run_id);
  assert.match(translated.synopsis, /[\u3400-\u9fff]/);
  for (const chapter of translated.chapters) {
    assert.match(chapter.title, /[\u3400-\u9fff]/);
    assert.match(chapter.summary, /[\u3400-\u9fff]/);
  }
}
""")


def test_translation_preserves_numbers_ids_boundaries_and_original_objects() -> None:
    run_node(r"""
const numbers = value => (value.match(/\d+(?:\.\d+)?/g) || []).sort();
for (const original of originals) {
  const before = JSON.stringify(original), translated = api.translate(original, 'zh-CN');
  const {synopsis: originalSynopsis, chapters: originalChapters, ...originalMeta} = original;
  const {synopsis, chapters, ...meta} = translated;
  assert.deepEqual(meta, originalMeta);
  assert.deepEqual(numbers(synopsis), numbers(originalSynopsis), original.run_id + ':synopsis');
  for (let index = 0; index < chapters.length; index++) {
    const {title: originalTitle, summary: originalSummary, ...originalBoundary} = originalChapters[index];
    const {title, summary, ...boundary} = chapters[index];
    assert.deepEqual(boundary, originalBoundary);
    assert.deepEqual(numbers(title), numbers(originalTitle), original.run_id + ':' + boundary.id + ':title');
    assert.deepEqual(numbers(summary), numbers(originalSummary), original.run_id + ':' + boundary.id + ':summary');
  }
  assert.equal(JSON.stringify(original), before, 'English evidence must remain untouched');
  assert.equal(api.translate(original, 'en'), original);
  assert.equal(api.translate(original, 'fr'), original);
}
""")


def test_unknown_or_stale_outlines_fall_back_to_original_english() -> None:
    run_node("""
const original = originals[0];
for (const changed of [
  {...original, run_id: 'not-a-published-trial'},
  {...original, run_id: '__proto__', trajectory: {}},
  {...original, trajectory: {...original.trajectory, source_fingerprint: 'stale'}},
  {...original, chapters: original.chapters.slice(1)},
  {...original, chapters: original.chapters.map((c, i) => i ? c : {...c, id: 'unknown'})},
]) assert.equal(api.translate(changed, 'zh-CN'), changed);
assert.equal(api.translate(null, 'zh-CN'), null);
assert.equal(api.translate(original, 'zh').synopsis, api.translate(original, 'zh-CN').synopsis);
""")
