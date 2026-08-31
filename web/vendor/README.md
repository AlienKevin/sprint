# jsdiff 9.0.0

`jsdiff-9.0.0.min.js` is the `dist/diff.min.js` browser distribution from the
published `diff@9.0.0` npm package, with only a final newline added. It creates `window.Diff` and is
served locally (no CDN/runtime dependency). The full BSD-3-Clause license is
in `jsdiff-9.0.0.LICENSE`.

- Source and API documentation: https://github.com/kpdecker/jsdiff
- Package: https://registry.npmjs.org/diff/-/diff-9.0.0.tgz
- Package SHA-512: `svtcdpS8CgJyqAjEQIXdb3OjhFVVYjzGAPO8WGCmRbrml64SPw/jJD4GoE98aR7r25A0XcgrK3F02yw9R/vhQw==`
- Local distribution SHA-256: `c850fa0f149286b8b52b7dddb939f29bf84891b3378541478a3bbc2e0339d9df`

The trajectory viewer uses only `diffArrays` and `diffChars`, with input,
edit-distance, time, and DOM-size limits. To update, replace the pinned browser
distribution and license together and rerun `tests/test_trajectory_diff.py`.
