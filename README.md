# DZhang3210.github.io

Personal site: writing about machine-learning projects. Plain HTML, served by GitHub Pages (no Jekyll; see `.nojekyll`).

**Write a post:** add `src/posts/<slug>.md` with front matter

```
---
title: ...
date: November 2026
description: one sentence for the post list
draft: true          # optional: listed under /drafts/ only, not on the home page
---
```

then rebuild and commit the generated HTML:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
.venv/bin/python tools/build.py
```

Files in `src/unpublished/` are kept but not built (two other-length versions of the address-matching post).

`{{chart:NAME}}` in a post inserts an SVG chart defined in `tools/charts.py`.
