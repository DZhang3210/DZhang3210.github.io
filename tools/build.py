"""Build the site: src/posts/*.md -> posts/<slug>/index.html, plus index.html (published posts) and
drafts/index.html (posts with `draft: true`). Plain HTML out, no Jekyll (see .nojekyll).

    .venv/bin/python tools/build.py

A post starts with front matter between --- lines (title, date, description, optional draft: true,
optional reading). In the body, {{chart:NAME}} inserts a chart from charts.py.
"""
import glob
import os
import re
from html import escape

import markdown

from charts import CHARTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = "David Zhang"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="stylesheet" href="{base}assets/style.css">
</head>
<body>
<header class="site"><div class="wrap"><a class="name" href="{base}">{site}</a>
<nav><a href="{base}">Writing</a><a href="https://github.com/DZhang3210">GitHub</a></nav></div></header>
<main><div class="wrap">
{body}
</div></main>
<footer class="site"><div class="wrap">{site}</div></footer>
</body>
</html>
"""


def read_post(path):
    text = open(path, encoding="utf-8").read()
    _, front, body = text.split("---", 2)
    meta = dict(line.split(":", 1) for line in front.strip().splitlines())
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    meta["slug"] = os.path.splitext(os.path.basename(path))[0]
    meta["draft"] = meta.get("draft", "false").lower() == "true"
    meta["body"] = body
    return meta


def render_body(md_text):
    charts = {}

    def stash(m):
        key = f"CHARTPLACEHOLDER{len(charts)}"
        charts[key] = CHARTS[m.group(1)]()
        return f"\n\n{key}\n\n"

    md_text = re.sub(r"\{\{chart:(\w+)\}\}", stash, md_text)
    html = markdown.markdown(md_text, extensions=["extra", "smarty", "sane_lists"])
    html = re.sub(r"<table>", '<div class="table-scroll"><table>', html).replace("</table>", "</table></div>")
    for key, chart in charts.items():
        html = html.replace(f"<p>{key}</p>", chart)
    return html


def words(md_text):
    return len(re.sub(r"\{\{[^}]*\}\}|<[^>]+>", " ", md_text).split())


def post_list(posts, prefix):
    items = "".join(
        f'<li><a class="title" href="{prefix}posts/{p["slug"]}/">{escape(p["title"])}</a>'
        f'<div class="date">{escape(p["date"])} · {p["minutes"]} min read</div>'
        f'<p class="desc">{escape(p["description"])}</p></li>'
        for p in posts)
    return f'<ul class="posts">{items}</ul>'


def main():
    posts = [read_post(p) for p in sorted(glob.glob(os.path.join(ROOT, "src", "posts", "*.md")))]
    for p in posts:
        p["minutes"] = max(1, round(words(p["body"]) / 230))
        body = (f'<article><h1>{escape(p["title"])}</h1>'
                f'<div class="meta">{escape(p["date"])} · {p["minutes"]} min read'
                + (" · draft" if p["draft"] else "") + "</div>"
                + render_body(p["body"]) + "</article>")
        out = os.path.join(ROOT, "posts", p["slug"], "index.html")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w", encoding="utf-8").write(PAGE.format(
            title=escape(p["title"]), description=escape(p["description"]), base="../../", site=SITE, body=body))

    published = [p for p in posts if not p["draft"]]
    drafts = [p for p in posts if p["draft"]]
    home = ('<h1>Writing</h1><p class="lede">Notes on machine-learning projects I build: what I tried, what worked, and what I learned.</p>'
            + (post_list(published, "") if published else "<p>First post coming soon.</p>"))
    open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(PAGE.format(
        title=SITE, description="Writing by David Zhang", base="", site=SITE, body=home))
    if not drafts:   # no drafts: no drafts page
        print(f"built {len(posts)} posts (0 drafts)")
        return
    os.makedirs(os.path.join(ROOT, "drafts"), exist_ok=True)
    draft_page = "<h1>Drafts</h1><p>Not linked from the home page.</p>" + post_list(drafts, "../")
    open(os.path.join(ROOT, "drafts", "index.html"), "w", encoding="utf-8").write(PAGE.format(
        title=f"Drafts · {SITE}", description="Drafts", base="../", site=SITE, body=draft_page))
    print(f"built {len(posts)} posts ({len(drafts)} drafts)")


if __name__ == "__main__":
    main()
