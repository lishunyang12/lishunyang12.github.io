# lishunyang12.github.io

Sylar's personal blog. Built with [Stuart](https://github.com/w-henderson/Stuart)
(a static site generator) + SCSS, and deployed to GitHub Pages via GitHub Actions.

- `content/` — pages and blog posts (`content/blog/*.md`)
- `styles/` — SCSS, compiled to `style.css` at build time
- `static/` — static assets served at the web root
- `gen_post.py` — helper that generates the benchmark post (charts/tables) from raw data

## Build locally

```bash
sass styles/index.scss static/style.css --no-source-map --style=compressed
stuart build   # outputs to dist/
```
