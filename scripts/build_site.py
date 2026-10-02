"""Assemble docs/index.html (served by GitHub Pages) from the template and bundle."""
from __future__ import annotations

import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REPO = os.environ.get("WFV_REPO", "alanhuang116/wildfirevuln")


def main():
    with open(os.path.join(ROOT, "web", "site_template.html"), encoding="utf-8") as fh:
        html = fh.read()
    with open(os.path.join(ROOT, "web", "bundle.json"), encoding="utf-8") as fh:
        bundle = fh.read()
    assert "/*__BUNDLE__*/null" in html
    html = html.replace("/*__BUNDLE__*/null", bundle).replace("__REPO__", REPO)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    out = os.path.join(ROOT, "docs", "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
