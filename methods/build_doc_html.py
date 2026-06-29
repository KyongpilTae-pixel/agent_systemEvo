"""methods 패키지 .md 문서 → 스타일 적용 HTML (자체완결형).

markdown(tables/fenced_code) 렌더 + 상대 .md 링크를 .html 로 치환.
사용:
  python -m agent_system.methods.build_doc_html PERCELL_ISO.md GENUS_ISO.md README.md
  (인자 없으면 PERCELL_ISO.md GENUS_ISO.md 기본)
"""
from __future__ import annotations

import re
import sys

import markdown

from agent_system.methods import base as B

DEFAULT = ["PERCELL_ISO.md", "GENUS_ISO.md"]

CSS = """
 body{font-family:-apple-system,system-ui,'Malgun Gothic',sans-serif;background:#f4f6f9;
   margin:0;color:#1f2430;line-height:1.65}
 .wrap{max-width:980px;margin:0 auto;padding:30px 34px 90px;background:#fff;
   box-shadow:0 1px 4px rgba(0,0,0,.06);min-height:100vh}
 h1{font-size:24px;border-bottom:3px solid #2979ff;padding-bottom:10px;margin-top:6px}
 h2{font-size:18px;color:#1565c0;margin:30px 0 8px;border-left:5px solid #2979ff;padding-left:11px}
 h3{font-size:15px;color:#37474f;margin:20px 0 6px}
 blockquote{border-left:4px solid #2979ff;background:#f5f9ff;color:#444;margin:14px 0;
   padding:8px 16px;font-size:13.5px}
 table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;
   box-shadow:0 1px 2px rgba(0,0,0,.05)}
 th,td{border:1px solid #e3e7ee;padding:7px 10px;text-align:left;vertical-align:top}
 th{background:#eef4ff;color:#0d47a1}
 code{background:#eceff3;padding:1px 6px;border-radius:4px;font-size:12.5px;
   font-family:ui-monospace,Menlo,monospace}
 pre{background:#1e1e2e;color:#dce3f0;border-radius:8px;padding:14px 16px;overflow-x:auto;
   font-size:12.5px;line-height:1.5}
 pre code{background:none;color:inherit;padding:0}
 a{color:#1565c0;text-decoration:none} a:hover{text-decoration:underline}
 ul,ol{padding-left:22px} li{margin:3px 0}
 hr{border:none;border-top:1px solid #e0e4ea;margin:26px 0}
 .foot{color:#888;font-size:11.5px;margin-top:30px;border-top:1px solid #ddd;padding-top:12px}
"""


def render(md_name: str) -> str:
    md_path = B.PKG_DIR / md_name
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    # 상대 .md 링크 → .html
    body = re.sub(r'href="([^"]+)\.md"', r'href="\1.html"', body)
    title = md_name.replace(".md", "")
    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — methods</title><style>{CSS}</style></head>
<body><div class="wrap">
{body}
<p class="foot">rendered from <code>agent_system/methods/{md_name}</code> ·
<code>python -m agent_system.methods.build_doc_html {md_name}</code></p>
</div></body></html>"""
    out = B.PKG_DIR / (title + ".html")
    out.write_text(html, encoding="utf-8")
    return str(out)


def main() -> None:
    names = sys.argv[1:] or DEFAULT
    for n in names:
        print("saved ->", render(n))


if __name__ == "__main__":
    main()
