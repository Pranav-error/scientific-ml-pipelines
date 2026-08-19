"""Find broken internal links in kubernetes/website English docs.

Conservative by design: a PR full of false positives is worse than no PR. Anything
that cannot be resolved with certainty is reported separately as "unsure" rather than
claimed as broken.

Hugo maps content/en/docs/a/b.md      -> /docs/a/b/
              content/en/docs/a/_index.md -> /docs/a/
so a link target /docs/a/b/ is valid if either b.md or b/_index.md exists. Front-matter
`aliases:` also create valid URLs, so those are collected first.
"""
import os, re, sys, json, collections

ROOT = os.path.expanduser("~/Documents/gsoc-prep/k8s-website/content/en")
DOCS = os.path.join(ROOT, "docs")

LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
ALIAS_BLOCK = re.compile(r"^aliases:\s*\n((?:\s*-\s*\S+\n)+)", re.M)
ALIAS_INLINE = re.compile(r"^aliases:\s*\[([^\]]*)\]", re.M)

# generated at build time or served from elsewhere - not resolvable from this repo
SKIP_PREFIX = (
    "/docs/reference/generated/", "/docs/reference/kubernetes-api/",
    "/docs/reference/command-line-tools-reference/kubelet-authentication-authorization/",
    "/_print", "/blog/", "/training/", "/case-studies/", "/partners/", "/community/",
    "/releases/", "/docs/reference/issues-security/", "/search",
)


def url_for(path):
    rel = os.path.relpath(path, ROOT)
    rel = rel[:-3] if rel.endswith(".md") else rel
    # Hugo: branch bundles use _index.md, LEAF bundles use index.md. Both map the
    # directory itself to a URL. Missing the index.md case produces mass false positives.
    for marker in ("/_index", "/index"):
        if rel.endswith(marker):
            rel = rel[: -len(marker)]
            break
    else:
        if rel in ("_index", "index"):
            rel = ""
    return "/" + rel + "/" if rel else "/"


def main():
    files = []
    for dp, _, fn in os.walk(DOCS):
        for f in fn:
            if f.endswith(".md"):
                files.append(os.path.join(dp, f))
    print(f"scanning {len(files)} files")

    valid = set()
    for p in files:
        valid.add(url_for(p).rstrip("/") or "/")
    # aliases
    for p in files:
        txt = open(p, encoding="utf-8", errors="replace").read()
        head = txt[:3000]
        for m in ALIAS_BLOCK.finditer(head):
            for line in m.group(1).splitlines():
                a = line.strip().lstrip("-").strip().strip("\"'")
                if a.startswith("/"):
                    valid.add(a.rstrip("/") or "/")
        for m in ALIAS_INLINE.finditer(head):
            for a in m.group(1).split(","):
                a = a.strip().strip("\"'")
                if a.startswith("/"):
                    valid.add(a.rstrip("/") or "/")

    broken, unsure = [], []
    for p in files:
        txt = open(p, encoding="utf-8", errors="replace").read()
        for m in LINK.finditer(txt):
            tgt = m.group(2).strip()
            if tgt.startswith(("http://", "https://", "mailto:", "#", "<")):
                continue
            if "{{" in tgt or "{%" in tgt:      # shortcode-built link
                continue
            base = tgt.split("#")[0].split("?")[0]
            if not base:
                continue
            line = txt[:m.start()].count("\n") + 1
            rec = {"file": os.path.relpath(p, ROOT), "line": line,
                   "text": m.group(1)[:40], "target": tgt}

            if base.startswith("/"):
                if base.startswith(SKIP_PREFIX):
                    continue
                if not base.startswith("/docs/"):
                    unsure.append(rec)
                    continue
                if base.rstrip("/") in valid:
                    continue
                broken.append(rec)
            else:
                # relative link
                cand = os.path.normpath(os.path.join(os.path.dirname(p), base))
                if os.path.exists(cand) or os.path.exists(cand + ".md") or \
                   os.path.exists(os.path.join(cand, "_index.md")) or os.path.exists(os.path.join(cand, "index.md")):
                    continue
                if any(base.endswith(e) for e in (".yaml", ".json", ".sh", ".png", ".svg",
                                                  ".jpg", ".txt", ".conf", ".crt")):
                    unsure.append(rec)   # example manifests live elsewhere
                    continue
                broken.append(rec)

    print(f"\nBROKEN (high confidence): {len(broken)}")
    by_target = collections.Counter(b["target"] for b in broken)
    for t, n in by_target.most_common(25):
        ex = next(b for b in broken if b["target"] == t)
        print(f"  {n:>3}x  {t}")
        print(f"        e.g. {ex['file']}:{ex['line']}  [{ex['text']}]")
    print(f"\nunsure (not reported): {len(unsure)}")
    json.dump({"broken": broken, "unsure": unsure},
              open(os.path.expanduser("~/Documents/gsoc-prep/k8s_links.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
