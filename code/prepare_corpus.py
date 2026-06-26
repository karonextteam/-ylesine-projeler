from __future__ import annotations

from pathlib import Path


def read_all_texts(clean_dir: Path) -> str:
    parts: list[str] = []

    for p in sorted(clean_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            print(f"[WARN] boş dosya atlandı: {p.name}")
            continue

        parts.append(f"\n\n<BOOK_START:{p.stem}>\n\n")
        parts.append(text)
        parts.append(f"\n\n<BOOK_END:{p.stem}>\n\n")

        print(f"[OK] eklendi: {p.name} ({len(text)} chars)")

    return "".join(parts).strip() + "\n"


def main() -> None:
    base = Path(__file__).resolve().parent
    clean_dir = base / "datasets_clean"
    out_dir = base / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not clean_dir.exists():
        raise SystemExit(f"Missing folder: {clean_dir}")

    text = read_all_texts(clean_dir)

    n = len(text)
    split = int(n * 0.9)

    train = text[:split]
    val = text[split:]

    (out_dir / "train.txt").write_text(train, encoding="utf-8")
    (out_dir / "val.txt").write_text(val, encoding="utf-8")

    print(f"\n[OK] Wrote data/full_corpus.txt ({len(text)} chars)")
    print(f"[OK] Wrote data/train.txt ({len(train)} chars)")
    print(f"[OK] Wrote data/val.txt   ({len(val)} chars)")


if __name__ == "__main__":
    main()