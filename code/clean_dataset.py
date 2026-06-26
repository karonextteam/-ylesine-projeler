import re
from pathlib import Path


def _normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\f", "\n")
    return text


def _remove_pdf_artifacts(text: str) -> str:
    # Soft hyphen and zero-width characters are common in PDF-extracted text.
    text = text.replace("\u00ad", "")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")
    return text


def _strip_gutenberg_like_boilerplate(lines: list[str]) -> list[str]:
    # Keep generic; doesn't assume any specific source.
    drop_patterns = [
        r"^\s*\*\s*\*\s*\*\s*$",
        r"^\s*Bu sayısal eser hakkında\s*$",
        r"^\s*↑\s*https?://",
        r"Vikikaynak",
        r"Creative\s*Commons",
        r"ISBN\s*\d",
        r"Yayınları",
        r"Yayınevi",
        r"Baskı",
        r"Kapak\s*Tasarımı",
        r"Matbaası",
        r"telif hakları",
        r"©",
        r"Telefon:",
        r"Faks:",
        r"İstiklal Caddesi",
        r"Yapı Kredi",
        r"10 Nisan 2026 tarihinde oluşturuldu",
    ]
    drop_re = re.compile("(?:" + "|".join(drop_patterns) + ")", re.IGNORECASE)

    cleaned: list[str] = []
    for line in lines:
        if drop_re.search(line):
            continue
        cleaned.append(line)
    return cleaned


def _remove_page_headers(lines: list[str]) -> list[str]:
    # Example: "İçimizdeki Şeytan 9" or "4 Sabahattin Ali"
    out: list[str] = []
    header_re = re.compile(r"^\s*(?:\d+\s+)?[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü'’\- ]{0,40}\s+\d+\s*$")
    for line in lines:
        if header_re.match(line):
            continue
        out.append(line)
    return out


def _collapse_whitespace(lines: list[str]) -> list[str]:
    out: list[str] = []
    blank_streak = 0
    for raw in lines:
        line = raw.rstrip()
        # normalize tabs and repeated spaces
        line = re.sub(r"\t+", " ", line)
        line = re.sub(r"[ ]{2,}", " ", line)
        if not line.strip():
            blank_streak += 1
            if blank_streak <= 1:
                out.append("")
            continue
        blank_streak = 0
        out.append(line.strip(" "))
    # trim leading/trailing blanks
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return out


def _dehyphenate_wrapped_words(lines: list[str]) -> list[str]:
    # Fix OCR/PDF line wrapping such as "gü-\nvertesinde".
    out: list[str] = []
    hyphen_end_re = re.compile(r"[-\u2010\u2011\u2212\u2013\u2014]$")
    i = 0
    while i < len(lines):
        line = lines[i]
        if hyphen_end_re.search(line.rstrip()):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and re.match(r"^[a-zçğıöşü]", lines[j].lstrip()):
                out.append(hyphen_end_re.sub("", line.rstrip()) + lines[j].lstrip())
                i = j + 1
                continue
        out.append(line)
        i += 1
    return out


def _remove_page_number_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if re.match(r"^\s*\d{1,4}\s*$", line):
            continue
        out.append(line)
    return out


def _remove_standalone_markers(lines: list[str]) -> list[str]:
    # Common in PDF/OCR: page/section markers like "I", "ı", "l".
    out: list[str] = []
    for line in lines:
        if re.match(r"^\s*[IıIl]\s*$", line):
            continue
        out.append(line)
    return out


def _remove_inline_footnote_markers(lines: list[str]) -> list[str]:
    # Remove inline footnote numbers that are surrounded by spaces and likely not meaningful.
    # Example: "yamçılarına 1 bürünmüş" -> "yamçılarına bürünmüş"
    out: list[str] = []
    for line in lines:
        line = re.sub(
            r"\b([A-Za-zÇĞİÖŞÜçğıöşü'’]+)\s+(\d{1,2})\s+([a-zçğıöşü])",
            r"\1 \3",
            line,
        )
        # Example: "ihram2" -> "ihram" (only for short trailing digits)
        line = re.sub(r"\b([A-Za-zÇĞİÖŞÜçğıöşü'’]{3,})\d{1,2}\b", r"\1", line)
        out.append(line)
    return out


def _merge_wrapped_lines(lines: list[str]) -> list[str]:
    # Merge line-wrapped PDF text into paragraph-like lines.
    merged: list[str] = []

    def should_join(a: str, b: str) -> bool:
        if not a or not b:
            return False
        a_stripped = a.rstrip()
        b_stripped = b.lstrip()
        if not a_stripped or not b_stripped:
            return False
        # Don't join if previous line clearly ends a paragraph/sentence.
        if re.search(r"[.!?…:;»]$", a_stripped):
            return False
        # Join if next line looks like continuation (starts with lowercase or quote/paren).
        if re.match(r"^[a-zçğıöşü'’\"(«]", b_stripped):
            return True
        # Also join if previous ends with comma or dash-like continuation.
        if re.search(r"[,—–-]$", a_stripped):
            return True
        return False

    def join_with_space_or_not(a: str, b: str) -> str:
        # If the last token in a is very short (common in OCR-splitting), join without a space.
        # Example: "Ay" + "dın" -> "Aydın"
        if re.search(r"\b[\wÇĞİÖŞÜçğıöşü]{1,2}$", a) and re.match(r"^[a-zçğıöşü]", b):
            return a.rstrip() + b.lstrip()
        return a.rstrip() + " " + b.strip()

    buf = ""
    for line in lines:
        if not line.strip():
            if buf:
                merged.append(buf)
                buf = ""
            merged.append("")
            continue

        if not buf:
            buf = line.strip()
            continue

        if should_join(buf, line):
            buf = join_with_space_or_not(buf, line)
        else:
            merged.append(buf)
            buf = line.strip()

    if buf:
        merged.append(buf)

    return merged


def _fix_inline_hyphen_breaks(lines: list[str]) -> list[str]:
    # Fix cases like "bulu- nanı" (hyphen kept, but wrap inserted a space)
    out: list[str] = []
    hyphen_chars = "-\u2010\u2011\u2212\u2013\u2014"
    for line in lines:
        line = re.sub(
            rf"\b([A-Za-zÇĞİÖŞÜçğıöşü]+)[{hyphen_chars}]\s+([a-zçğıöşü]+)\b",
            r"\1\2",
            line,
        )
        out.append(line)
    return out


def _fix_inword_space_splits(lines: list[str]) -> list[str]:
    # Conservative fixer for OCR splits inside words: "fesin den" -> "fesinden"
    # Only merges when the right-hand token is a short, common Turkish suffix.
    suffixes = {
        "da",
        "de",
        "dan",
        "den",
        "dır",
        "dir",
        "dur",
        "dür",
        "lar",
        "ler",
        "nin",
        "nın",
        "nun",
        "nün",
        "na",
        "ne",
        "ya",
        "ye",
        "ri",
        "rı",
        "ru",
        "rü",
        "ti",
        "tı",
        "tu",
        "tü",
        "ki",
    }

    out: list[str] = []
    for line in lines:
        def repl(m: re.Match[str]) -> str:
            left = m.group(1)
            right = m.group(2)
            if right.lower() in suffixes and left.islower() and right.islower():
                return left + right
            return m.group(0)

        # Merge only if both sides are lowercase letters (avoid names/acronyms)
        line = re.sub(r"\b([a-zçğıöşü]{3,})\s+([a-zçğıöşü]{2,4})\b", repl, line)
        out.append(line)

    return out


def _remove_isolated_running_headers(lines: list[str]) -> list[str]:
    # Remove isolated single-word headers that appear between broken lines.
    # Example encountered: blank, "Savcı.", blank, then continuation.
    out: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\.$", stripped):
            prev_blank = idx > 0 and not lines[idx - 1].strip()
            next_blank = idx + 1 < len(lines) and not lines[idx + 1].strip()
            if prev_blank and next_blank:
                continue
        out.append(line)
    return out


def _find_start_index(lines: list[str], start_regexes: list[str]) -> int:
    for rx in start_regexes:
        cre = re.compile(rx)
        for idx, line in enumerate(lines):
            if cre.search(line):
                return idx
    return 0

def _clean_degirmen_specific(lines: list[str]) -> list[str]:
    out: list[str] = []

    for line in lines:
        s = line.strip()

        # Önsöz / dipnot / yayıncı artıkları
        if s.startswith("Yazarın Önsözü"):
            continue
        if "Varlık Yayınları" in s:
            continue
        if "ed.n." in s:
            continue
        if s == "vii":
            continue

        # İçindekiler ve kısım başlıkları
        if s.startswith("Birinci Kısım"):
            continue
        if s.startswith("İkinci Kısım"):
            continue
        if s.startswith("Üçüncü Kısım"):
            continue

        # Tek başına çok kısa gürültü satırları
        if re.fullmatch(r"[·._\-—– ]{3,}", s):
            continue

        # Dipnot numarası gibi tek başına 1-2 karakterli satırlar
        if re.fullmatch(r"\d{1,2}", s):
            continue

        out.append(line)

    return out


def clean_file(text: str, filename: str) -> str:
    text = _normalize_newlines(text)
    text = _remove_pdf_artifacts(text)
    lines = text.split("\n")

    lines = _strip_gutenberg_like_boilerplate(lines)
    lines = _remove_page_headers(lines)

    start_rules: dict[str, list[str]] = {
        "Birdenbire_Sönen_Kandilin_Hikâyesi.txt": [r"^\s*Hikaye\s*:\s*$", r"^\s*Birdenbire\s+sönen\b"],
        "Kürk_Mantolu_Madonna.txt": [r"^\s*Simdiye\s+kadar\b", r"^\s*Şimdiye\s+kadar\b"],
        "İcimizdeki_Seytan.txt": [r"^\s*Öğleden\s+evvel\b"],
        "Kuyucaklı_Yusuf.txt": [r"^\s*Birinci\s+Bölüm\s*$"],
        "Degirmen.txt": [r"^\s*Hiç sen bir su değirmeninin içini dolaştın mı adaşım"],
        "Sirca_Kosk.txt": [r"^\s*Portakal\s*$"],
    }

    start_idx = _find_start_index(lines, start_rules.get(filename, [r"\S"]))

    # For "Hikaye:" we want to skip the marker line itself.
    if filename == "Birdenbire_Sönen_Kandilin_Hikâyesi.txt":
        for j in range(start_idx, min(start_idx + 3, len(lines))):
            if re.match(r"^\s*Hikaye\s*:\s*$", lines[j], flags=re.IGNORECASE):
                start_idx = j + 1
                break

    if filename == "Kuyucaklı_Yusuf.txt":
        if start_idx < len(lines) and re.match(r"^\s*Birinci\s+Bölüm\s*$", lines[start_idx], flags=re.IGNORECASE):
            start_idx += 1

    lines = lines[start_idx:]
    if filename == "Degirmen.txt":
        lines = _clean_degirmen_specific(lines)

    lines = _collapse_whitespace(lines)
    lines = _remove_page_number_lines(lines)
    lines = _remove_standalone_markers(lines)
    lines = _dehyphenate_wrapped_words(lines)
    lines = _fix_inline_hyphen_breaks(lines)
    lines = _merge_wrapped_lines(lines)
    lines = _remove_isolated_running_headers(lines)
    lines = _fix_inword_space_splits(lines)
    lines = _remove_inline_footnote_markers(lines)
    lines = _collapse_whitespace(lines)

    cleaned = "\n".join(lines).strip() + "\n"
    return cleaned


def main() -> None:
    base = Path(__file__).resolve().parent
    src_dir = base / "datasets"
    out_dir = base / "datasets_clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise SystemExit(f"Missing datasets folder: {src_dir}")

    for path in sorted(src_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_file(raw, path.name)

        # Basic sanity check: warn if file seems to contain only boilerplate.
        if len(cleaned) < 5000 and path.name in {"Kuyucaklı_Yusuf.txt"}:
            print(f"[WARN] {path.name}: cleaned text is very short ({len(cleaned)} chars). Source file may be incomplete.")

        (out_dir / path.name).write_text(cleaned, encoding="utf-8")
        print(f"[OK] {path.name} -> datasets_clean/{path.name} ({len(cleaned)} chars)")


if __name__ == "__main__":
    main()
