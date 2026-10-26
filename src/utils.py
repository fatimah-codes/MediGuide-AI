"""Helpers: safe JSON parsing, safety pre-screen, hashing and report builders."""
import hashlib
import html
import json
import re
from datetime import datetime

from . import config as cfg

EXPECTED_KEYS = [
    "summary", "possible_conditions", "urgency_level",
    "recommended_next_steps", "questions_for_doctor", "warning_signs",
]


# ---------------------------------------------------------------------------
# Safe JSON parsing - bad model output must NEVER crash the app
# ---------------------------------------------------------------------------
def clean_json_text(raw: str) -> str:
    """Strip ```json fences and any text around the outermost { ... } object."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return text.strip()


def _str_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def normalise_assessment(obj: dict) -> dict:
    """Make sure every expected key exists with the right type."""
    conditions = []
    raw_conditions = obj.get("possible_conditions", [])
    if isinstance(raw_conditions, list):
        for item in raw_conditions:
            if isinstance(item, dict):
                name, reason = str(item.get("name", "")).strip(), str(item.get("reason", "")).strip()
            else:
                name, reason = str(item).strip(), ""
            if name:
                conditions.append({"name": name, "reason": reason})

    level = str(obj.get("urgency_level", "")).strip().upper()
    defaulted = level not in cfg.URGENCY_LEVELS
    if defaulted:
        level = "MEDIUM"  # unknown level -> cautious middle value

    return {
        "summary": str(obj.get("summary", "")).strip(),
        "possible_conditions": conditions,
        "urgency_level": level,
        "recommended_next_steps": _str_list(obj.get("recommended_next_steps")),
        "questions_for_doctor": _str_list(obj.get("questions_for_doctor")),
        "warning_signs": _str_list(obj.get("warning_signs")),
        "urgency_defaulted": defaulted,
    }


def parse_assessment(raw: str) -> tuple[dict | None, str | None]:
    """Return (data, None) on success or (None, error_message) on failure."""
    text = clean_json_text(raw)
    if not text:
        return None, "The model returned an empty response."
    obj, error = None, "Invalid JSON."
    for candidate in (text, re.sub(r",\s*([}\]])", r"\1", text)):  # 2nd try: remove trailing commas
        try:
            obj = json.loads(candidate)
            error = None
            break
        except json.JSONDecodeError as exc:
            error = f"Invalid JSON: {exc}"
    if error:
        return None, error
    if not isinstance(obj, dict) or not any(k in obj for k in ("summary", "urgency_level")):
        return None, "JSON did not contain the expected fields."
    return normalise_assessment(obj), None


# ---------------------------------------------------------------------------
# Safety pre-screen
# ---------------------------------------------------------------------------
def detect_red_flags(text: str) -> list[str]:
    """Return messages for any emergency-style symptom combinations found."""
    blob = text.lower()
    return [msg for groups, msg in cfg.RED_FLAG_RULES
            if all(any(k in blob for k in group) for group in groups)]


def detect_crisis(text: str) -> bool:
    blob = text.lower()
    return any(k in blob for k in cfg.CRISIS_KEYWORDS)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def format_symptoms(selected: list[str], free_text: str) -> str:
    parts = list(selected)
    if free_text.strip():
        parts.append(free_text.strip())
    return ", ".join(parts)


def inputs_hash(*items) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()[:24]


def chunk_text(text: str, words_per_chunk: int = 3) -> list[str]:
    """Split text into small pieces so a cached narrative can still 'type' on screen."""
    tokens = re.findall(r"\S+\s*", text)
    return ["".join(tokens[i:i + words_per_chunk]) for i in range(0, len(tokens), words_per_chunk)]


def esc(value) -> str:
    return html.escape(str(value))


def html_block(markup: str) -> str:
    """Collapse indented HTML to one line so Markdown never treats it as a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# Reports (HTML for every language, PDF for Latin-script languages)
# ---------------------------------------------------------------------------
def build_html_report(entry: dict) -> str:
    data, inp = entry["data"], entry["prompt_inputs"]
    style = cfg.URGENCY_STYLE[data["urgency_level"]]
    rtl = entry["language"] in cfg.RTL_LANGUAGES

    def ul(items):
        return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>" if items else "<p>-</p>"

    conds = "".join(
        f"<li><b>{esc(c['name'])}</b> - {esc(c['reason'])}</li>" for c in data["possible_conditions"]
    )
    narrative = esc(entry.get("narrative") or "").replace("\n", "<br>")
    return f"""<!doctype html><html lang="en" dir="{'rtl' if rtl else 'ltr'}"><head><meta charset="utf-8">
<title>MediGuide AI Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1e1b4b}}
.urg{{background:{style['c1']};color:#fff;padding:1rem 1.4rem;border-radius:14px;font-size:1.3rem}}
.box{{border:2px solid #ef4444;background:#fef2f2;padding:.8rem 1rem;border-radius:10px;margin:1rem 0}}
h2{{border-bottom:2px solid #ddd;padding-bottom:.2rem}}</style></head><body>
<h1>MediGuide AI - Guidance Report</h1><p>Generated {esc(entry['time'])}</p>
<div class="box"><b>Medical disclaimer:</b> {esc(cfg.DISCLAIMER_LONG)}</div>
<div class="urg">{style['emoji']} Urgency level: <b>{esc(data['urgency_level'])}</b></div>
<h2>Patient details</h2><p>Age {esc(inp['age'])}, {esc(inp['gender'])}<br>Symptoms: {esc(inp['symptoms'])}<br>
Duration: {esc(inp['duration'])} | Severity: {esc(inp['severity'])}/10<br>Conditions: {esc(inp['conditions'])}<br>
Medications: {esc(inp['medications'])}<br>Notes: {esc(inp['notes'])}</p>
<h2>Summary</h2><p>{esc(data['summary'])}</p>
<h2>Possible conditions (education only, not a diagnosis)</h2><ul>{conds}</ul>
<h2>Recommended next steps</h2>{ul(data['recommended_next_steps'])}
<h2>Questions to ask a healthcare professional</h2>{ul(data['questions_for_doctor'])}
<h2>Warning signs needing immediate attention</h2>{ul(data['warning_signs'])}
<h2>AI explanation</h2><p>{narrative}</p>
<div class="box"><b>Reminder:</b> This report is educational and is not a diagnosis. Consult a qualified healthcare professional.</div>
</body></html>"""


_PDF_REPLACE = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    "•": "-", "…": "...", " ": " ", "**": "", "→": "->",
}


def _pdf_text(text) -> str:
    text = str(text)
    for old, new in _PDF_REPLACE.items():
        text = text.replace(old, new)
    text = re.sub(r"(\S{60})(?=\S)", r"\1 ", text)  # break very long tokens
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf_report(entry: dict) -> bytes:
    """Create a simple PDF report with fpdf2 (Latin-script languages only)."""
    from fpdf import FPDF

    data, inp = entry["data"], entry["prompt_inputs"]
    style = cfg.URGENCY_STYLE[data["urgency_level"]]

    class Report(FPDF):
        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, f"MediGuide AI - educational only, not medical advice - page {self.page_no()}", align="C")

    pdf = Report()
    pdf.set_auto_page_break(True, 18)
    pdf.add_page()

    def write(text, size=10, bold=False, color=(30, 27, 75), gap=1.5):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_text_color(*color)
        pdf.multi_cell(0, 5.5, _pdf_text(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)

    def heading(text):
        pdf.ln(2)
        write(text, 13, True, (91, 33, 182), 1)

    write("MediGuide AI - Guidance Report", 18, True, (91, 33, 182), 1)
    write(f"Generated {entry['time']}", 9, color=(110, 110, 110), gap=3)

    pdf.set_fill_color(254, 226, 226)
    pdf.set_text_color(153, 27, 27)
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(0, 5.5, _pdf_text("MEDICAL DISCLAIMER: " + cfg.DISCLAIMER_LONG), fill=True,
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    h = style["c1"].lstrip("#")
    pdf.set_fill_color(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 11, _pdf_text(f"  Urgency level: {data['urgency_level']}"), fill=True,
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    write(style["advice"], 10, color=(60, 60, 60))

    heading("Patient details")
    write(f"Age: {inp['age']} | Gender: {inp['gender']}\nSymptoms: {inp['symptoms']}\n"
          f"Duration: {inp['duration']} | Severity: {inp['severity']}/10\n"
          f"Conditions: {inp['conditions']}\nMedications: {inp['medications']}\nNotes: {inp['notes']}")
    heading("Summary")
    write(data["summary"])
    heading("Possible conditions (education only - not a diagnosis)")
    for c in data["possible_conditions"]:
        write(f"- {c['name']}: {c['reason']}")
    heading("Recommended next steps")
    for i, s in enumerate(data["recommended_next_steps"], 1):
        write(f"{i}. {s}")
    heading("Questions to ask a healthcare professional")
    for q in data["questions_for_doctor"]:
        write(f"- {q}")
    heading("Warning signs needing immediate attention")
    for w in data["warning_signs"]:
        write(f"- {w}")
    if entry.get("narrative"):
        heading("AI explanation")
        write(entry["narrative"])
    return bytes(pdf.output())


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
