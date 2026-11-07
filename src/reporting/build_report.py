"""Write the assignment final report as Markdown and a compact PDF."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def build_final_report(results: pd.DataFrame, config: dict, counts: dict) -> tuple[Path, Path]:
    k = int(config["top_k"])
    table = [
        f"| Model | Precision@{k} | Recall@{k} | NDCG@{k} | Coverage | Diversity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        table.append(
            f"| {row.model} | {getattr(row, f'precision_at_{k}'):.3f} | "
            f"{getattr(row, f'recall_at_{k}'):.3f} | {getattr(row, f'ndcg_at_{k}'):.3f} | "
            f"{row.coverage:.3f} | {row.diversity:.3f} |"
        )
    valid_interactions = counts["interactions_valid"]
    report = f"""# Movie Recommendation Data Pipeline — Final Report

## Design and data management

The project creates a deterministic, synthetic English-language movie catalog
of exactly {counts['movies_valid']:,} records. Every synthetic movie is labelled
English (`language_code=en`); titles and descriptions are generated templates,
not scraped real-world movie records. It also generates {counts['users_valid']:,}
synthetic user profiles and {valid_interactions:,} valid ratings. Deliberately
malformed interaction records exercise the validation and
quarantine path. No external dataset is downloaded.

The modular workflow validates required fields, duplicates, types, 1–5 rating
bounds, timestamps, English-language catalog policy, and foreign keys.
Invalid interactions and source profile/movie rows are logged separately.
Preparation fills missing ages, normalizes fields, and holds out at least
{config['test_interactions_per_user']} latest event(s) per user; all events
tied at the cutoff timestamp are held out together. Training-only transformations
produce activity summaries, movie popularity, a user-item matrix, and reusable
user/movie/interaction features.

## Models and evaluation

The popularity baseline ranks movies by training interaction count times mean
rating. The personalized content-based model learns each user's genre
preferences from their training ratings and recommends unseen movies in
preferred genres, using popularity as a tie-breaker and cold-start fallback.
To make the synthetic offline task testable, users rate four of the five
reserved movies in their preferred genre in training and the remaining one
is held out as a future positive event.
Both recommenders rank rather than predict explicit rating values, so RMSE and
MAE do not apply. Ratings >= {config['relevance_threshold']} in the holdout
are relevant; K={k}. All test users contribute to the averaged ranking
metrics. Results from this reproducible run:

{chr(10).join(table)}

## Orchestration, tracking, and limitations

`dags/workflow.py` runs the dependency-checked stages; `dags/run_daily.sh`
supports cron scheduling. The JSON run record tracks configuration, metrics,
and artifact paths without a tracking server. CSV is the active storage
backend. PostgreSQL schema and adapter placeholders are included for a later
migration; no database connection or persistence is attempted by this run.

The controlled synthetic preferences make this suitable for testing data
management, not estimating real recommendation quality. The content model
uses genre only; cold-start quality, real-world bias, privacy, production
serving, and repeated temporal evaluation remain outside this assignment
baseline. A future study should evaluate on a licensed public dataset.
"""
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path, pdf_path = report_dir / "final_report.md", report_dir / "final_report.pdf"
    md_path.write_text(report, encoding="utf-8")
    markdown_to_pdf(report, pdf_path)
    return md_path, pdf_path


def markdown_to_pdf(markdown: str, output: Path) -> None:
    lines = []
    for source in markdown.splitlines():
        line = source.strip()
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            line = " | ".join(cell.strip() for cell in line.strip("|").split("|"))
        line = re.sub(r"^#{1,6}\s*", "", line).replace("`", "").replace("**", "")
        lines.extend(textwrap.wrap(line, width=88) if line else [""])
    pages = [lines[index:index + 48] for index in range(0, len(lines), 48)]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{3 + 2*i} 0 R' for i in range(len(pages)))}] /Count {len(pages)} >>".encode(),
    ]
    font_id = 3 + 2 * len(pages)
    for index, page_lines in enumerate(pages):
        content_id = 4 + 2 * index
        commands = ["BT", "/F1 9 Tf", "48 748 Td", "12 TL"]
        for value in page_lines:
            safe = value.encode("latin-1", errors="replace").decode("latin-1")
            safe = safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.extend([f"({safe}) Tj", "T*"])
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        objects.extend(
            [
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
                f"<< /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode(),
                b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            ]
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    output.write_bytes(pdf)
