"""
Incident report PDFs (reportlab). One report per alert, plus a shift summary report.
"""

import os
import time
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SEV_COLOR = {"CRITICAL": colors.HexColor("#c0392b"), "HIGH": colors.HexColor("#e67e22"),
             "MEDIUM": colors.HexColor("#f1c40f"), "LOW": colors.HexColor("#7f8c8d")}


def _fmt(ts: Optional[float]) -> str:
    return time.strftime("%d %b %Y  %H:%M:%S", time.localtime(ts)) if ts else "-"


def _esc(s) -> str:
    return str(s if s is not None else "-").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _table(rows, widths):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def build_incident_report(alert: Dict, out_path: str, snapshot_dir: str = "data/snapshots") -> str:
    """Create a one-incident PDF. ``alert`` is a row from AlertStore."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Title"], fontSize=18, textColor=colors.HexColor("#1b2a41"))
    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"IBVAP Incident {alert['id']}", author="IBVAP")
    sev = alert.get("severity", "LOW")
    story = [Paragraph("INCIDENT REPORT", h),
             Paragraph("IBVAP - Intelligent Border Video Analytics Platform | SIH26187", ss["Normal"]), Spacer(1, 6 * mm)]
    banner = Table([[f"SEVERITY: {sev}   |   {alert.get('type', '')}"]], colWidths=[174 * mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SEV_COLOR.get(sev, colors.grey)),
                                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white), ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [banner, Spacer(1, 4 * mm)]
    extra = alert.get("extra") or {}
    rows = [["Incident ID", _esc(alert["id"])], ["Detected at", _fmt(alert.get("created"))],
            ["Camera / Sector", f"{_esc(alert.get('camera_id'))}  /  {_esc(alert.get('sector'))}"],
            ["Event", Paragraph(_esc(alert.get("message")), ss["BodyText"])],
            ["Subject", f"{_esc(alert.get('class_name'))}  (track #{_esc(alert.get('track_id'))})"],
            ["Zone / Line", _esc(alert.get("zone"))], ["Subtype", _esc(alert.get("subtype"))],
            ["Status", "ACKNOWLEDGED - " + _esc(alert.get("ack_note")) if alert.get("acknowledged") else "OPEN - awaiting operator action"]]
    for k in ("plate", "plate_type", "reason", "poi_id", "similarity", "count"):
        if k in extra:
            rows.append([k.replace("_", " ").title(), _esc(extra[k])])
    story += [_table(rows, [38 * mm, 136 * mm]), Spacer(1, 5 * mm)]
    snap = alert.get("snapshot")
    path = os.path.join(snapshot_dir, snap) if snap else None
    if path and os.path.exists(path):
        story += [Paragraph("Evidence frame", ss["Heading3"]), Image(path, width=174 * mm, height=174 * mm * 0.5625, kind="proportional")]
    story += [Spacer(1, 6 * mm), Paragraph(
        "<i>Automatically generated. AI detections are decision support and must be verified by an operator "
        "before action is taken.</i>", ss["Italic"])]
    doc.build(story)
    return out_path


def build_summary_report(alerts: List[Dict], stats: Dict, out_path: str, title: str = "Shift Summary Report") -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ss = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, title=title)
    story = [Paragraph(title, ss["Title"]), Paragraph(f"Generated {_fmt(time.time())}", ss["Normal"]), Spacer(1, 5 * mm)]
    sev_rows = [["Severity", "Count"]] + [[k, v] for k, v in sorted(stats.get("by_severity", {}).items())]
    typ_rows = [["Alert type", "Count"]] + [[k, v] for k, v in sorted(stats.get("by_type", {}).items())]
    story += [Paragraph(f"Total alerts: <b>{stats.get('total', 0)}</b>", ss["Heading3"]), Table(sev_rows), Spacer(1, 4 * mm), Table(typ_rows), Spacer(1, 6 * mm),
              Paragraph("Latest alerts", ss["Heading3"])]
    rows = [["Time", "Cam", "Sev", "Type", "Message"]]
    for a in alerts[:40]:
        rows.append([_fmt(a.get("created")), _esc(a.get("camera_id")), _esc(a.get("severity")), _esc(a.get("type")),
                     Paragraph(_esc(a.get("message")), ParagraphStyle("s", fontSize=7))])
    t = Table(rows, colWidths=[34 * mm, 18 * mm, 18 * mm, 38 * mm, 70 * mm], repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2a41")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)
    doc.build(story)
    return out_path
