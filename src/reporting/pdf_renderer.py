"""
PdfRenderer converts HTML to PDF via weasyprint.

Install with: pip install weasyprint
On Windows, you may also need the GTK runtime:
https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
"""

from __future__ import annotations
from pathlib import Path
from src.reporting.models import AnalysisReport
from src.reporting.html_renderer import HtmlRenderer


class PdfRenderer:
    """Renders a report to PDF via weasyprint."""

    def __init__(self, html_renderer: HtmlRenderer):
        self._html_renderer = html_renderer

    def render_to_file(self, report: AnalysisReport, output_path: str | Path) -> Path:
        """Save the PDF report to a file and return the path."""
        try:
            from weasyprint import HTML
        except ImportError as exc:
            raise RuntimeError(
                "weasyprint is not installed. Run: pip install weasyprint"
            ) from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_str = self._html_renderer.render(report)
        HTML(string=html_str).write_pdf(str(output_path))
        return output_path
