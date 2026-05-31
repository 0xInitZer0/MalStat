"""
HtmlRenderer renders an AnalysisReport into a self-contained HTML document via Jinja2.
"""

from __future__ import annotations
from pathlib import Path
from src.reporting.models import AnalysisReport


class HtmlRenderer:
    """Renders a report into an HTML string."""

    def __init__(self, template_path: str | Path):
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        template_path = Path(template_path)
        loader = FileSystemLoader(str(template_path.parent))
        self._env = Environment(
            loader=loader,
            autoescape=select_autoescape(["html", "j2"]),
        )
        self._template = self._env.get_template(template_path.name)

    def render(self, report: AnalysisReport) -> str:
        """Return the report as an HTML string."""
        return self._template.render(
            report=report,
            group_labels=_GROUP_LABELS,
        )

    def render_to_file(self, report: AnalysisReport, output_path: str | Path) -> Path:
        """Save the HTML report to a file and return the path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = self.render(report)
        output_path.write_text(html, encoding="utf-8")
        return output_path


# Human-readable group labels for the HTML report
_GROUP_LABELS: dict[str, str] = {
    "header": "PE Header",
    "entry_point": "Entry Point",
    "sections": "Sections",
    "imports": "Imports (APIs and DLLs)",
    "strings": "Strings",
    "resources": "Resources (.rsrc)",
    "file_general": "General File Characteristics",
    "other": "Other Features",
}
