"""Generate clean, professional PDF run guide for J26-DS-314 from the zip archive."""
import os
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "HOW_TO_RUN_PROJECT.pdf"
DESKTOP_PDF = Path(os.path.expanduser("~")) / "Desktop" / "HOW_TO_RUN_PROJECT.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0284c7"))
        
        # Header (pages after cover/first page)
        if self._pageNumber > 1:
            self.drawString(54, 755, "J26-DS-314: Project Extraction & Execution Guide")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.75)
            self.line(54, 747, 612 - 54, 747)

        # Footer (all pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 36, "Kelani River Basin Satellite Hydrology & Early Warning System")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, page_str)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.75)
        self.line(54, 48, 612 - 54, 48)
        self.restoreState()


def make_callout(title, text, style_title, style_body, bg_color="#f0f9ff", border_color="#0284c7"):
    content = [
        Paragraph(f"<b>{title}</b>", style_title),
        Spacer(1, 3),
        Paragraph(text, style_body)
    ]
    t = Table([[content]], colWidths=[504])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LINELEFT', (0,0), (0,0), 3.5, colors.HexColor(border_color)),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    return t


def make_code_box(code_text, style_code):
    formatted = "<br/>".join([line.replace(" ", "&nbsp;") for line in code_text.strip().split("\n")])
    p = Paragraph(f"<font face='Courier'>{formatted}</font>", style_code)
    t = Table([[p]], colWidths=[504])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#0f172a")),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#334155")),
    ]))
    return t


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    callout_title = ParagraphStyle(
        'CalloutTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0369a1")
    )

    callout_body = ParagraphStyle(
        'CalloutBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#38bdf8")
    )

    story = []

    # Title & Header
    story.append(Paragraph("J26-DS-314 Execution & Run Guide", title_style))
    story.append(Paragraph("Step-by-Step Instructions to Extract, Configure, and Run the Project", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # Meta Info Table
    meta_data = [
        [
            Paragraph("<b>Archive File:</b> J26-DS-314-submission.zip", body_style),
            Paragraph("<b>Target Python:</b> 3.10 - 3.12 (64-bit)", body_style)
        ],
        [
            Paragraph("<b>Primary Component:</b> C1 Satellite Hydrology", body_style),
            Paragraph("<b>OS Supported:</b> Windows / Linux / macOS", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Callout Note
    story.append(make_callout(
        "Notice Regarding Archive Optimization",
        "The archive was prepared by excluding local virtual environment binaries (.venv, saving ~1.4 GB) and sensitive Copernicus API credentials. Precomputed dataset CSVs are included so the entire evaluation and inference pipeline runs out of the box without downloading massive satellite granules.",
        callout_title, callout_body
    ))
    story.append(Spacer(1, 8))

    # Step 1
    story.append(Paragraph("1. Extract the Project Archive", h1_style))
    story.append(Paragraph("Extract <b>J26-DS-314-submission.zip</b> using Windows Explorer (Right-click &rarr; Extract All...) or run the following PowerShell command:", body_style))
    story.append(make_code_box(
        "Expand-Archive -Path \"J26-DS-314-submission.zip\" -DestinationPath \"J26-DS-314\"\ncd \"J26-DS-314\"",
        code_style
    ))
    story.append(Spacer(1, 8))

    # Step 2
    story.append(Paragraph("2. Create & Activate Python Virtual Environment", h1_style))
    story.append(Paragraph("Ensure a fresh, isolated Python environment is used to prevent dependency conflicts:", body_style))
    story.append(make_code_box(
        "# Create the virtual environment\npython -m venv .venv\n\n# Activate on Windows PowerShell\n.\\.venv\\Scripts\\Activate.ps1\n\n# (Or on Linux / macOS / Bash: source .venv/bin/activate)",
        code_style
    ))
    story.append(Paragraph("<i>Tip: If PowerShell gives an execution policy error, run: <code>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass</code> before activating.</i>", body_style))
    story.append(Spacer(1, 8))

    # Step 3
    story.append(Paragraph("3. Install Required Dependencies", h1_style))
    story.append(Paragraph("Install verified packages directly from the included requirements lock file:", body_style))
    story.append(make_code_box(
        "pip install --upgrade pip\npip install -r c1_satellite/requirements-lock.txt",
        code_style
    ))
    story.append(Spacer(1, 8))

    # Step 4
    story.append(Paragraph("4. Configure Copernicus API Credentials (Optional)", h1_style))
    story.append(Paragraph("All models and evaluations run offline with the provided dataset. If you want to download <b>new Sentinel-1 radar scenes</b> from Copernicus:", body_style))
    story.append(Paragraph("&bull; Copy the included template: <code>Copy-Item c1_satellite/secrets.env.example c1_satellite/secrets.env</code>", bullet_style))
    story.append(Paragraph("&bull; Edit <code>c1_satellite/secrets.env</code> with your Copernicus OAuth Client ID and Secret.", bullet_style))
    story.append(Spacer(1, 8))

    # Step 5
    story.append(Paragraph("5. Verify Setup via Automated Tests", h1_style))
    story.append(Paragraph("Run the pytest suite to ensure all modules, spatial processors, and model pipelines pass:", body_style))
    story.append(make_code_box(
        "python -m pytest c1_satellite/tests/ -v",
        code_style
    ))
    story.append(Spacer(1, 12))

    story.append(PageBreak())

    # Step 6
    story.append(Paragraph("6. Running the C1 Satellite Hydrology Inference API", h1_style))
    story.append(Paragraph("Start the FastAPI inference and diagnostic server locally:", body_style))
    story.append(make_code_box(
        "python c1_satellite/scripts/run_c1_api.py",
        code_style
    ))
    story.append(Paragraph("Once started, access the interactive OpenAPI documentation and test endpoints at:", body_style))
    story.append(Paragraph("&bull; <b>Interactive Swagger UI:</b> <font color='#0284c7'>http://localhost:8000/docs</font>", bullet_style))
    story.append(Paragraph("&bull; <b>Healthcheck Endpoint:</b> <font color='#0284c7'>http://localhost:8000/health</font>", bullet_style))
    story.append(Paragraph("&bull; <b>Station Prediction:</b> <code>POST /predict</code> with station ID and observation data.", bullet_style))
    story.append(Spacer(1, 10))

    # Step 7
    story.append(Paragraph("7. Running Benchmarks & Model Experiments", h1_style))
    story.append(Paragraph("Each key phase of the project has a standalone, executable script in <code>c1_satellite/scripts/</code>:", body_style))

    scripts_table_data = [
        [Paragraph("<b>Script Name</b>", body_style), Paragraph("<b>Description & Output</b>", body_style)],
        [Paragraph("<code>19_predict_station.py</code>", body_style), Paragraph("Generates flood risk predictions for Kelani basin stations.", body_style)],
        [Paragraph("<code>20_extract_sar_cnn_features.py</code>", body_style), Paragraph("Extracts CNN spatial embeddings from Sentinel-1 radar patches.", body_style)],
        [Paragraph("<code>21_benchmark_with_cnn.py</code>", body_style), Paragraph("Benchmarks baseline Ridge vs SAR-CNN enhanced models.", body_style)],
        [Paragraph("<code>22_extract_catchment_swi_features.py</code>", body_style), Paragraph("Computes catchment-wide ASCAT Soil Water Index.", body_style)],
        [Paragraph("<code>25_benchmark_multi_horizon.py</code>", body_style), Paragraph("Evaluates multi-horizon lead time forecasting (t+1 to t+3).", body_style)],
        [Paragraph("<code>26_benchmark_with_routing.py</code>", body_style), Paragraph("Executes upstream-to-downstream hydraulic routing benchmark.", body_style)],
    ]
    st = Table(scripts_table_data, colWidths=[180, 324])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(st)
    story.append(Spacer(1, 12))

    # Troubleshooting & FAQ
    story.append(Paragraph("8. Troubleshooting & FAQ", h1_style))

    faq_data = [
        [
            Paragraph("<b>Issue</b>", body_style),
            Paragraph("<b>Resolution</b>", body_style)
        ],
        [
            Paragraph("PowerShell script execution disabled", body_style),
            Paragraph("Run: <code>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass</code> in the PowerShell window.", body_style)
        ],
        [
            Paragraph("Copernicus download fails / 401 Unauthorized", body_style),
            Paragraph("Ensure <code>c1_satellite/secrets.env</code> is present with valid OAuth credentials from dataspace.copernicus.eu.", body_style)
        ],
        [
            Paragraph("CUDA / PyTorch GPU Warning", body_style),
            Paragraph("The CNN feature extractor automatically falls back to CPU if an NVIDIA GPU is not detected.", body_style)
        ]
    ]
    faq_table = Table(faq_data, colWidths=[180, 324])
    faq_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(faq_table)
    story.append(Spacer(1, 14))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated PDF at: {PDF_PATH}")

    # Copy to Desktop
    import shutil
    shutil.copy(PDF_PATH, DESKTOP_PDF)
    print(f"Copied to Desktop at: {DESKTOP_PDF}")


if __name__ == "__main__":
    build_pdf()
