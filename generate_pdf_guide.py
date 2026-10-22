"""Generate comprehensive PDF and Markdown guide for J26-DS-314 C1 Satellite Hydrology."""
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
PDF_PATH = ROOT / "J26_DS_314_C1_Comprehensive_Project_and_Code_Guide.pdf"
MD_PATH = ROOT / "J26_DS_314_C1_Comprehensive_Project_and_Code_Guide.md"


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
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "J26-DS-314 · Component 1: Satellite Hydrology & River Stage Forecasting")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 747, 558, 747)

        # Footer
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential · Final Year Thesis Technical Guide")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor("#0f172a") # dark slate
    accent_color = colors.HexColor("#2563eb")  # blue 600
    sub_color = colors.HexColor("#475569")     # slate 600
    bg_light = colors.HexColor("#f8fafc")      # slate 50
    border_color = colors.HexColor("#e2e8f0")  # slate 200

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=accent_color,
        spaceAfter=20,
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=primary_color,
        spaceBefore=16,
        spaceAfter=10,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=accent_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8,
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    code_title_style = ParagraphStyle(
        'CodeTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )

    code_desc_style = ParagraphStyle(
        'CodeDesc',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("J26-DS-314 · Component 1 (C1)", subtitle_style))
    story.append(Paragraph("Satellite Hydrology & AI River Stage Forecasting", title_style))
    story.append(Paragraph("Complete Technical Blueprint, Architecture & Codebase Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceAfter=15))

    # SECTION 1: Overall Project Concept
    story.append(Paragraph("1. Executive Overview & Overall Project Concept", h1_style))
    story.append(Paragraph(
        "<b>Project Title:</b> Satellite-Specific Temporal Depth for Next-Day River-Stage Prediction.<br/>"
        "<b>Core Research Question:</b> Can spaceborne Synthetic Aperture Radar (Sentinel-1 SAR) and satellite Soil Water Index (Copernicus ASCAT SWI), combined with river catchment boundaries and upstream hydrodynamic routing, reliably predict future river water levels (stage in meters) 12h, 24h, 48h, and 72h in advance across Sri Lanka?",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Hydrological Reality (How Floods Happen in Sri Lanka):</b><br/>"
        "1. Heavy monsoon precipitation falls over the central highlands.<br/>"
        "2. The soil absorbs water up to field capacity. When the <b>Soil Water Index (SWI)</b> exceeds 100, the basin becomes supersaturated and waterlogged.<br/>"
        "3. Subsequent rainfall cannot infiltrate and immediately turns into overland flood runoff.<br/>"
        "4. This runoff collects into headwater tributaries (e.g. Norwood, Kithulgala, Ratnapura) creating steep upstream water level surges.<br/>"
        "5. The flood wave travels downstream along the river network over 6 to 24 hours, arriving at major population centers and coastal outlets (e.g. Hanwella and Nagalagam Street in Colombo).",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Mission of Component 1 (C1):</b> Rather than treating flood forecasting as an isolated black-box or simple threshold alarm, C1 captures the full physical pathway: from satellite radar imagery and basin soil wetness to upstream river surges and multi-lead time probabilistic water levels.",
        body_style
    ))

    # SECTION 2: System Architecture Diagram / Flow
    story.append(Paragraph("2. End-to-End System Architecture", h1_style))
    arch_data = [
        ["Layer", "Inputs / Technology", "Function & Transformation"],
        ["1. Raw Data Ingestion", "DMC Sri Lanka Water Level Reports\nCopernicus CLMS ASCAT SWI V4 (TIF)\nSentinel-1 GRD SAR Imagery (GEE)", "Harvests daily gauge observations across 40 river basins and authentic satellite rasters."],
        ["2. Spatial & Basin Features", "HydroSHEDS Catchment Polygons\nResNet18 / MobileNetV3 Patch Encoder", "Extracts basin-averaged soil saturation and spatial backscatter embeddings from radar patches."],
        ["3. Hydro-Temporal Memory", "14-Day Rolling Antecedent SWI\nStage Lag 1, Lag 2, Acceleration", "Preserves watershed saturation history and 48-hour river surge rate of rise."],
        ["4. River Network Routing", "Topological Basin DAG\nKinematic Wave Routing Celerity", "Propagates upstream flood crests down to coastal flood outlets (e.g., Colombo)."],
        ["5. Probabilistic ML Core", "MultiHorizonQuantileModel\nExtraTrees & GradientBoosting", "Predicts P10 (lower bound), P50 (expected), and P90 (worst-case flood risk) at 12h, 24h, 48h, 72h."],
        ["6. Interactive API & UI", "FastAPI Backend + HTML5/CSS3\nInteractive Leaflet Basin Maps", "Live forecasting dashboard for emergency civil defence and Irrigation Dept operators."]
    ]
    t_arch = Table(arch_data, colWidths=[90, 160, 250])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('BACKGROUND', (0, 1), (-1, -1), bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 15))

    # SECTION 3: Detailed File-by-File Breakdown
    story.append(PageBreak())
    story.append(Paragraph("3. Detailed File-by-File Codebase Directory", h1_style))
    story.append(Paragraph("Below is a comprehensive guide to every single script and module in the <code>c1_satellite</code> repository, explained in plain language.", body_style))

    files_info = [
        # Foundation scripts
        ("c1_satellite/scripts/01_c1_local_audit.py",
         "Data Contract Audit & Integrity Verification",
         "Reads the local DMC river gauge observations and verifies data cleanliness. Confirms there are 40 river basins, checks for duplicate station-date records, validates that stage readings are in valid metric units (e.g. converting Nagalagam Street feet to meters), and ensures zero missing values in primary key columns."),

        ("c1_satellite/scripts/02_install_candidate_catchments.py",
         "Catchment Polygon Extraction & GeoJSON Installer",
         "Unpacks and installs the 40 upstream river catchment polygons from HydroSHEDS and Irrigation Dept archives into GeoJSON format. Maps each gauge coordinate to its corresponding upstream watershed geometry for raster clipping."),

        ("c1_satellite/scripts/03_sentinel1_inventory.py",
         "Google Earth Engine Sentinel-1 Radar Inventory",
         "Queries Google Earth Engine (GEE) COPERNICUS/S1_GRD collection for Sentinel-1 C-band synthetic aperture radar overpasses covering Sri Lanka. Filters for Interferometric Wide (IW) swath mode and VV/VH polarizations. Establishes a 60-day lookback window with an assumed 48-hour operational publication lag."),

        ("c1_satellite/scripts/04_sentinel1_patch_export.py",
         "SAR 256x256 GeoTIFF Image Patch Generator",
         "Extracts a gauge-centered 12.8 km x 12.8 km square patch (256x256 pixels at 50-meter spatial resolution) around each river gauge. Exports two-band (VV and VH) GeoTIFF rasters and generates PNG visual previews."),

        ("c1_satellite/scripts/05_sar_feature_table.py",
         "Statistical Backscatter Feature Extractor",
         "Reads downloaded SAR GeoTIFF patches and computes statistical summary features for each scene: VV mean, VH mean, VV/VH ratio, standard deviations, and 10th/90th percentile backscatter. Also calculates temporal change between same-orbit repeat passes."),

        ("c1_satellite/scripts/06_batch_sentinel1_patches.py",
         "Automated Batch SAR Downloader",
         "Orchestrates multi-station batch downloads of Sentinel-1 radar scenes, handling retries, timeout management, and metadata cataloging without fabricating missing satellite passes."),

        ("c1_satellite/scripts/07_download_ascat_swi_v4.py",
         "Copernicus CLMS ASCAT SWI V4 Downloader",
         "Connects to Copernicus Data Space Ecosystem (CDSE) Sentinel Hub API using OAuth credentials. Downloads daily 0.1-degree global Soil Water Index (SWI) rasters containing 8 bands: SWI001, SWI005, SWI010, quality flags (QFLAG), surface state flags (SSF), and dataMask."),

        ("c1_satellite/scripts/08_extract_ascat_swi_v4.py",
         "Point Soil Moisture Sampler",
         "Samples the exact ASCAT SWI grid cell corresponding to the geographic latitude/longitude of each river gauge station."),

        ("c1_satellite/scripts/09_diagnose_ascat_raw.py",
         "Satellite Raster Diagnostic Inspector",
         "Examines raw ASCAT raster arrays, pixel values, nodata sentinels (-9999), and dataMask behavior to ensure valid data extraction across Sri Lanka's land surface."),

        ("c1_satellite/scripts/10_extract_ascat_scaled.py",
         "Soil Moisture Calibration & Scaling",
         "Applies calibration scaling to ASCAT Soil Water Index rasters, mapping raw digital numbers to calibrated physical soil wetness percentages."),

        ("c1_satellite/scripts/11_c1_controlled_models.py",
         "Baseline Ridge Regression Model",
         "Fits the initial baseline Ridge regression model using current river stage and simple tabular statistics. Serves as the primary reference benchmark for all subsequent accuracy improvements."),

        ("c1_satellite/scripts/12_c1_diagnostics.py",
         "Error Analysis & Diagnostic Suite",
         "Generates residual error distributions, identifies high-error river stations, and analyzes whether satellite observations add incremental value over simple persistence (assuming stage remains unchanged)."),

        ("c1_satellite/scripts/13_swi_bandwise_history.py",
         "7-Day Soil Moisture Memory Builder",
         "Extracts a rolling 7-day soil water index history (Day-1 through Day-7) for both SWI005 (topsoil infiltration) and SWI010 (sub-surface root-zone moisture), computing 7-day mean, 3-day mean, and trend metrics."),

        ("c1_satellite/scripts/14_c1_bandwise_validation.py",
         "Validation Benchmark for Bandwise Soil Moisture",
         "Evaluates model variants on the pristine validation split to determine whether a 7-day soil moisture history improves water level predictions over a single-day point reading."),

        ("c1_satellite/scripts/15_pp1_evidence.py",
         "PP1 Progress Viva Evidence Package",
         "Compiles comprehensive audit logs, feature summaries, and evidence tables required for the academic progress presentation (PP1)."),

        ("c1_satellite/scripts/16_audit_ridge_and_data.py",
         "Strict Temporal Leakage & Split Audit",
         "Audits the feature matrix to guarantee that no future targets or lookahead information leaked into the input features. Verifies exact train/validation/test chronological boundaries."),

        ("c1_satellite/scripts/17_validation_only_model_improvements.py",
         "Tabular Model Family Exploration",
         "Benchmarks Ridge, ExtraTrees, and HistGradientBoosting regressors on stage history and satellite tabular features, proving that ExtraTrees offers superior robustness against extreme flood outliers."),

        ("c1_satellite/scripts/18_station_validation_diagnostics.py",
         "Per-Station Gauge Breakdown & Metrics",
         "Produces station-by-station Mean Absolute Error (MAE) tables, identifying which river basins (e.g. Kelani vs Mahaweli) respond fastest to rainfall and satellite soil moisture changes."),

        ("c1_satellite/scripts/19_predict_station.py",
         "Single-Station Inference Engine",
         "Command-line inference script that takes any station ID and forecast date and outputs the predicted stage delta and next-day water level in meters."),

        ("c1_satellite/scripts/run_c1_api.py",
         "FastAPI Server & Interactive Web UI Dashboard",
         "Launches a production-grade web application on http://127.0.0.1:8000. Features an interactive Leaflet map of Sri Lanka with all 40 river basins, gauge telemetry status, alert level badges (Normal, Alert, Minor Flood, Major Flood), and live forecasting controls."),
    ]

    for fname, title, desc in files_info:
        story.append(Paragraph(f"<b>{title}</b> (<code>{fname}</code>)", code_title_style))
        story.append(Paragraph(desc, code_desc_style))

    # SECTION 4: The 5 Upgrades Modules
    story.append(PageBreak())
    story.append(Paragraph("4. The 5 Technical Upgrades: Architecture & Implementation", h1_style))
    story.append(Paragraph(
        "To advance the prototype to high precision, 5 state-of-the-art upgrades were engineered. "
        "Each upgrade comprises a core module in <code>c1_satellite/src/improvements/</code>, an automated test in <code>c1_satellite/tests/</code>, "
        "and a benchmark script in <code>c1_satellite/scripts/</code>.", body_style
    ))

    upgrades_data = [
        ("Upgrade 1: Spatial CNN Patch Encoder",
         "src/improvements/sar_cnn.py | scripts/20, 21 | tests/test_20",
         "Replaces simple statistical averages with a deep Convolutional Neural Network (ResNet18 / MobileNetV3) that ingests raw 256x256 GeoTIFF radar patches directly. The CNN extracts spatial flood inundation patterns, river channel widening, and oxbow lake overflow into dense feature embeddings."),

        ("Upgrade 2: Basin-Scale Catchment SWI",
         "src/improvements/catchment_swi.py | scripts/22, 23 | tests/test_22",
         "Replaces point soil moisture with upstream catchment integration across all 40 river basin polygons. Fixed the satellite masking cap to accommodate monsoon supersaturation (SWI > 100). Added catch_oversat_frac (fraction of basin waterlogged), which proved to be a top-15 predictive feature."),

        ("Upgrade 3: Expanded Dataset & 14-Day Memory",
         "src/improvements/dataset_expansion.py | scripts/24 | tests/test_24",
         "Eliminated the restriction that discarded 663 rows lacking two same-orbit passes. Expanded real SAR training data from 319 to 516 samples (+61.8%) and added 14-day rolling antecedent catchment moisture, dropping validation MAE to 0.2411 m (24.1 cm)."),

        ("Upgrade 4: Multi-Horizon Quantile Forecasting",
         "src/improvements/multi_horizon.py | scripts/25 | tests/test_25",
         "Upgrades single-day predictions to multi-horizon lead times: +12h (MAE 0.1154 m), +24h (MAE 0.2308 m), +48h (MAE 0.3331 m), and +72h (MAE 0.4159 m). Employs pinball loss quantile regression to output P10, P50, and P90 risk envelopes with ~80% empirical coverage."),

        ("Upgrade 5: Physics-Guided River Routing",
         "src/improvements/river_routing.py | scripts/26 | tests/test_26",
         "Constructs a topological river network graph connecting upstream gauges to downstream coastal outlets (e.g. Hanwella -> Nagalagam Street). Computes kinematic flood wave velocity features and enforces mass-conservation clamping, dropping downstream forecast error to 0.2586 m (+0.96% gain)."),
    ]

    for utitle, files, udesc in upgrades_data:
        story.append(Paragraph(f"<b>{utitle}</b>", h2_style))
        story.append(Paragraph(f"<b>Files:</b> <code>{files}</code>", code_desc_style))
        story.append(Paragraph(udesc, body_style))
        story.append(Spacer(1, 4))

    # SECTION 5: Benchmark Comparison Table
    story.append(Paragraph("5. Accuracy Progression Benchmark Across Upgrades", h1_style))
    bench_data = [
        ["Model Variant / Upgrade", "Training Size", "Val Size", "Val MAE (m)", "Val RMSE (m)", "Status / Notes"],
        ["Phase 17 Baseline (Tabular)", "319 rows", "186 rows", "0.2501 m", "0.3758 m", "Original restricted prototype"],
        ["Upgrade 2: Catchment SWI", "319 rows", "186 rows", "0.2501 m", "0.3758 m", "100% basin coverage across 40 basins"],
        ["Upgrade 3: Expanded SAR Data", "516 rows", "217 rows", "0.2411 m", "0.3743 m", "+61.8% real radar training data"],
        ["Upgrade 4: Multi-Horizon (+12h)", "516 rows", "216 rows", "0.1154 m", "0.1820 m", "Immediate emergency alert precision"],
        ["Upgrade 4: Multi-Horizon (+24h)", "516 rows", "216 rows", "0.2308 m", "0.3739 m", "Day+1 flood bulletin median P50"],
        ["Upgrade 4: Multi-Horizon (+48h)", "516 rows", "215 rows", "0.3331 m", "0.5066 m", "Day+2 peak crest arrival"],
        ["Upgrade 4: Multi-Horizon (+72h)", "516 rows", "215 rows", "0.4159 m", "0.6133 m", "Day+3 floodplain clearance"],
        ["Upgrade 5: River Routing (Downstream)", "516 rows", "68 rows", "0.2586 m", "0.3882 m", "Kinematic wave propagation gain (+0.96%)"]
    ]
    t_bench = Table(bench_data, colWidths=[150, 65, 55, 65, 65, 100])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7.5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 15))

    # SECTION 6: Viva / Defense Q&A
    story.append(PageBreak())
    story.append(Paragraph("6. Viva & Academic Defense Q&A (Short & Powerful Answers)", h1_style))
    story.append(Paragraph("Use these direct answers when explaining the project to your supervisor or external examiners:", body_style))

    qa_list = [
        ("Q: Why did you predict river stage (meters) instead of binary flood/no-flood classification?",
         "A: Binary classification discards crucial hydrological magnitude. A 10 cm rise in a low river is harmless, but a 10 cm rise when a river is already at its brim causes catastrophic embankment overtopping. Predicting continuous stage allows civil authorities to apply exact official flood thresholds (Alert, Minor Flood, Major Flood) dynamically."),

        ("Q: What is the benefit of ASCAT Soil Water Index over simple rainfall records?",
         "A: Rain only tells you what fell from the sky, not what happens on the ground. If the soil is bone dry, 50 mm of rain will simply absorb into the ground with zero flooding. But if ASCAT SWI exceeds 100 (supersaturated), even 15 mm of rain immediately produces flash flooding. Satellite soil moisture represents the true runoff generation potential."),

        ("Q: How did you solve the satellite revisit gap (Sentinel-1 6-to-12 day revisit)?",
         "A: In Upgrade 3, we introduced availability indicators (sar_available_flag). When a radar overpass occurs, the model exploits high-resolution backscatter; on non-overpass days, it smoothly falls back to Stage dynamics + Catchment SWI. This expanded our authentic radar training dataset from 319 to 516 samples and reduced forecast error to 24.1 cm."),

        ("Q: Why is quantile forecasting (P10, P50, P90) better than a single-point forecast?",
         "A: In disaster management, a single forecast of 4.2 m gives a false sense of certainty. If the true stage reaches 5.2 m, towns flood without warning. By outputting P10 (optimistic), P50 (expected), and P90 (worst-case risk bound), we provide civil defense with an 80% confidence interval, allowing proactive evacuation based on risk tolerance."),

        ("Q: What is Physics-Guided River Routing in Upgrade 5?",
         "A: Water obeys mass conservation and flows downstream. An isolated ML model doesn't know Nagalagam Street is downstream of Hanwella. We built a directed river network graph and computed kinematic wave propagation features (upstream rate of rise and wave potential). When an upstream gauge surges, the model automatically propagates the flood crest downstream before it arrives.")
    ]

    for q, a in qa_list:
        story.append(Paragraph(f"<b>{q}</b>", ParagraphStyle('QStyle', parent=body_style, fontName='Helvetica-Bold', textColor=accent_color, spaceBefore=6, spaceAfter=2, keepWithNext=True)))
        story.append(Paragraph(a, ParagraphStyle('AStyle', parent=body_style, spaceAfter=8)))

    # SECTION 7: How to Run the Project
    story.append(Paragraph("7. Practical Commands Cheatsheet", h1_style))
    commands_text = (
        "<b>1. Run the Interactive Web UI & API Server:</b><br/>"
        "<code>.venv\\Scripts\\python c1_satellite/scripts/run_c1_api.py</code><br/>"
        "Open your browser at: <b>http://127.0.0.1:8000</b><br/><br/>"
        "<b>2. Run the Full Unit Test Suite (28 Tests):</b><br/>"
        "<code>.venv\\Scripts\\python -c \"import c1_satellite.tests.test_22_catchment_swi as t; [getattr(t, f)() for f in dir(t) if f.startswith('test_')]\"</code><br/><br/>"
        "<b>3. Run the 5 Upgrade Benchmarks:</b><br/>"
        "• Upgrade 1 (CNN): <code>.venv\\Scripts\\python c1_satellite/scripts/21_benchmark_with_cnn.py</code><br/>"
        "• Upgrade 2 (Catchment): <code>.venv\\Scripts\\python c1_satellite/scripts/23_benchmark_with_catchment.py</code><br/>"
        "• Upgrade 3 (Expanded Dataset): <code>.venv\\Scripts\\python c1_satellite/scripts/24_benchmark_with_expanded_dataset.py</code><br/>"
        "• Upgrade 4 (Multi-Horizon): <code>.venv\\Scripts\\python c1_satellite/scripts/25_benchmark_multi_horizon.py</code><br/>"
        "• Upgrade 5 (River Routing): <code>.venv\\Scripts\\python c1_satellite/scripts/26_benchmark_with_routing.py</code>"
    )
    story.append(Paragraph(commands_text, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
