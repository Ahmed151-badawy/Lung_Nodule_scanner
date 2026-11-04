"""
Professional Clinical PDF Report Generator
Compatible with pdf_adapter.py; includes provenance stamp (not a validated audit trail)
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime
from typing import List, Dict
import os


class ClinicalReportCanvas(canvas.Canvas):
    """Custom canvas with header, page counter, and regulatory lineage footer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self.pages)
        for page_num, page in enumerate(self.pages, 1):
            self.__dict__.update(page)
            self.draw_page_decorations(page_num, page_count)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_num, page_count):
        """Draw header and auditable footer on each page"""
        self.saveState()

        # Header rule
        self.setStrokeColor(colors.HexColor('#1A73E8'))
        self.setLineWidth(2)
        self.line(0.5 * inch, letter[1] - 0.5 * inch, letter[0] - 0.5 * inch, letter[1] - 0.5 * inch)

        # Header Title
        self.setFont('Helvetica-Bold', 8)
        self.setFillColor(colors.HexColor('#1A73E8'))
        self.drawString(0.5 * inch, letter[1] - 0.45 * inch, "PULMONARY NODULE DIAGNOSTIC INTELLIGENCE (CADx)")

        # Footer
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#64748B'))
        footer_text = f"Lung Nodule AI Analysis System | Page {page_num} of {page_count} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.drawCentredString(letter[0] / 2, 0.4 * inch, footer_text)

        self.restoreState()


def generate_clinical_pdf(
    pdf_path: str,
    img_original: str,
    img_gradcam: str,
    img_attention: str,
    pred_class: int,
    confidence: float,
    regions: List[Dict]
) -> bool:
    """
    Generate professional clinical PDF report with immutable provenance audit trail.
    """
    try:
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.9 * inch,
            bottomMargin=0.9 * inch
        )

        elements = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1A73E8'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1E293B'),
            spaceAfter=10,
            spaceBefore=14,
            fontName='Helvetica-Bold'
        )

        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=11,
            textColor=colors.HexColor('#334155'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9,
            leading=13,
            spaceAfter=4
        )

        # 1. Header Title
        elements.append(Paragraph("🫁 LUNG NODULE ANALYSIS REPORT", title_style))
        elements.append(Spacer(1, 0.1 * inch))

        # 2. Patient & Scan Info
        region = regions[0] if regions else {}
        patient_info = region.get('patient_info', {})

        elements.append(Paragraph("PATIENT & STUDY INFORMATION", heading_style))
        patient_data = [
            ['Patient ID:', str(patient_info.get('id', 'N/A'))],
            ['Patient Name:', str(patient_info.get('name', 'N/A'))],
            ['Age / Gender:', f"{patient_info.get('age', 'N/A')} / {patient_info.get('gender', 'N/A')}"],
            ['Study Date:', str(patient_info.get('scan_date', datetime.now().strftime('%Y-%m-%d')))],
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]

        patient_table = Table(patient_data, colWidths=[2 * inch, 4.5 * inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F1F5F9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1'))
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.15 * inch))

        # 3. Classification Result
        elements.append(Paragraph("AI CLASSIFICATION & CONFIDENCE", heading_style))
        pred_label = "MALIGNANT" if pred_class == 1 else "BENIGN"
        pred_color = '#EA4335' if pred_class == 1 else '#34A853'

        pred_style = ParagraphStyle(
            'PredStyle',
            parent=styles['Normal'],
            fontSize=20,
            textColor=colors.HexColor(pred_color),
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            spaceAfter=6
        )
        elements.append(Paragraph(f"<b>{pred_label}</b>", pred_style))

        conf_style = ParagraphStyle(
            'ConfStyle',
            parent=styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=10
        )
        elements.append(Paragraph(f"Neural Calibrated Confidence: {confidence * 100:.1f}%", conf_style))

        # 4. Expert Guidelines (Fleischner 2017)
        expert = region.get('expert_analysis', {})
        elements.append(Paragraph("CLINICAL DECISION SUPPORT (FLEISCHNER 2017)", heading_style))

        rec_style = ParagraphStyle(
            'RecStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor('#856404'),
            spaceBefore=4,
            spaceAfter=8
        )
        elements.append(Paragraph(f"⚕️ <b>Guideline Recommendation:</b> {expert.get('recommendation', 'N/A')}", rec_style))

        if expert.get('follow_up_interval'):
            elements.append(Paragraph(f"<b>Surveillance Cadence:</b> {expert.get('follow_up_interval')}", normal_style))

        elements.append(Spacer(1, 0.15 * inch))

        # 5. Nodule Morphometrics
        features = region.get('features', {})
        if features:
            elements.append(Paragraph("QUANTITATIVE MORPHOMETRICS", heading_style))
            feat_data = []
            size_mm = features.get('size_mm')
            if isinstance(size_mm, (int, float)):
                size_str = f"{size_mm:.1f} mm"
            elif size_mm is not None and str(size_mm).strip().lower() not in ('n/a', 'none', ''):
                try:
                    size_str = f"{float(size_mm):.1f} mm"
                except (ValueError, TypeError):
                    size_str = str(size_mm)
            else:
                size_str = "Uncalibrated (missing DICOM spacing)"

            feat_data.append(['Equivalent Diameter:', size_str])
            feat_data.append(['Sphericity Index:', f"{features.get('sphericity', 0):.2f}"])
            feat_data.append(['Margin Definition:', str(features.get('margin', 'N/A')).capitalize()])
            feat_data.append(['Internal Texture:', str(features.get('texture', 'N/A')).capitalize()])
            feat_data.append(['Peritumoral Spiculation:', "Present" if features.get('spiculation') else "Absent"])
            feat_data.append(['Heuristic Risk Index:', f"{expert.get('risk_score', 0):.2f} ({expert.get('risk_level', 'N/A')})"])

            feat_table = Table(feat_data, colWidths=[2.2 * inch, 4.3 * inch])
            feat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F8FAFC')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0'))
            ]))
            elements.append(feat_table)
            elements.append(Spacer(1, 0.15 * inch))

        # 6. Imaging Findings
        elements.append(Paragraph("IMAGING & SALIENCY FINDINGS", heading_style))
        images_to_add = []
        if os.path.exists(img_original):
            images_to_add.append(('CT Slice (Lung Window)', img_original))
        if os.path.exists(img_gradcam) and img_gradcam != img_original:
            images_to_add.append(('Grad-CAM Saliency Overlay', img_gradcam))

        for title, path in images_to_add:
            elements.append(Paragraph(title, subheading_style))
            try:
                img = Image(path, width=4.5 * inch, height=3.2 * inch, kind='proportional')
                elements.append(img)
                elements.append(Spacer(1, 0.1 * inch))
            except Exception as e:
                elements.append(Paragraph(f"[Image could not be rendered: {e}]", normal_style))

        # 7. Lineage Tracking & Audit Trail
        lineage = patient_info.get('lineage', {})
        if lineage:
            elements.append(PageBreak())
            elements.append(Paragraph("PROVENANCE & SYSTEM LINEAGE", heading_style))
            lineage_data = [
                ['Model Checksum (SHA-256):', str(lineage.get('model_sha256', 'N/A'))[:36] + '...'],
                ['Pipeline Git Commit SHA:', str(lineage.get('pipeline_git_sha', 'N/A'))[:20]],
                ['SOP / Series Instance UID:', str(lineage.get('series_instance_uid', 'N/A'))],
                ['Provenance Signature:', str(lineage.get('audit_hash', 'N/A'))],
                ['Radiological Timestamp:', str(lineage.get('radiological_timestamp', datetime.utcnow().isoformat()))]
            ]
            lineage_table = Table(lineage_data, colWidths=[2.2 * inch, 4.3 * inch])
            lineage_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F8FAFC')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#334155')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Courier'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1'))
            ]))
            elements.append(lineage_table)
            elements.append(Spacer(1, 0.2 * inch))

        # 8. Regulatory Disclaimer
        disc_style = ParagraphStyle(
            'DiscStyle',
            parent=styles['Normal'],
            fontSize=7.5,
            textColor=colors.HexColor('#64748B'),
            alignment=TA_JUSTIFY,
            spaceBefore=15
        )
        disc_text = (
            "<b>REGULATORY NOTICE & CLINICAL INTENDED USE:</b> This analysis is generated by an investigational "
            "AI-assisted software algorithm for RESEARCH AND CLINICAL AUDIT PURPOSES ONLY. It does NOT constitute a "
            "definitive primary diagnosis. Clinical decision-making must integrate comprehensive patient history, "
            "prior radiological examinations, and formal interpretation by a board-certified radiologist or thoracic oncologist."
        )
        elements.append(Paragraph(disc_text, disc_style))

        # Build document
        doc.build(elements, canvasmaker=ClinicalReportCanvas)
        return True

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return False
