import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_writeup_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "submission_writeup.pdf")
    pdf_path = os.path.abspath(pdf_path)
    
    # 0.5-inch margins to fit everything neatly on a single page
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Define styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=1, # Center
        spaceAfter=12
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#475569'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0284c7'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=5
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#1e293b'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    story = []
    
    # Header
    story.append(Paragraph("Hackathon Submission: Catch the Invisible AI Cheater", title_style))
    story.append(Paragraph("Track: Interview Integrity  |  Project: IntegrityGuard AI  |  Author: AK Srivastava", meta_style))
    
    # Section 1: Project Overview
    story.append(Paragraph("1. Project Overview & Architecture", heading_style))
    story.append(Paragraph(
        "IntegrityGuard AI is a zero-install browser-native prototype designed to detect candidates secretly using "
        "real-time AI interview copilots (such as Cluely, Parakeet, or screen teleprompters). Traditional proctoring "
        "software is easily bypassed by these tools because they run as transparent system overlays or on separate devices. "
        "Our system operates entirely within the interview tab on localhost, simulating a real-time mock AI interviewer "
        "using browser speech APIs while passively streaming behavioral telemetry to a separate reviewer console.",
        body_style
    ))
    
    # Section 2: How It Detects (Multi-Modal Signals)
    story.append(Paragraph("2. Core Detection & Telemetry Channels", heading_style))
    story.append(Paragraph(
        "Rather than relying on invasive screen recording or system scans, we monitor candidate behavior through four core channels:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Speech Latency (Timing)</b>: Measures the exact time gap between the end of the question voice playback "
        "and the candidate's first spoken word. Cheating tools create an inevitable 5-8+ second delay to generate answers, "
        "whereas natural speech begins within 1.5-2.5 seconds.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Calibrated Iris Deviation (Gaze)</b>: Uses client-side MediaPipe FaceMesh to track iris coordinate movement "
        "relative to the eye socket corners. On interview start, the system runs a 50-frame baseline calibration. This allows "
        "natural monitor-reading shifts, while capturing large vertical or horizontal deflections (like reading from a phone or second display).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Tab Focus Tracking (Blurs)</b>: Registers events when the candidate switches tabs or clicks outside "
        "the interview window (e.g. to copy/paste text or search coordinates).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>AI-Likeness Phrasing Scorer</b>: Evaluates verbal transcript structure (via Grok API or offline keyword heuristics) "
        "for AI-generated written styles (e.g., transitional lists, lack of speech fillers). Short transcripts (&lt;15 words) "
        "are bypassed to prevent false flags.",
        bullet_style
    ))
    
    # Section 3: Risk Fusion & Reviewer Experience
    story.append(Paragraph("3. Multi-Modal Risk Fusion & Reviewer Experience", heading_style))
    story.append(Paragraph(
        "To satisfy hackathon requirements, the system <b>never auto-rejects</b> candidates. Telemetry is fused by a risk aggregator "
        "into a composite score and classification (Clean, Suspicious, or High Risk). High-severity overrides are implemented "
        "to trigger alerts immediately if a single signal is definitive (e.g., Pause &gt; 10s, Gaze &gt; 50%, or Tab Blur &gt;= 1). "
        "The reviewer dashboard renders raw telemetry numbers, transcript snippets, and AI rationale logs in real time, "
        "providing a transparent audit trail.",
        body_style
    ))
    
    # Section 4: Honest Coverage & Technical Limits
    story.append(Paragraph("4. Honest Coverage & System Limitations", heading_style))
    story.append(Paragraph(
        "IntegrityGuard AI acts as a smart behavioral signal aggregator, not an absolute verdict tool. Understanding its limits is crucial:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>What it Catches</b>: Real-time LLM query-generation delays; constant reading shifts from screen overlays or second monitors; "
        "physical glances down to check a phone; tab departure focus losses.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>System Boundaries</b>: It cannot detect an off-screen collaborator whispering answers if the candidate is articulate; "
        "pre-memorized answers; or candidates with strong physical reading discipline. Low room lighting or backlighting can "
        "also degrade FaceMesh tracking accuracy.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Citations & Open Source</b>: Built using MediaPipe FaceMesh WASM client, webkitSpeechRecognition API, "
        "FastAPI WebSockets, and ReportLab. It contains zero third-party agent installs or extensions.",
        bullet_style
    ))

    doc.build(story)
    print(f"Writeup PDF successfully generated at: {pdf_path}")

if __name__ == '__main__':
    generate_writeup_pdf()
