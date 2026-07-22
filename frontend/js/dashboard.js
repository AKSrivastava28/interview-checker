class ReviewerDashboard {
    constructor(wsClient, sessionId) {
        this.wsClient = wsClient;
        this.sessionId = sessionId;
        this.evidenceListContainer = null;
    }

    init() {
        this.evidenceListContainer = document.getElementById("evidence-list");

        // Set up report redirect button
        const btnReport = document.getElementById("btn-view-report");
        if (btnReport) {
            btnReport.addEventListener("click", () => {
                window.open(`/report/${this.sessionId}`, '_blank');
            });
        }

        // 1. Connection status listeners
        this.wsClient.on("candidate_status", (data) => {
            const statusDot = document.getElementById("candidate-status-dot");
            const statusLabel = document.getElementById("candidate-status-label");
            if (statusDot && statusLabel) {
                if (data.connected) {
                    statusDot.className = "status-dot online";
                    statusLabel.textContent = "Candidate Connected";
                } else {
                    statusDot.className = "status-dot offline";
                    statusLabel.textContent = "Candidate Disconnected";
                }
            }
        });

        // 2. Question progress listeners
        this.wsClient.on("new_question", (data) => {
            const currentQTitle = document.getElementById("current-question-title");
            if (currentQTitle) {
                currentQTitle.textContent = `Q${data.question_n}: ${data.question_text}`;
            }
        });

        // 3. Question evaluation results listeners
        this.wsClient.on("question_result", (data) => {
            if (data.result) {
                this.renderEvidenceCard(data.result);
            }
        });

        // 4. Interview complete listener
        this.wsClient.on("interview_complete", () => {
            const currentQTitle = document.getElementById("current-question-title");
            if (currentQTitle) {
                currentQTitle.textContent = "Interview Session Completed";
            }
            alert("Interview complete! Click 'View Audit Report' to see the full document.");
        });
    }

    renderEvidenceCard(result) {
        if (!this.evidenceListContainer) return;

        // Remove placeholder if present
        const emptyState = document.getElementById("empty-evidence-state");
        if (emptyState) emptyState.remove();

        const riskClass = result.risk; // clean | suspicious | high_risk
        const riskTitle = riskClass.toUpperCase().replace("_", " ");

        // Check if card for this question already exists to prevent duplication
        const existingCard = document.getElementById(`q-card-${result.question_n}`);
        if (existingCard) {
            existingCard.remove();
        }

        const cardHtml = `
            <div id="q-card-${result.question_n}" class="evidence-card ${riskClass}">
                <div class="card-header">
                    <strong style="font-size:1.1rem; color:var(--text-primary);">Question ${result.question_n}</strong>
                    <span class="risk-badge ${riskClass}">${riskTitle}</span>
                </div>
                
                <div style="font-size:0.95rem; font-weight:500; color:var(--text-primary); margin-bottom:1rem;">
                    "${result.question_text}"
                </div>

                <div class="metric-row">
                    <div class="metric-card">
                        <label>Pause Latency</label>
                        <span>${result.pause_s}s</span>
                    </div>
                    <div class="metric-card">
                        <label>Offscreen Gaze</label>
                        <span>${result.gaze_offscreen_pct}%</span>
                    </div>
                    <div class="metric-card">
                        <label>Focus Blurs</label>
                        <span>${result.blur_count}</span>
                    </div>
                    <div class="metric-card">
                        <label>AI Likeness</label>
                        <span>${result.ai_likeness_score}/100</span>
                    </div>
                </div>

                <div class="dashboard-rationale">
                    <strong style="color:var(--accent-blue); display:block; margin-bottom:0.25rem;">AI Evaluation & Rationale</strong>
                    ${result.ai_rationale}
                </div>

                ${result.transcript_text ? `
                    <div style="margin-top:1rem; font-size:0.85rem; color:var(--text-secondary); border-top:1px solid rgba(255,255,255,0.03); padding-top:0.75rem;">
                        <span style="font-weight:600; color:var(--text-primary);">Response Transcript:</span> "${result.transcript_text}"
                    </div>
                ` : ''}
            </div>
        `;

        this.evidenceListContainer.insertAdjacentHTML("afterbegin", cardHtml);
    }
}

window.ReviewerDashboard = ReviewerDashboard;
