class InterviewerVoice {
    constructor() {
        this.synth = window.speechSynthesis;
        this.currentUtterance = null;
    }

    speak(text, onStartCallback, onEndCallback) {
        if (!this.synth) {
            console.warn("[TTS] SpeechSynthesis not supported in this browser.");
            // Immediately trigger callbacks if TTS fails/not supported
            if (onStartCallback) onStartCallback();
            setTimeout(() => { if (onEndCallback) onEndCallback(); }, 2000);
            return;
        }

        // Cancel active audio
        this.synth.cancel();

        this.currentUtterance = new SpeechSynthesisUtterance(text);
        
        // Try selecting a professional sounding english voice if available
        const voices = this.synth.getVoices();
        const enVoice = voices.find(v => v.lang.startsWith("en") && v.name.includes("Google")) || 
                        voices.find(v => v.lang.startsWith("en"));
        if (enVoice) {
            this.currentUtterance.voice = enVoice;
        }

        this.currentUtterance.onstart = () => {
            console.log("[TTS] Started speaking question.");
            if (onStartCallback) onStartCallback();
        };

        this.currentUtterance.onend = () => {
            console.log("[TTS] Finished speaking question.");
            if (onEndCallback) onEndCallback();
        };

        this.currentUtterance.onerror = (e) => {
            console.warn("[TTS] Error encountered during speaking:", e);
            if (onEndCallback) onEndCallback();
        };

        this.synth.speak(this.currentUtterance);
    }

    stop() {
        if (this.synth) {
            this.synth.cancel();
        }
    }
}

window.InterviewerVoice = InterviewerVoice;
