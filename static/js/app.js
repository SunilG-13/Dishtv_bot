/**
 * DISH TV Multilingual AI Portal & Knowledge Assistant
 * Supports 6 Regional Indian Languages: English, Hindi, Kannada, Telugu, Tamil, Gujarati
 * Connects directly with DeepSeek API, BAAI/bge-m3 embeddings, FAISS vector store, and TTS/STT services.
 */

class DishTVPortal {
  constructor() {
    this.currentLanguage = "en";
    this.autoPlayAudio = false;
    this.conversationHistory = [];
    this.isRecording = false;
    this.recognition = null;
    this.currentAudio = null;

    // Language Metadata for the 6 languages
    this.languages = {
      en: {
        code: "en",
        label: "EN",
        name: "English",
        native: "English",
        flag: "🇬🇧",
        voice_code: "en-IN",
        greeting: "Hello! I am your DISH TV AI Assistant. How can I assist you with your DISH TV recharge, packages, channel selection, or error codes today?"
      },
      hi: {
        code: "hi",
        label: "HI",
        name: "Hindi",
        native: "हिन्दी",
        flag: "🇮🇳",
        voice_code: "hi-IN",
        greeting: "नमस्ते! मैं आपका DISH TV AI सहायक हूँ। आज मैं आपकी डिश टीवी सेवा, एरर कोड, रिचार्ज या पैक से संबंधित कैसे मदद कर सकता हूँ?"
      },
      kn: {
        code: "kn",
        label: "KN",
        name: "Kannada",
        native: "ಕನ್ನಡ",
        flag: "🇮🇳",
        voice_code: "kn-IN",
        greeting: "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ DISH TV AI ಸಹಾಯಕ. ನಿಮ್ಮ DISH TV ರೀಚಾರ್ಜ್, ದೋಷ ಕೋಡ್‌ಗಳು ಅಥವಾ ಪ್ಯಾಕ್‌ಗಳ ಕುರಿತು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?"
      },
      te: {
        code: "te",
        label: "TE",
        name: "Telugu",
        native: "తెలుగు",
        flag: "🇮🇳",
        voice_code: "te-IN",
        greeting: "నమస్కారం! నేను మీ DISH TV AI అసిస్టెంట్‌ని. మీ DISH TV రీఛార్జ్, ఎర్رర్ కోడ్‌లు లేదా ప్యాక్‌ల గురించి మీకు ఎలా సహాయపడగలను?"
      },
      ta: {
        code: "ta",
        label: "TA",
        name: "Tamil",
        native: "தமிழ்",
        flag: "🇮🇳",
        voice_code: "ta-IN",
        greeting: "வணக்கம்! நான் உங்கள் DISH TV AI உதவியாளர். உங்கள் DISH TV ரீசார்ஜ், பிழை குறியீடுகள் அல்லது பேக்குகள் பற்றி உங்களுக்கு நான் எவ்வாறு உதவ முடியும்?"
      },
      gu: {
        code: "gu",
        label: "GU",
        name: "Gujarati",
        native: "ગુજરાતી",
        flag: "🇮🇳",
        voice_code: "gu-IN",
        greeting: "નમસ્તે! હું તમારો DISH TV AI સહાયક છું. આજે હું તમને DISH TV રિચાર્જ, એરર કોડ્સ અથવા ચેનલ પેક સંબંધિત કેવી રીતે મદદ કરી શકું?"
      }
    };

    // DOM Elements
    this.themeSwitch = document.getElementById("theme-switch");
    this.headerLangBtn = document.getElementById("headerLangBtn");
    this.headerLangMenu = document.getElementById("headerLangMenu");
    this.currentLangLabel = document.getElementById("currentLangLabel");
    this.mobileMenuBtn = document.getElementById("mobileMenuBtn");
    this.headerNav = document.getElementById("headerNav");

    // Carousel Elements
    this.carouselSlides = document.querySelectorAll(".carousel-slide");
    this.indicatorDots = document.querySelectorAll(".indicator-dot");
    this.prevBtn = document.getElementById("carouselPrevBtn");
    this.nextBtn = document.getElementById("carouselNextBtn");
    this.currentSlideIndex = 0;
    this.carouselTimer = null;

    // Bot Elements
    this.chatLauncher = document.getElementById("chatLauncher");
    this.widget = document.getElementById("knowledgeBotWidget");
    this.btnCloseWidget = document.getElementById("btnCloseWidget");
    this.messagesStream = document.getElementById("messagesStream");
    this.chatInput = document.getElementById("chatInput");
    this.btnSend = document.getElementById("btnSend");
    this.btnMic = document.getElementById("btnMic");
    this.btnAudioToggle = document.getElementById("btnAudioToggle");
    this.voiceVisualizer = document.getElementById("voiceVisualizer");
    this.btnCancelVoice = document.getElementById("btnCancelVoice");
    this.quickPromptsBar = document.getElementById("quickPromptsBar");
    this.siteLogo = document.getElementById("siteLogo");

    this.init();
  }

  async init() {
    this.initTheme();
    this.initCarousel();
    this.initAccordions();
    this.bindEvents();
    this.initSpeechRecognition();
    this.renderInitialGreeting();
  }

  /* --- 1. THEME SWITCHER (DARK / LIGHT) --- */
  initTheme() {
    const savedTheme = localStorage.getItem("dishtv_theme") || "light";
    document.body.className = savedTheme;
    this.updateLogoForTheme(savedTheme);

    if (this.themeSwitch) {
      this.themeSwitch.checked = savedTheme === "light";
      this.themeSwitch.addEventListener("change", (e) => {
        const newTheme = e.target.checked ? "light" : "dark";
        document.body.className = newTheme;
        localStorage.setItem("dishtv_theme", newTheme);
        this.updateLogoForTheme(newTheme);
      });
    }
  }

  updateLogoForTheme(theme) {
    if (this.siteLogo) {
      this.siteLogo.src = theme === "light"
        ? "/static/images/dishtv-logonew.png"
        : "/static/images/dishtv-logo-white-trans.png";
    }
  }

  detectScript(text) {
    const counts = {
      kn: (text.match(/[\u0C80-\u0CFF]/g) || []).length,
      te: (text.match(/[\u0C00-\u0C7F]/g) || []).length,
      ta: (text.match(/[\u0B80-\u0BFF]/g) || []).length,
      gu: (text.match(/[\u0A80-\u0AFF]/g) || []).length,
      hi: (text.match(/[\u0900-\u097F]/g) || []).length,
    };
    let maxLang = null;
    let maxCount = 0;
    for (const [lang, count] of Object.entries(counts)) {
      if (count > maxCount) {
        maxCount = count;
        maxLang = lang;
      }
    }
    return maxLang;
  }

  /* --- 2. HERO CAROUSEL AUTO-SLIDER --- */
  initCarousel() {
    if (!this.carouselSlides.length) return;

    const showSlide = (index) => {
      this.carouselSlides.forEach((slide, i) => {
        slide.classList.toggle("active", i === index);
      });
      this.indicatorDots.forEach((dot, i) => {
        dot.classList.toggle("active", i === index);
      });
      this.currentSlideIndex = index;
    };

    const nextSlide = () => {
      const nextIndex = (this.currentSlideIndex + 1) % this.carouselSlides.length;
      showSlide(nextIndex);
    };

    const prevSlide = () => {
      const prevIndex = (this.currentSlideIndex - 1 + this.carouselSlides.length) % this.carouselSlides.length;
      showSlide(prevIndex);
    };

    if (this.nextBtn) this.nextBtn.addEventListener("click", () => { nextSlide(); this.resetCarouselTimer(); });
    if (this.prevBtn) this.prevBtn.addEventListener("click", () => { prevSlide(); this.resetCarouselTimer(); });

    this.indicatorDots.forEach((dot, idx) => {
      dot.addEventListener("click", () => {
        showSlide(idx);
        this.resetCarouselTimer();
      });
    });

    this.startCarouselTimer(nextSlide);
  }

  startCarouselTimer(callback) {
    this.carouselTimer = setInterval(callback, 5500);
  }

  resetCarouselTimer() {
    if (this.carouselTimer) {
      clearInterval(this.carouselTimer);
      this.startCarouselTimer(() => {
        const nextIndex = (this.currentSlideIndex + 1) % this.carouselSlides.length;
        this.carouselSlides.forEach((slide, i) => slide.classList.toggle("active", i === nextIndex));
        this.indicatorDots.forEach((dot, i) => dot.classList.toggle("active", i === nextIndex));
        this.currentSlideIndex = nextIndex;
      });
    }
  }

  /* --- 3. ACCORDIONS (OFFERINGS & FAQS) --- */
  initAccordions() {
    // Offerings Accordion
    const offeringHeaders = document.querySelectorAll(".offering-header");
    offeringHeaders.forEach((header) => {
      header.addEventListener("click", () => {
        const card = header.closest(".offering-card");
        const body = card.querySelector(".offering-body");
        const icon = card.querySelector(".offering-toggle-icon i");
        const isOpen = body.classList.contains("open");

        // Close other offering cards
        document.querySelectorAll(".offering-card").forEach((c) => {
          c.classList.remove("active");
          c.querySelector(".offering-body").classList.remove("open");
          c.querySelector(".offering-toggle-icon i").className = "fas fa-plus";
        });

        if (!isOpen) {
          card.classList.add("active");
          body.classList.add("open");
          icon.className = "fas fa-minus";
        }
      });
    });

    // FAQs Accordion
    const faqBtns = document.querySelectorAll(".faq-question-btn");
    faqBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const card = btn.closest(".faq-card");
        const isOpened = card.classList.contains("open");

        // Toggle current card
        card.classList.toggle("open", !isOpened);
        btn.setAttribute("aria-expanded", !isOpened);
      });
    });
  }

  /* --- 4. EVENT BINDINGS --- */
  bindEvents() {
    // Mobile Navigation Toggle
    if (this.mobileMenuBtn && this.headerNav) {
      this.mobileMenuBtn.addEventListener("click", () => {
        this.headerNav.classList.toggle("mobile-open");
      });
    }

    // Header Language Dropdown Toggle
    if (this.headerLangBtn) {
      this.headerLangBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.headerLangBtn.closest(".lang-dropdown-wrapper").classList.toggle("open");
      });

      // Close language dropdown on outside click
      document.addEventListener("click", () => {
        document.querySelectorAll(".lang-dropdown-wrapper").forEach((el) => el.classList.remove("open"));
      });
    }

    // Language Option Selection in Header
    document.querySelectorAll(".lang-option-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const lang = btn.getAttribute("data-lang");
        this.switchLanguage(lang);
      });
    });

    // Language Tags in Footer
    document.querySelectorAll(".footer-lang-tags .lang-tag").forEach((tag) => {
      tag.addEventListener("click", () => {
        const lang = tag.getAttribute("data-lang");
        this.switchLanguage(lang);
        this.openWidget();
      });
    });


    // Assistant Open / Close
    if (this.chatLauncher) {
      this.chatLauncher.addEventListener("click", () => this.toggleWidget());
    }

    if (this.btnCloseWidget) {
      this.btnCloseWidget.addEventListener("click", () => this.closeWidget());
    }

    // Assistant Navigation Link Trigger
    const navAssistantTrigger = document.getElementById("navAssistantTrigger");
    if (navAssistantTrigger) {
      navAssistantTrigger.addEventListener("click", () => this.openWidget());
    }

    // Launch bot buttons throughout the page
    document.querySelectorAll(".launch-bot-btn").forEach((btn) => {
      btn.addEventListener("click", () => this.openWidget());
    });

    // Buttons triggering specific query
    document.querySelectorAll(".launch-bot-query").forEach((btn) => {
      btn.addEventListener("click", () => {
        const prompt = btn.getAttribute("data-prompt");
        this.openWidget();
        if (prompt) {
          this.chatInput.value = prompt;
          this.handleSendMessage();
        }
      });
    });

    // Quick Prompts Chips inside the bot
    if (this.quickPromptsBar) {
      this.quickPromptsBar.addEventListener("click", (e) => {
        const chip = e.target.closest(".quick-chip");
        if (chip) {
          const query = chip.getAttribute("data-query");
          if (query) {
            this.chatInput.value = query;
            this.handleSendMessage();
          }
        }
      });
    }

    // Send Message Triggers
    if (this.btnSend) {
      this.btnSend.addEventListener("click", () => this.handleSendMessage());
    }

    if (this.chatInput) {
      this.chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.handleSendMessage();
        }
      });
    }

    // Voice Microphone Button
    if (this.btnMic) {
      this.btnMic.addEventListener("click", () => this.toggleSpeechRecognition());
    }

    if (this.btnCancelVoice) {
      this.btnCancelVoice.addEventListener("click", () => this.stopSpeechRecognition());
    }

    // Audio Autoplay Toggle
    if (this.btnAudioToggle) {
      this.btnAudioToggle.addEventListener("click", () => {
        this.autoPlayAudio = !this.autoPlayAudio;
        this.btnAudioToggle.innerHTML = this.autoPlayAudio 
          ? '<i class="fas fa-volume-up"></i>' 
          : '<i class="fas fa-volume-mute"></i>';
        this.btnAudioToggle.title = this.autoPlayAudio ? "Auto-Audio: ON" : "Auto-Audio: OFF";
      });
    }
  }

  /* --- 5. LANGUAGE SWITCHER LOGIC --- */
  switchLanguage(langCode, updateGreeting = false) {
    if (!this.languages[langCode]) return;
    this.currentLanguage = langCode;

    const info = this.languages[langCode];

    // Update Header Label
    if (this.currentLangLabel) {
      this.currentLangLabel.textContent = info.label;
    }

    // Update Header Active Class
    document.querySelectorAll(".lang-option-btn").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-lang") === langCode);
    });

    // Update Speech Recognition Language
    if (this.recognition) {
      this.recognition.lang = info.voice_code;
    }

    if (updateGreeting) {
      this.renderInitialGreeting(true);
    }

    // Update Visualizer Label
    const visText = document.getElementById("visualizerText");
    if (visText) {
      visText.textContent = `Listening in ${info.native} (${info.name})...`;
    }
  }

  /* --- 6. WIDGET TOGGLE CONTROLS --- */
  openWidget() {
    if (this.widget) {
      this.widget.classList.add("open");
      const launcherWrap = document.querySelector(".assistant-launcher-container");
      if (launcherWrap) launcherWrap.classList.add("hide-hint");
      if (this.chatInput) this.chatInput.focus();
    }
  }

  closeWidget() {
    if (this.widget) {
      this.widget.classList.remove("open");
      const launcherWrap = document.querySelector(".assistant-launcher-container");
      if (launcherWrap) launcherWrap.classList.remove("hide-hint");
      this.stopSpeechRecognition();
      if (this.currentAudio) {
        this.currentAudio.pause();
        this.currentAudio = null;
      }
    }
  }

  toggleWidget() {
    if (this.widget) {
      const isOpen = this.widget.classList.contains("open");
      if (isOpen) {
        this.closeWidget();
      } else {
        this.openWidget();
      }
    }
  }

  /* --- 7. INITIAL BOT GREETING --- */
  renderInitialGreeting(force = false) {
    if (force || this.messagesStream.children.length === 0) {
      this.messagesStream.innerHTML = "";
      const welcomeContent = `
        <div style="font-size: 0.94rem; line-height: 1.55;">
          <p style="margin-bottom: 8px;">👋 <b>Hi! Welcome to DishTV.</b></p>
          <p style="margin-bottom: 10px;">I can help you find information about products, packs, channels, recharge, connections, and error codes available on this website.</p>
          <div style="background: rgba(239, 90, 27, 0.08); border-radius: 8px; padding: 10px 12px; border-left: 3px solid #ef5a1b; font-size: 0.84rem; margin-top: 6px;">
            <div style="font-weight: 700; color: #e94f14; margin-bottom: 4px;">
              ✦ Auto 6-Language Understanding
            </div>
            Ask in <strong>English, हिन्दी, ಕನ್ನಡ, తెలుగు, தமிழ், or ગુજરાતી</strong> — I will automatically respond in your language!
          </div>
        </div>
      `;
      this.appendMessage({
        role: "bot",
        text: welcomeContent,
        citations: []
      });
    }
  }

  /* --- 8. SPEECH RECOGNITION (STT) — AUTO-SEND ON SILENCE --- */
  /**
   * Smart mic mode:
   * - Click mic → mic turns ON, starts listening
   * - User speaks → text appears live in chat input
   * - User stops speaking → after 1.5s silence, auto-sends message
   * - Mic stays ON for the next question automatically
   * - Click mic again → stops mic completely
   *
   * Uses continuous=false + auto-restart to prevent text duplication bug.
   */
  initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("Speech Recognition API not supported in this browser.");
      if (this.btnMic) {
        this.btnMic.title = "Voice input is supported in Google Chrome & Microsoft Edge";
      }
      return;
    }

    // Current speech segment text
    this.currentUtterance = "";
    // Timer that fires auto-send after silence
    this.silenceTimer = null;
    // How long to wait after last speech before auto-sending (ms)
    this.SILENCE_TIMEOUT = 1500;
    // Flag: true when user wants mic to stay ON
    this.wantsContinuousListening = false;

    this.recognition = new SpeechRecognition();
    this.recognition.continuous = false;      // Single utterance — prevents text duplication
    this.recognition.interimResults = true;   // Show live partial text while speaking
    this.recognition.maxAlternatives = 1;

    const langInfo = this.languages[this.currentLanguage];
    this.recognition.lang = langInfo ? langInfo.voice_code : "en-IN";

    this.recognition.onstart = () => {
      this.isRecording = true;
      if (this.btnMic) this.btnMic.classList.add("recording");
      if (this.chatInput) {
        this.chatInput.placeholder = "Listening... Speak now";
        this.chatInput.focus();
      }
    };

    this.recognition.onresult = (event) => {
      // Cancel any pending auto-send timer — user is still speaking
      this.clearSilenceTimer();

      let interimTranscript = "";
      let finalTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const item = event.results[i];
        if (item.isFinal) {
          finalTranscript += item[0].transcript;
        } else {
          interimTranscript += item[0].transcript;
        }
      }

      // Show live preview directly in input while user is speaking
      if (interimTranscript) {
        this.chatInput.value = interimTranscript;
      }

      // When a phrase is finalized, save it and start silence timer
      if (finalTranscript) {
        const cleanFinal = finalTranscript.trim();
        if (cleanFinal) {
          this.currentUtterance = cleanFinal;
          this.chatInput.value = cleanFinal;
          // Start silence timer — auto-send after 1.5s if no more speech
          this.startSilenceTimer();
        }
      }
    };

    this.recognition.onerror = (event) => {
      console.warn("Speech recognition error:", event.error);
      if (event.error === "not-allowed" || event.error === "permission-denied") {
        alert("Microphone permission denied. Please allow microphone access in your browser to use voice input.");
        this.stopSpeechRecognition();
      } else if (event.error === "no-speech") {
        // No speech detected — just restart and keep listening
        const visText = document.getElementById("visualizerText");
        if (visText) visText.textContent = "🎙️ Speak now...";
      } else if (event.error === "aborted") {
        this.stopSpeechRecognition();
      }
    };

    this.recognition.onend = () => {
      // Auto-restart mic if user hasn't clicked stop
      if (this.wantsContinuousListening) {
        const currentLangInfo = this.languages[this.currentLanguage];
        if (currentLangInfo) this.recognition.lang = currentLangInfo.voice_code;
        try {
          this.recognition.start();
        } catch (e) {
          setTimeout(() => {
            if (this.wantsContinuousListening) {
              try { this.recognition.start(); } catch (err) {
                this.stopSpeechRecognition();
              }
            }
          }, 250);
        }
      } else {
        this.isRecording = false;
        if (this.btnMic) this.btnMic.classList.remove("recording");
        if (this.voiceVisualizer) this.voiceVisualizer.classList.remove("active");
      }
    };
  }

  /** Starts a timer — if no new speech within SILENCE_TIMEOUT, auto-send the message */
  startSilenceTimer() {
    this.clearSilenceTimer();
    this.silenceTimer = setTimeout(() => {
      if (this.currentUtterance && this.currentUtterance.trim()) {
        this.chatInput.value = this.currentUtterance.trim();

        const visText = document.getElementById("visualizerText");
        if (visText) visText.textContent = "📤 Sending...";

        // Auto-send the message
        this.handleSendMessage();

        // Reset for next utterance (mic stays ON)
        this.currentUtterance = "";
      }
    }, this.SILENCE_TIMEOUT);
  }

  /** Clears the silence auto-send timer */
  clearSilenceTimer() {
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }
  }

  toggleSpeechRecognition() {
    if (!this.recognition) {
      alert("Voice input is supported in Google Chrome and Microsoft Edge browsers.");
      return;
    }

    if (this.isRecording) {
      // User clicked STOP — stop everything
      this.wantsContinuousListening = false;
      this.clearSilenceTimer();
      try { this.recognition.stop(); } catch (e) {}
      this.stopSpeechRecognition();
    } else {
      // User clicked START — begin listening
      this.currentUtterance = "";
      this.chatInput.value = "";
      this.wantsContinuousListening = true;

      const langInfo = this.languages[this.currentLanguage];
      if (langInfo) this.recognition.lang = langInfo.voice_code;
      try {
        this.recognition.start();
      } catch (e) {
        try {
          this.recognition.stop();
          setTimeout(() => {
            if (this.wantsContinuousListening) this.recognition.start();
          }, 200);
        } catch (err) {}
      }
    }
  }

  stopSpeechRecognition() {
    this.isRecording = false;
    this.wantsContinuousListening = false;
    this.clearSilenceTimer();
    if (this.btnMic) this.btnMic.classList.remove("recording");
    if (this.voiceVisualizer) this.voiceVisualizer.classList.remove("active");
  }

  /* --- 9. SEND MESSAGE & RAG CONVERSATION --- */
  async handleSendMessage() {
    const query = this.chatInput.value.trim();
    if (!query) return;

    this.chatInput.value = "";
    this.chatInput.focus();

    // Check if user entered an Indic script
    const detectedScript = this.detectScript(query);
    if (detectedScript) {
      this.currentLanguage = detectedScript;
      if (this.recognition) {
        this.recognition.lang = this.languages[detectedScript].voice_code;
      }
    }

    // Append user message bubble
    this.appendMessage({
      role: "user",
      text: query
    });

    this.conversationHistory.push({ role: "user", content: query });

    // Show typing loader
    const typingIndicator = this.showTypingIndicator();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          language: this.currentLanguage,
          history: this.conversationHistory.slice(-4)
        })
      });

      const data = await response.json();
      typingIndicator.remove();

      if (data.answer) {
        // Automatically synchronize language with the backend's detected response language
        if (data.language && this.languages[data.language]) {
          this.currentLanguage = data.language;
          if (this.recognition) {
            this.recognition.lang = this.languages[data.language].voice_code;
          }
        }

        this.conversationHistory.push({ role: "assistant", content: data.answer });

        this.appendMessage({
          role: "bot",
          text: data.answer,
          citations: data.citations || [],
          metrics: data.metrics,
          audioUrl: data.audio_url
        });

        // Auto-play audio if enabled
        if (this.autoPlayAudio && data.audio_url) {
          this.speakMessage(data.answer, data.audio_url, null);
        }
      } else {
        this.appendMessage({
          role: "bot",
          text: "I could not retrieve an answer right now. Please try asking in another way.",
          citations: []
        });
      }
    } catch (error) {
      console.error("Chat error:", error);
      typingIndicator.remove();
      this.appendMessage({
        role: "bot",
        text: "Connection error. Please verify the DishTV service is running.",
        citations: []
      });
    }
  }

  showTypingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row bot typing-row";
    row.innerHTML = `
      <div class="typing-bubble">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    `;
    this.messagesStream.appendChild(row);
    this.scrollToBottom();
    return row;
  }

  appendMessage({ role, text, citations = [], metrics = null, audioUrl = null }) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const formattedText = this.formatMarkdown(text);

    let citationsHtml = "";
    if (citations && citations.length > 0) {
      const chips = citations.map((c) => {
        const cleanTitle = c.title || "Official DishTV Source";
        const shortTitle = cleanTitle.length > 28 ? cleanTitle.substring(0, 28) + "..." : cleanTitle;
        return `
          <a href="${c.url}" target="_blank" rel="noopener noreferrer" class="citation-link" title="${cleanTitle}">
            <i class="fas fa-external-link-alt"></i> ${shortTitle}
          </a>
        `;
      }).join("");

      citationsHtml = `
        <div class="citations-box">
          <div class="citations-header">
            <i class="fas fa-check-circle"></i> Official DishTV Source (${citations.length})
          </div>
          <div class="citation-list">${chips}</div>
        </div>
      `;
    }

    let actionsHtml = "";
    if (role === "bot") {
      const latencyStr = metrics && metrics.total_latency_seconds ? `⚡ ${metrics.total_latency_seconds}s` : "";
      actionsHtml = `
        <div class="message-actions">
          <button class="action-btn btn-speak" title="Listen (TTS)">
            <svg class="tts-speaker-icon" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="display:inline-block; vertical-align:middle; margin-right:3px;">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
              <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
            </svg> Listen
          </button>
          <button class="action-btn btn-copy" title="Copy text">
            <i class="fas fa-copy"></i> Copy
          </button>
          ${latencyStr ? `<span class="latency-tag">${latencyStr}</span>` : ""}
        </div>
      `;
    }

    row.innerHTML = `
      <div class="message-bubble">${role === "bot" && text.length > 20 ? '<span class="stream-cursor"></span>' : formattedText}</div>
      ${citationsHtml}
      ${actionsHtml}
    `;

    this.messagesStream.appendChild(row);

    if (role === "bot" && text && text.length > 20) {
      const bubble = row.querySelector(".message-bubble");
      const tokens = text.match(/(\n+|\S+\s*)/g) || [text];
      let tIdx = 0;
      let curr = [];
      const stream = () => {
        if (tIdx < tokens.length) {
          curr.push(tokens[tIdx++]);
          bubble.innerHTML = this.formatMarkdown(curr.join('')) + '<span class="stream-cursor"></span>';
          this.scrollToBottom();
          const lastTok = tokens[tIdx - 1] || '';
          const delay = lastTok.includes('\n') ? 60 : (/[.,:;!?]$/.test(lastTok.trim()) ? 40 : 22);
          setTimeout(stream, delay);
        } else {
          bubble.innerHTML = formattedText;
          this.scrollToBottom();
        }
      };
      stream();
    }

    // Bind Speak and Copy events
    if (role === "bot") {
      const btnSpeak = row.querySelector(".btn-speak");
      if (btnSpeak) {
        btnSpeak.addEventListener("click", () => this.speakMessage(text, audioUrl, btnSpeak));
      }

      const btnCopy = row.querySelector(".btn-copy");
      if (btnCopy) {
        btnCopy.addEventListener("click", () => {
          navigator.clipboard.writeText(text);
          btnCopy.innerHTML = '<i class="fas fa-check"></i> Copied';
          setTimeout(() => {
            btnCopy.innerHTML = '<i class="fas fa-copy"></i> Copy';
          }, 2000);
        });
      }
    }

    this.scrollToBottom();
  }

  /* --- 10. TEXT TO SPEECH (TTS) --- */
  speakMessage(text, audioUrl, buttonEl) {
    // If currently playing, pause and reset
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio = null;
      if (this.activeSpeakButton) {
        this.activeSpeakButton.classList.remove("active");
        this.activeSpeakButton.innerHTML = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle; margin-right:3px;"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg> Listen';
        this.activeSpeakButton = null;
      }
      return;
    }

    if (window.speechSynthesis && window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      if (this.activeSpeakButton) {
        this.activeSpeakButton.classList.remove("active");
        this.activeSpeakButton.innerHTML = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle; margin-right:3px;"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg> Listen';
        this.activeSpeakButton = null;
      }
      return;
    }

    this.activeSpeakButton = buttonEl;
    if (buttonEl) {
      buttonEl.classList.add("active");
      buttonEl.innerHTML = '<i class="fas fa-pause"></i> Playing...';
    }

    const resetBtn = () => {
      if (buttonEl) {
        buttonEl.classList.remove("active");
        buttonEl.innerHTML = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle; margin-right:3px;"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg> Listen';
      }
      this.activeSpeakButton = null;
      this.currentAudio = null;
    };

    if (audioUrl) {
      const audio = new Audio(audioUrl);
      this.currentAudio = audio;

      audio.play().then(() => {
        audio.onended = resetBtn;
      }).catch((err) => {
        console.warn("Direct audio stream failed, falling back to Web Speech:", err);
        this.fallbackWebSpeechSynthesis(text, buttonEl, resetBtn);
      });
    } else {
      this.fallbackWebSpeechSynthesis(text, buttonEl, resetBtn);
    }
  }

  fallbackWebSpeechSynthesis(text, buttonEl, onComplete) {
    if (!('speechSynthesis' in window)) {
      if (onComplete) onComplete();
      return;
    }

    // Clean markdown, links, and bullet punctuation
    const cleanText = text
      .replace(/https?:\/\/\S+/g, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/[*_#`~>]/g, '')
      .replace(/^\s*[-•*+]\s+/gm, '')
      .replace(/\s+/g, ' ')
      .trim();

    const utterance = new SpeechSynthesisUtterance(cleanText);
    const langInfo = this.languages[this.currentLanguage];
    if (langInfo) {
      utterance.lang = langInfo.voice_code;
      // Attempt to pick matching regional voice if available
      const voices = window.speechSynthesis.getVoices();
      const match = voices.find((v) => 
        v.lang === langInfo.voice_code || v.lang.startsWith(langInfo.code)
      );
      if (match) utterance.voice = match;
    }

    utterance.onend = () => {
      if (onComplete) onComplete();
    };

    utterance.onerror = () => {
      if (onComplete) onComplete();
    };

    window.speechSynthesis.speak(utterance);
  }

  /* --- 11. MARKDOWN PARSER --- */
  formatMarkdown(text) {
    if (!text) return "";

    if (typeof marked !== "undefined" && typeof marked.parse === "function") {
      try {
        marked.setOptions({ breaks: true, gfm: true });
        return marked.parse(text);
      } catch (e) {
        console.warn("marked.parse fallback:", e);
      }
    }

    // 0. Remove footnote/disclaimer asterisks from prices and durations (e.g. ₹1,440* -> ₹1,440, 1 Year** -> 1 Year)
    text = text.replace(/(₹\s*[\d,]+(?:\.\d+)?)\*+/g, '$1');
    text = text.replace(/(\b\d+\s+(?:Year|Month|Day|Hour)s?)\*+/gi, '$1');
    text = text.replace(/(\b[\w\u0900-\u0D7F]+)\*(?!\*|\w)/g, '$1');

    // Fallback simple parser
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Repair unbalanced asterisks if odd count
    if ((html.split('**').length - 1) % 2 !== 0) {
      html = html.replace(/(?:^|(?<=\s))\*([^* \n]+)\*\*(?=\s|[-—:.,]|$)/g, '**$1**');
      if ((html.split('**').length - 1) % 2 !== 0) {
        html = html.replace(/(?:^|(?<=\s))([A-Za-z\u0900-\u0D7F ]+?)\*\*(?=\s*[-—:])/g, '**$1**');
      }
    }

    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="bot-bold">$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/(\+?91[\s-]?[6-9][\d\s-]{8,12}\d|\b1800[\s-]?\d{3}[\s-]?(?:\d{3,4}|[xX]{4})\b|\b95017[\s-]?95017\b)/g, (match) => {
      const rawNumber = match.replace(/[\s-]/g, '');
      return `<a href="tel:${rawNumber}" class="bot-phone-link" title="Call ${match}"><i class="fas fa-phone-alt"></i> ${match}</a>`;
    });
    html = html.replace(/(₹\s?\d+(?:,\d+)*(?:\.\d+)?)/g, '<span class="bot-price">$1</span>');
    html = html.replace(/^\s*\d+[\.\)]\s+(.*)$/gm, '<li>$1</li>');
    html = html.replace(/^\s*[-•*+]\s+(.*)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>(\r?\n)?)+/gs, (m) => `<ol>${m}</ol>`);
    return html;
  }

  scrollToBottom() {
    this.messagesStream.scrollTop = this.messagesStream.scrollHeight;
  }
}

// Instantiate once DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.dishPortal = new DishTVPortal();
});
