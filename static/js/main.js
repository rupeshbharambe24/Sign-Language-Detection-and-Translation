class SignLanguageTranslator {
    constructor() {
      this.translationMode = "letter"
      this.isTranslating = false
      this.currentSentence = ""
      this.capturedLetters = "" // Store captured letters for sentence mode
      this.videoElement = document.getElementById("video")
      this.predictionElement = document.getElementById("prediction-text")
      this.sentenceElement = document.getElementById("current-translation")
      this.capturedLettersElement = document.getElementById("captured-letters-text")
      this.selectedLanguage = "en" // Default language
      this.audioContext = null
      this.showMesh = true
      this.left_triplets = [
        [4, 2, 0],
        [4, 1, 2],
        [4, 8, 12],
        [4, 20, 0],
        [8, 5, 0],
        [8, 9, 0],
        [8, 12, 16],
        [12, 9, 0],
        [12, 8, 16],
        [16, 13, 0],
        [16, 12, 20],
        [20, 17, 0],
        [20, 16, 12],
        [0, 5, 17],
        [0, 1, 17],
        [0, 8, 20],
        [0, 4, 20],
        [0, 1, 17],
        [0, 5, 17],
        [4, 20, 0],
        [0, 4, 8],
        [0, 4, 20],
        [8, 12, 16],
        [0, 8, 20],
        [4, 8, 12],
        [4, 8, 0],
        [4, 12, 0],
        [0, 5, 17],
        [0, 4, 20],
        [8, 12, 16],
      ]
      this.right_triplets = [
        [4, 2, 0],
        [4, 1, 2],
        [4, 8, 12],
        [4, 20, 0],
        [8, 5, 0],
        [8, 9, 0],
        [8, 12, 16],
        [12, 9, 0],
        [12, 8, 16],
        [16, 13, 0],
        [16, 12, 20],
        [20, 17, 0],
        [20, 16, 12],
        [0, 5, 17],
        [0, 1, 17],
        [0, 8, 20],
        [0, 4, 20],
        [0, 1, 17],
        [0, 5, 17],
        [4, 20, 0],
        [0, 4, 8],
        [0, 4, 20],
        [8, 12, 16],
        [0, 8, 20],
        [4, 8, 12],
        [4, 8, 0],
        [4, 12, 0],
        [0, 5, 17],
        [0, 4, 20],
        [8, 12, 16],
      ]
  
      // Initialize audio context
      this.setupAudio()
  
      // Bind methods
      this.startTranslation = this.startTranslation.bind(this)
      this.stopTranslation = this.stopTranslation.bind(this)
      this.setMode = this.setMode.bind(this)
      this.captureFrame = this.captureFrame.bind(this)
  
      // Initialize components
      this.setupEventListeners()
      this.setupWebcam()
      this.updateUI()
    }
  
    setupAudio() {
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)()
      }
    }
  
    setupEventListeners() {
      // Language selector
      document.getElementById("language-selector").addEventListener("change", (e) => {
        this.selectedLanguage = e.target.value
      })
  
      // Mode selection buttons
      document.querySelectorAll(".mode-btn").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          const mode = e.target.dataset.mode
          console.log(`Mode selected: ${mode}`)
          this.setMode(mode)
        })
      })
  
      // Translation control buttons
      document.getElementById("start-translate")?.addEventListener("click", () => {
        console.log("Start Translation clicked")
        this.startTranslation()
      })
  
      document.getElementById("stop-translate")?.addEventListener("click", () => {
        console.log("Stop Translation clicked")
        this.stopTranslation()
      })
  
      document
        .getElementById("ok-button")
        ?.addEventListener("click", () => {
          console.log("OK button clicked")
          this.stopTranslation()
          alert(`Final Sentence: ${this.currentSentence}`)
        })
  
      // Sentence mode buttons
      ;["space", "fullstop"].forEach((action) => {
        document.getElementById(`add-${action}`)?.addEventListener("click", () => {
          console.log(`${action} button clicked`)
          this.sendAction(action)
        })
      })
  
      // Mesh toggle
      document.getElementById("mesh-toggle")?.addEventListener("click", () => {
        this.showMesh = !this.showMesh
        document.getElementById("mesh-toggle").textContent = this.showMesh ? "Hide Mesh" : "Show Mesh"
      })
    }
  
    setupWebcam() {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          this.videoElement.srcObject = stream
          this.videoElement.play()
          console.log("Webcam started")
        })
        .catch((error) => {
          console.error("Error accessing webcam:", error)
          alert("Unable to access webcam. Please ensure you have granted camera permissions.")
        })
    }
  
    drawLandmarks(context, handLandmarks, landmarkColor, connectionColor) {
      // Draw hand landmarks and connections
      const HAND_CONNECTIONS = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4], // Thumb
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8], // Index finger
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12], // Middle finger
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16], // Ring finger
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20], // Pinky
      ]
      const landmarks = handLandmarks
      const connections = HAND_CONNECTIONS
  
      // Draw connections
      connections.forEach((connection) => {
        const startLandmark = landmarks[connection[0]]
        const endLandmark = landmarks[connection[1]]
  
        if (startLandmark && endLandmark) {
          context.beginPath()
          context.moveTo(startLandmark[0] * context.canvas.width, startLandmark[1] * context.canvas.height)
          context.lineTo(endLandmark[0] * context.canvas.width, endLandmark[1] * context.canvas.height)
          context.strokeStyle = `rgb(${connectionColor.join(",")})`
          context.lineWidth = 2
          context.stroke()
        }
      })
  
      // Draw landmarks
      landmarks.forEach((landmark, index) => {
        if (landmark) {
          const x = landmark[0] * context.canvas.width
          const y = landmark[1] * context.canvas.height
  
          context.beginPath()
          context.arc(x, y, 3, 0, 2 * Math.PI)
          context.fillStyle = `rgb(${landmarkColor.join(",")})`
          context.fill()
        }
      })
    }
  
    drawTriplets(context, handLandmarks, width, height, tripletColor, indexColor) {
      // Draw triplet lines and indices
      const landmarks = handLandmarks
  
      // Combine left and right triplets
      const allTriplets = this.left_triplets.concat(this.right_triplets)
  
      for (const triplet of allTriplets) {
        const [p1, p2, p3] = triplet
  
        if (p1 < landmarks.length && p2 < landmarks.length && p3 < landmarks.length) {
          const pt1 = landmarks[p1]
          const pt2 = landmarks[p2]
          const pt3 = landmarks[p3]
  
          // Draw lines for the triplet
          ;[pt1, pt2, pt3].forEach((pt, i) => {
            const nextPt = [pt2, pt3, pt1][i]
            if (pt && nextPt) {
              context.beginPath()
              context.moveTo(pt[0] * width, pt[1] * height)
              context.lineTo(nextPt[0] * width, nextPt[1] * height)
              context.strokeStyle = `rgb(${tripletColor.join(",")})`
              context.lineWidth = 2
              context.stroke()
            }
          })
  
          // Display indices for the triplet points
          ;[pt1, pt2, pt3].forEach((pt, i) => {
            if (pt) {
              context.fillStyle = `rgb(${indexColor.join(",")})`
              context.font = "16px Arial"
              context.fillText(triplet[i], pt[0] * width, pt[1] * height)
            }
          })
        }
      }
    }
  
    async captureFrame() {
      if (!this.isTranslating) return
  
      const canvas = document.createElement("canvas")
      canvas.width = this.videoElement.videoWidth
      canvas.height = this.videoElement.videoHeight
      const context = canvas.getContext("2d")
      context.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height)
  
      if (this.showMesh) {
        // Get canvas dimensions
        const height = canvas.height
        const width = canvas.width
  
        // Draw hand landmarks and connections
        if (this.multiHandLandmarks) {
          for (const handLandmarks of this.multiHandLandmarks) {
            // Draw MediaPipe hand landmarks and connections
            this.drawLandmarks(
              context,
              handLandmarks,
              [0, 255, 0], // Landmark color
              [0, 0, 255], // Connection color
            )
  
            // Draw triplet lines and indices
            this.drawTriplets(
              context,
              handLandmarks,
              width,
              height,
              [255, 0, 0], // Triplet line color
              [0, 255, 255], // Index color
            )
          }
        }
      }
      const imageData = canvas.toDataURL("image/jpeg")
  
      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: imageData }),
        })
  
        const rawResponse = await response.text()
        console.log("Raw Response:", rawResponse)
  
        let data
        try {
          data = JSON.parse(rawResponse)
        } catch (error) {
          console.error("Failed to parse response as JSON:", error)
          return
        }
  
        console.log("Prediction Response:", data)
  
        if (data.error) {
          console.error("Backend error:", data.error)
          return
        }
  
        this.predictionElement.textContent = `${data.prediction} (${data.confidence.toFixed(2)})`
  
        if (this.translationMode !== "letter" && data.captured) {
          this.currentSentence += data.prediction
          this.capturedLetters += data.prediction
          this.sentenceElement.textContent = this.currentSentence
          this.capturedLettersElement.textContent = this.capturedLetters
        }
  
        // Update hand landmarks for drawing
        this.multiHandLandmarks = data.multi_hand_landmarks
      } catch (error) {
        console.error("Error capturing frame:", error)
      }
  
      setTimeout(this.captureFrame.bind(this), 100)
    }
  
    async startTranslation() {
      try {
        this.isTranslating = true
        this.currentSentence = "" // Reset sentence
        this.capturedLetters = "" // Reset captured letters
        console.log("Starting translation...")
  
        const response = await fetch("/start_translation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        })
  
        const data = await response.json()
        console.log("Start Translation Response:", data)
  
        this.captureFrame()
        this.updateUI()
      } catch (error) {
        console.error("Error starting translation:", error)
      }
    }
  
    async stopTranslation() {
        try {
            this.isTranslating = false;
            
            // Expand shortcuts in the final sentence
            const response = await fetch('/expand_shortcut', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: this.currentSentence,
                    language: this.selectedLanguage
                })
            });
            
            const data = await response.json();
            let finalText = data.expanded_text;
            
            console.log("Final text before correction:", finalText);  // Debugging
            
            // Analyze and correct the final text using Gemini
            const correctionResponse = await fetch('/analyze_and_correct', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: finalText
                })
            });
            
            const correctionData = await correctionResponse.json();
            if (correctionData.status === 'success') {
                finalText = correctionData.corrected_text;
                console.log("Final text after correction:", finalText);  // Debugging
            } else {
                console.error("Error correcting text:", correctionData.error);  // Debugging
            }
            
            // Generate audio for the final text
            const audioResponse = await fetch('/generate_audio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: finalText,
                    language: this.selectedLanguage
                })
            });
            
            console.log("Audio response:", audioResponse);  // Debugging
            
            const audioData = await audioResponse.json();
            if (audioData.status === 'success') {
                await this.playAudio(audioData.audio_base64); // Play the audio directly
            } else {
                console.error("Error generating audio:", audioData.error);  // Debugging
            }
            
            return {
                status: 'Translation stopped',
                final_sentence: finalText,
                is_translating: false
            };
        } catch (error) {
            console.error('Error stopping translation:', error);
            return {
                error: error.message,
                status: 'error'
            };
        }
    }

    
    async playAudio(audioBase64) {
      try {
        // Create an audio element
        const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`)
  
        // Play the audio
        audio.play()
      } catch (error) {
        console.error("Error playing audio:", error)
      }
    }
  
    async setMode(mode) {
      try {
        this.translationMode = mode
        console.log(`Setting mode to: ${mode}`)
  
        const response = await fetch("/set_mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode }),
        })
  
        const data = await response.json()
        console.log("Set Mode Response:", data)
        this.updateUI()
      } catch (error) {
        console.error("Error setting mode:", error)
      }
    }
  
    async sendAction(action) {
      try {
        console.log(`Sending action: ${action}`)
        const response = await fetch(`/add_${action}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        })
  
        const data = await response.json()
        console.log("Send Action Response:", data)
        this.updateUI()
      } catch (error) {
        console.error(`Error sending action ${action}:`, error)
      }
    }
  
    updateUI() {
      try {
        console.log("Updating UI...")
  
        // Update mode buttons
        document.querySelectorAll(".mode-btn").forEach((btn) => {
          btn.classList.toggle("active", btn.dataset.mode === this.translationMode)
        })
  
        // Update translation controls
        document.getElementById("start-translate")?.classList.toggle("hidden", this.isTranslating)
        document.getElementById("stop-translate")?.classList.toggle("hidden", !this.isTranslating)
  
        // Update sentence controls
        document.getElementById("sentence-controls")?.classList.toggle("hidden", this.translationMode !== "sentence")
  
        // Update translation display
        this.sentenceElement.textContent = this.currentSentence
  
        // Update mesh toggle button
        document.getElementById("mesh-toggle").textContent = this.showMesh ? "Hide Mesh" : "Show Mesh"
  
        // Fetch UI updates from server
        fetch("/update_ui")
          .then((response) => response.json())
          .then((data) => {
            console.log("Update UI Response:", data)
          })
          .catch((error) => {
            console.error("Error fetching UI updates:", error)
          })
      } catch (error) {
        console.error("Error updating UI:", error)
      }
    }
  }
  
  // Initialize when DOM is loaded
  document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM loaded. Initializing SignLanguageTranslator...")
    new SignLanguageTranslator()
  })
  
  