from flask import Flask, render_template, request, jsonify
import cv2
import base64
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import time
from flask_cors import CORS
from symspellpy import SymSpell, Verbosity
from gtts import gTTS
import os
from translate import Translator
import io
import base64
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Function to analyze and correct text using Gemini
def analyze_and_correct_text(text):
    """
    Analyze and correct confusing words using Gemini API.
    """
    prompt = f"""
    Analyze the following text translated from sign language and correct any confusing or misspelled words:
    - {text}

    Return only the corrected text without any additional explanations.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        corrected_text = response.text.strip()
        return corrected_text
    except Exception as e:
        print(f"Error analyzing text with Gemini: {e}")
        return text  # Return original text if there's an error

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks and connections

# Load trained model and preprocessing tools
model = tf.keras.models.load_model("all-letter-model.h5")
with open("all-letter-model_scaler_and_encoder.pkl", "rb") as f:
    data = pickle.load(f)  # Ensure `data` is a dictionary
if isinstance(data, dict) and "scaler" in data and "label_encoder" in data:
    scaler = data["scaler"]
    label_encoder = data["label_encoder"]
else:
    raise ValueError("The file does not contain expected keys: 'scaler' and 'label_encoder'.")

# Define triplets for cosine angle computation
left_triplets = [
    (4, 2, 0), (4, 1, 2), (4, 8, 12), (4, 20, 0),
    (8, 5, 0), (8, 9, 0), (8, 12, 16),
    (12, 9, 0), (12, 8, 16),
    (16, 13, 0), (16, 12, 20),
    (20, 17, 0), (20, 16, 12),
    (0, 5, 17), (0, 1, 17), (0, 8, 20), (0, 4, 20),
    (0, 1, 17), (0, 5, 17), (4, 20, 0),
    (0, 4, 8), (0, 4, 20), (8, 12, 16), (0, 8, 20), (4, 8, 12),
    (4, 8, 0), (4, 12, 0),
    (0, 5, 17), (0, 4, 20), (8, 12, 16)
]
right_triplets = [
    (4, 2, 0), (4, 1, 2), (4, 8, 12), (4, 20, 0),
    (8, 5, 0), (8, 9, 0), (8, 12, 16),
    (12, 9, 0), (12, 8, 16),
    (16, 13, 0), (16, 12, 20),
    (20, 17, 0), (20, 16, 12),
    (0, 5, 17), (0, 1, 17), (0, 8, 20), (0, 4, 20),
    (0, 1, 17), (0, 5, 17), (4, 20, 0),
    (0, 4, 8), (0, 4, 20), (8, 12, 16), (0, 8, 20), (4, 8, 12),
    (4, 8, 0), (4, 12, 0),
    (0, 5, 17), (0, 4, 20), (8, 12, 16)
]

# Function to compute cosine similarity angles
def compute_cosineSim_angles(landmarks, triplets):
    angles = []
    if len(landmarks) < 42:
        return [0] * len(triplets)
    for (a, b, c) in triplets:
        try:
            x1, y1 = landmarks[a * 2], landmarks[a * 2 + 1]
            x2, y2 = landmarks[b * 2], landmarks[b * 2 + 1]
            x3, y3 = landmarks[c * 2], landmarks[c * 2 + 1]
            AB = np.array([x1 - x2, y1 - y2])
            BC = np.array([x3 - x2, y3 - y2])
            dot_product = np.dot(AB, BC)
            norm_AB = np.linalg.norm(AB)
            norm_BC = np.linalg.norm(BC)
            if norm_AB == 0 or norm_BC == 0:
                theta = 0
            else:
                cos_theta = np.clip(dot_product / (norm_AB * norm_BC), -1.0, 1.0)
                theta = np.degrees(np.arccos(cos_theta))
            angles.append(theta)
        except IndexError:
            angles.append(0)
    return angles

# Function to extract hand landmarks
def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    left_hand_data = [0.0] * 42
    right_hand_data = [0.0] * 42
    hands_detected = False  # Flag to indicate if hands are detected
    if results.multi_hand_landmarks and results.multi_handedness:
        hands_detected = True
        for idx, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label
            hand_landmarks = results.multi_hand_landmarks[idx]
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            flat_landmarks = [coord for lm in landmarks for coord in lm]
            if label == "Left":
                left_hand_data = flat_landmarks
            else:
                right_hand_data = flat_landmarks
    return left_hand_data, right_hand_data, results.multi_hand_landmarks, hands_detected

# Function to make predictions
def predict_sign(image):
    left_landmarks, right_landmarks, multi_hand_landmarks, hands_detected = extract_hand_landmarks(image)
    left_angles = compute_cosineSim_angles(left_landmarks, left_triplets)
    right_angles = compute_cosineSim_angles(right_landmarks, right_triplets)
    features = np.array(left_angles + right_angles).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label, np.max(prediction), multi_hand_landmarks, hands_detected

def serialize_landmarks(multi_hand_landmarks):
    if not multi_hand_landmarks:
        return []
    serialized = []
    for hand_landmarks in multi_hand_landmarks:
        landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        serialized.append(landmarks)
    return serialized

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"  # Ensure this file exists
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_and_expand_text(text):
    """
    Correct and expand the text using SymSpell.
    """
    # First, check if the text is a short form and expand it
    text_lower = text.lower()
    if text_lower in SHORT_FORMS:
        return SHORT_FORMS[text_lower]

    # Use SymSpell for spell checking
    suggestions = sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term  # Return the closest corrected word
    return text  # Return the original text if no correction is found

# Enhanced SHORT_FORMS dictionary with language support
SHORT_FORMS = {
    "gm": {"en": "good morning", "mr": "शुभ सकाळ"},
    "gn": {"en": "good night", "mr": "शुभ रात्री"},
    "ty": {"en": "thank you", "mr": "धन्यवाद"},
    # Add more short forms as needed
}

@app.route('/expand_shortcut', methods=['POST'])
def expand_shortcut():
    data = request.json
    text = data.get("text", "")
    language = data.get("language", "en")
    
    words = text.lower().split()
    expanded_sentence = []
    
    for word in words:
        if word in SHORT_FORMS and language in SHORT_FORMS[word]:
            expanded_sentence.append(SHORT_FORMS[word][language])
        else:
            expanded_sentence.append(word)
    
    return jsonify({"expanded_text": " ".join(expanded_sentence)})
    
# Add to your app configuration
TRANSLATORS = {
    'mr': Translator(to_lang='mr')  # Marathi translator
}

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get("text", "")
    target_lang = data.get("language", "mr")
    
    if target_lang in TRANSLATORS:
        try:
            translator = TRANSLATORS[target_lang]
            translated_text = translator.translate(text)
            return jsonify({"translated_text": translated_text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Language not supported"}), 400
    
# State variables
last_capture_time = time.time()
CAPTURE_DELAY = 2.0  # 3 seconds delay between captures
is_translating = False  # State to track if translation is active
translation_mode = "letter"  # Default mode: letter, number, word, sentence
current_sentence = ""  # Store the current sentence being formed
frame_counter = 0  # Initialize frame counter

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    data = request.json
    text = data.get("text", "")
    language = data.get("language", "en-in")  # Use 'en-in' for Indian English
    
    print(f"Generating audio for text: {text}, language: {language}")  # Debugging
    
    try:
        # Generate TTS audio
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save the audio to a temporary in-memory buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Encode the audio as base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        print("Audio generated successfully")  # Debugging
        
        return jsonify({
            "audio_base64": audio_base64,
            "status": "success"
        })
    except Exception as e:
        print(f"Error generating audio: {e}")  # Debugging
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/analyze_and_correct', methods=['POST'])
def analyze_and_correct():
    data = request.json
    text = data.get("text", "")
    
    try:
        # Analyze and correct the text using Gemini
        corrected_text = analyze_and_correct_text(text)
        
        return jsonify({
            "corrected_text": corrected_text,
            "status": "success"
        })
    except Exception as e:
        print(f"Error analyzing and correcting text: {e}")  # Debugging
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_translation', methods=['POST'])
def start_translation():
    global is_translating, translation_mode, current_sentence
    is_translating = True
    current_sentence = ""  # Reset sentence when starting new translation
    print("Translation started")  # Debugging
    return jsonify({
        'status': 'Translation started',
        'mode': translation_mode,
        'is_translating': True
    })

@app.route('/stop_translation', methods=['POST'])
def stop_translation():
    global is_translating, current_sentence
    is_translating = False
    return jsonify({
        'status': 'Translation stopped',
        'final_sentence': current_sentence,
        'is_translating': False
    })

@app.route('/update_ui', methods=['GET'])
def update_ui():
    global translation_mode, is_translating
    return jsonify({
        'show_buttons': translation_mode == "sentence",
        'is_translating': is_translating
    })

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global translation_mode
    data = request.json
    if 'mode' not in data:
        return jsonify({'error': 'No mode provided'}), 400
    translation_mode = data['mode']
    return jsonify({'status': f'Mode set to {translation_mode}'})

@app.route('/add_space', methods=['POST'])
def add_space():
    global current_sentence
    current_sentence += " "
    return jsonify({'status': 'Space added', 'sentence': current_sentence})

@app.route('/add_fullstop', methods=['POST'])
def add_fullstop():
    global current_sentence
    current_sentence += "."
    return jsonify({'status': 'Full stop added', 'sentence': current_sentence})

@app.route('/end_sentence', methods=['POST'])
def end_sentence():
    global current_sentence
    final_sentence = current_sentence
    current_sentence = ""
    return jsonify({'status': 'Sentence ended', 'final_sentence': final_sentence})

@app.route('/toggle_mesh', methods=['POST'])
def toggle_mesh():
    global show_mesh
    show_mesh = not show_mesh
    return jsonify({
        'status': 'success',
        'show_mesh': show_mesh
    })

@app.route('/predict', methods=['POST'])
def predict():
    global last_capture_time, is_translating, current_sentence, translation_mode, frame_counter
    
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        img_str = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Skip first few frames in all modes
        if frame_counter < 5:  # Skip first 5 frames
            frame_counter += 1
            return jsonify({
                'prediction': '',
                'confidence': 0.0,
                'captured': False
            })
        
        # Extract hand landmarks and check if hands are detected
        left_landmarks, right_landmarks, multi_hand_landmarks, hands_detected = extract_hand_landmarks(frame)
        
        # If no hands are detected and in sentence mode, treat it as a space
        if not hands_detected and translation_mode == "sentence":
            current_sentence += " "
            return jsonify({
                'prediction': ' ',
                'confidence': 0.0,
                'captured': True,
                'current_sentence': current_sentence
            })
        
        # Get prediction using your existing model
        predicted_label, confidence, _, _ = predict_sign(frame)
        current_time = time.time()
        
        response = {
            'prediction': predicted_label,
            'confidence': float(confidence),
            'captured': False,
            'multi_hand_landmarks': serialize_landmarks(multi_hand_landmarks)  # Serialized landmarks
        }
        
        if is_translating and (current_time - last_capture_time >= CAPTURE_DELAY):
            last_capture_time = current_time
            response['captured'] = True
            
            if translation_mode in ["word", "sentence"]:
                # Correct and expand the predicted text
                corrected_text = correct_and_expand_text(predicted_label)
                current_sentence += corrected_text + " "  # Add space after each word
                response['current_sentence'] = current_sentence
                
        print("Prediction:", response)  # Debugging
        return jsonify(response)
    except Exception as e:
        print(f"Error in /predict: {e}")  # Debugging
        return jsonify({'error': str(e)}), 500
    
SHORT_FORMS = {
    "gm": "good morning",
    "gn": "good night",
    "ty": "thank you",
    "plz": "please",
    "btw": "by the way",
    "omg": "oh my god",
    "idk": "i don't know",
    "lol": "laugh out loud",
    "brb": "be right back",
    "np": "no problem",
    "asap": "as soon as possible",
    "ttyl": "talk to you later",
    "imo": "in my opinion",
    "btw": "by the way",
    "afaik": "as far as i know",
    "fyi": "for your information",
    "rofl": "rolling on the floor laughing",
    "wtf": "what the heck",
    "omw": "on my way",
    "idc": "i don't care",
    "irl": "in real life",
    "tbh": "to be honest",
    "jk": "just kidding",
    "ily": "i love you",
    "smh": "shaking my head",
    "nvm": "never mind",
    "thx": "thanks",
    "yw": "you're welcome",
    "gg": "good game",
    "gl": "good luck",
    "hf": "have fun",
    "wb": "welcome back",
    "cya": "see you",
    "bbl": "be back later",
    "gtg": "got to go",
    "hmu": "hit me up",
    "rn": "right now",
    "tmi": "too much information",
    "wbu": "what about you",
    "wyd": "what you doing",
    "wya": "where you at",
    "yolo": "you only live once",
    "fomo": "fear of missing out",
    "fwiw": "for what it's worth",
    "icymi": "in case you missed it",
    "tmi": "too much information",
    "smh": "shaking my head",
    "nbd": "no big deal",
    "ofc": "of course",
    "pov": "point of view",
    "srsly": "seriously",
    "tbf": "to be fair",
    "tgif": "thank god it's friday",
    "tmi": "too much information",
    "wo" : "without"
}

if __name__ == '__main__':
    app.run(debug=True)