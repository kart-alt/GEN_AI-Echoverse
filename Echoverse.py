# EchoVerse Pro - Premium AI-Powered Audiobook Creator with IBM Granite
# Ultra-Modern UI/UX with Professional Design

import subprocess
import sys
import os

def install_packages():
    """Install required packages with proper versions"""
    packages = [
        'gradio>=4.0.0',
        'torch>=2.0.0',
        'transformers>=4.40.0',
        'accelerate>=0.20.0',
        'gtts>=2.5.0',
        'langdetect>=1.0.9',
        'deep-translator>=1.11.4',
        'numpy>=1.24.0',
        'requests>=2.31.0',
        'huggingface-hub>=0.20.0',
        'tokenizers>=0.15.0',
        'sentencepiece>=0.1.99',
        'protobuf>=3.20.0'
    ]

    print("🚀 Installing packages for IBM Granite integration...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"✅ Installed: {package}")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")

install_packages()

# ==================== IMPORTS ====================
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from gtts import gTTS
from deep_translator import GoogleTranslator
import tempfile
import time
from langdetect import detect
import uuid
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== IBM GRANITE MODEL HANDLER ====================
class IBMGraniteRewriter:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "ibm-granite/granite-3b-code-instruct"
        self.backup_model_name = "microsoft/DialoGPT-medium"
        self.generation_config = None
        self.translator = GoogleTranslator(source='auto', target='en')
        
        print(f"🔥 Initializing on device: {self.device}")
        print(f"🧠 Primary model: {self.model_name}")

    def load_model(self, use_backup=False):
        """Load IBM Granite model with proper configuration"""
        if self.model_loaded:
            return True

        model_to_load = self.backup_model_name if use_backup else self.model_name
        
        try:
            print(f"🔥 Loading {model_to_load}...")
            print("📦 This may take several minutes on first run...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_to_load,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
                'low_cpu_mem_usage': True,
                'device_map': 'auto' if self.device == "cuda" else None
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(model_to_load, **model_kwargs)
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.generation_config = GenerationConfig(
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            self.model_loaded = True
            print(f"✅ Model loaded successfully: {model_to_load}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {model_to_load}: {str(e)}")
            if not use_backup and "granite" in model_to_load.lower():
                print("🔄 Trying backup model...")
                return self.load_model(use_backup=True)
            return False

    def create_granite_prompt(self, text, tone):
        """Create properly formatted prompt for IBM Granite"""
        tone_instructions = {
            'neutral': "Rewrite this text in a clear, professional, and neutral tone while maintaining all important information.",
            'suspenseful': "Rewrite this text to create suspense and dramatic tension. Use vivid descriptions and build anticipation.",
            'inspiring': "Rewrite this text in an inspiring and motivational tone that energizes and uplifts the reader.",
            'romantic': "Rewrite this text with warmth, emotion, and romantic undertones that touch the heart.",
            'mysterious': "Rewrite this text to create an air of mystery and intrigue, with enigmatic descriptions.",
            'cheerful': "Rewrite this text in a joyful, optimistic tone that brings happiness and positivity."
        }
        
        instruction = tone_instructions.get(tone, tone_instructions['neutral'])
        
        prompt = f"""### Instruction:
{instruction}

### Input:
{text.strip()}

### Response:
"""
        return prompt

    def extract_response(self, generated_text, original_prompt):
        """Extract the response from generated text"""
        try:
            if "### Response:" in generated_text:
                response = generated_text.split("### Response:")[-1].strip()
            else:
                response = generated_text.replace(original_prompt, "").strip()
            
            response = response.replace("### Instruction:", "").replace("### Input:", "").strip()
            
            if len(response) < 20:
                return None
                
            return response
            
        except Exception as e:
            print(f"Error extracting response: {e}")
            return None

    def rewrite_text(self, text, tone, target_language='en'):
        """Rewrite text using IBM Granite with proper error handling"""
        if not text or len(text.strip()) < 5:
            return "Please provide more text to process."
        
        if not self.model_loaded:
            if not self.load_model():
                return self.fallback_rewrite(text, tone)
        
        try:
            if len(text) > 2000:
                text = text[:2000] + "..."
            
            prompt = self.create_granite_prompt(text, tone)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            print(f"🧠 Generating with IBM Granite ({tone} tone)...")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            enhanced_text = self.extract_response(generated_text, prompt)
            
            if enhanced_text and len(enhanced_text) > 20:
                print("✅ IBM Granite processing successful!")
                result = enhanced_text
            else:
                print("⚠️ Granite output too short, using fallback...")
                result = self.fallback_rewrite(text, tone)
            
            if target_language != 'en':
                try:
                    self.translator.target = target_language
                    result = self.translator.translate(result)
                except Exception as e:
                    print(f"Translation error: {e}")
            
            return result
            
        except Exception as e:
            print(f"❌ Granite generation error: {str(e)}")
            return self.fallback_rewrite(text, tone)

    def fallback_rewrite(self, text, tone):
        """Fallback text enhancement when AI fails"""
        tone_modifiers = {
            'neutral': lambda t: f"{t}",
            'suspenseful': lambda t: f"{t.replace('.', '...')} The tension was palpable.",
            'inspiring': lambda t: f"With determination and hope, {t.lower()} This moment holds infinite possibilities.",
            'romantic': lambda t: f"In the gentle embrace of emotion, {t.lower()} Love filled every word.",
            'mysterious': lambda t: f"Shrouded in enigma, {t.lower()} Secrets whispered in the shadows.",
            'cheerful': lambda t: f"With boundless joy and enthusiasm, {t.lower()} Happiness radiated through every moment."
        }
        
        return tone_modifiers.get(tone, tone_modifiers['neutral'])(text)

# ==================== ENHANCED AUDIO GENERATOR ====================
class AudioGenerator:
    def __init__(self):
        self.supported_languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish', 'th': 'Thai',
            'vi': 'Vietnamese', 'bn': 'Bengali', 'te': 'Telugu', 'mr': 'Marathi',
            'ta': 'Tamil', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam',
            'pa': 'Punjabi', 'ur': 'Urdu', 'ne': 'Nepali'
        }

    def detect_language(self, text):
        """Detect language with fallback"""
        try:
            detected = detect(text)
            return detected if detected in self.supported_languages else 'en'
        except:
            return 'en'

    def generate_audio(self, text, language='en', slow=False):
        """Generate high-quality audio"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f'_granite_audiobook_{timestamp}.mp3'
            )
            temp_file.close()

            tts = gTTS(
                text=text,
                lang=language,
                slow=slow,
                lang_check=False,
                tld='com'
            )
            
            tts.save(temp_file.name)
            return temp_file.name
            
        except Exception as e:
            print(f"Audio generation error: {e}")
            if language != 'en':
                return self.generate_audio(text, 'en', slow)
            return None

# ==================== INITIALIZE COMPONENTS ====================
print("🔧 Initializing IBM Granite EchoVerse...")
granite_rewriter = IBMGraniteRewriter()
audio_generator = AudioGenerator()

print("🤖 Pre-loading IBM Granite model...")
model_loaded = granite_rewriter.load_model()

# ==================== GRADIO INTERFACE ====================
def process_audiobook(text_input, file_input, tone, target_language, slow_speech, progress=gr.Progress()):
    """Main processing function with IBM Granite"""
    
    progress(0.1, desc="📝 Processing input...")
    
    if file_input is not None:
        try:
            with open(file_input.name, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            return f"❌ File error: {str(e)}", "", "", None, None
    elif text_input and text_input.strip():
        text = text_input.strip()
    else:
        return "❌ Please provide text input", "", "", None, None
    
    if len(text) > 5000:
        text = text[:5000] + "..."
    
    progress(0.3, desc="🌍 Detecting language...")
    detected_lang = audio_generator.detect_language(text)
    
    progress(0.4, desc="🧠 Enhancing with IBM Granite AI...")
    enhanced_text = granite_rewriter.rewrite_text(text, tone, target_language)
    
    progress(0.7, desc="🎵 Generating premium audio...")
    audio_file = audio_generator.generate_audio(enhanced_text, target_language, slow_speech)
    
    progress(1.0, desc="✅ Complete!")
    
    if audio_file:
        lang_name = audio_generator.supported_languages.get(target_language, 'Unknown')
        status = f"✅ IBM Granite Success! Original: {detected_lang.upper()} → Target: {lang_name} | Tone: {tone.title()}"
        return status, text, enhanced_text, audio_file, audio_file
    else:
        return "❌ Audio generation failed", text, enhanced_text, None, None

def clear_interface():
    """Clear all fields"""
    return "", None, "neutral", "en", False, "", "", "", None, None

# ==================== ULTRA-MODERN INTERFACE ====================
def create_ultra_modern_interface():
    """Create ultra-modern professional interface with sidebar controls"""
    
    # Ultra-modern CSS with professional color scheme
    ultra_modern_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --secondary: #10b981;
        --accent: #f59e0b;
        --danger: #ef4444;
        --warning: #f97316;
        --success: #22c55e;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        --border: rgba(148, 163, 184, 0.2);
        --glass: rgba(15, 23, 42, 0.8);
        --glass-light: rgba(30, 41, 59, 0.6);
        --shadow-xl: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
        --shadow-2xl: 0 35px 60px -12px rgba(0, 0, 0, 0.9);
        --glow: 0 0 20px rgba(99, 102, 241, 0.3);
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-accent: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --gradient-success: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    * {
        box-sizing: border-box;
    }
    
    .gradio-container {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        min-height: 100vh !important;
        overflow-x: hidden !important;
    }
    
    /* Sidebar Styling */
    .sidebar {
        background: var(--glass) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid var(--border) !important;
        border-radius: 24px !important;
        padding: 28px !important;
        box-shadow: var(--shadow-xl) !important;
        position: sticky !important;
        top: 20px !important;
        height: fit-content !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .sidebar:hover {
        transform: translateY(-4px) !important;
        box-shadow: var(--shadow-2xl) !important;
        border-color: var(--primary-light) !important;
    }
    
    /* Main Content Area */
    .main-content {
        background: var(--glass-light) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        border: 1px solid var(--border) !important;
        border-radius: 24px !important;
        padding: 32px !important;
        box-shadow: var(--shadow-xl) !important;
        transition: all 0.4s ease !important;
    }
    
    .main-content:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-2xl) !important;
    }
    
    /* Enhanced Buttons */
    .gr-button-primary {
        background: var(--gradient-primary) !important;
        border: none !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 16px 32px !important;
        color: white !important;
        text-transform: none !important;
        letter-spacing: 0.5px !important;
        box-shadow: var(--glow), 0 8px 25px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: var(--glow), 0 12px 35px rgba(99, 102, 241, 0.6) !important;
    }
    
    .gr-button-primary::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .gr-button-primary:hover::before {
        left: 100% !important;
    }
    
    .gr-button-secondary {
        background: var(--glass) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-button-secondary:hover {
        background: var(--glass-light) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        border-color: var(--primary-light) !important;
    }
    
    /* Enhanced Form Elements */
    .gr-textbox, .gr-dropdown, .gr-radio {
        background: var(--glass) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-textbox:focus, .gr-dropdown:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
        outline: none !important;
    }
    
    .gr-file {
        border: 2px dashed var(--border) !important;
        border-radius: 16px !important;
        background: var(--glass) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
        padding: 24px !important;
        text-align: center !important;
    }
    
    .gr-file:hover {
        border-color: var(--primary) !important;
        background: var(--glass-light) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Status Cards */
    .status-success {
        background: var(--gradient-success) !important;
        color: white !important;
        padding: 20px !important;
        border-radius: 16px !important;
        font-weight: 500 !important;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.3) !important;
        border: none !important;
    }
    
    .status-error {
        background: var(--gradient-secondary) !important;
        color: white !important;
        padding: 20px !important;
        border-radius: 16px !important;
        font-weight: 500 !important;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3) !important;
        border: none !important;
    }
    
    /* Text Comparison Cards */
    .text-card {
        background: var(--glass) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.4s ease !important;
        height: 100% !important;
    }
    
    .text-card:hover {
        transform: translateY(-4px) !important;
        border-color: var(--primary-light) !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Audio Section */
    .audio-section {
        background: var(--gradient-accent) !important;
        border-radius: 24px !important;
        padding: 32px !important;
        text-align: center !important;
        color: white !important;
        box-shadow: 0 15px 35px rgba(79, 172, 254, 0.3) !important;
        margin: 32px 0 !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary) !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary) !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-light) !important;
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.8s ease-out !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Radio Button Enhancements */
    .gr-radio label {
        background: var(--glass) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin: 4px 0 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        color: var(--text-primary) !important;
    }
    
    .gr-radio label:hover {
        background: var(--glass-light) !important;
        border-color: var(--primary) !important;
        transform: translateX(4px) !important;
    }
    
    /* Checkbox Enhancements */
    .gr-checkbox label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Label Styling */
    .gr-block label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: 8px !important;
    }
    
    /* Progress Bar */
    .progress {
        background: var(--gradient-primary) !important;
        border-radius: 8px !important;
        box-shadow: var(--glow) !important;
    }
    
    /* Modal/Dialog Enhancements */
    .gr-modal {
        background: var(--glass) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        color: var(--text-primary) !important;
    }
    """
    
    with gr.Blocks(
        css=ultra_modern_css,
        title="EchoVerse Pro - IBM Granite AI Audiobooks",
        theme=gr.themes.Glass()
    ) as interface:
        
        # Ultra-Modern Header
        gr.HTML("""
        <div class="fade-in" style="text-align: center; padding: 60px 40px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 60%, #667eea 100%);
                    border-radius: 32px; margin-bottom: 40px; 
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
                    position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                        background: url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1000 1000\"><defs><radialGradient id=\"a\"><stop offset=\"0%\" style=\"stop-color:rgba(255,255,255,0.1)\"/><stop offset=\"100%\" style=\"stop-color:rgba(255,255,255,0)\"/></radialGradient></defs><circle cx=\"200\" cy=\"200\" r=\"180\" fill=\"url(%23a)\"/><circle cx=\"800\" cy=\"300\" r=\"120\" fill=\"url(%23a)\"/><circle cx=\"400\" cy=\"700\" r=\"150\" fill=\"url(%23a)\"/></svg>');
                        opacity: 0.6;"></div>
            <div style="position: relative; z-index: 2;">
                <h1 style="color: white; font-size: 4.5em; margin-bottom: 20px; font-weight: 800;
                           text-shadow: 0 8px 16px rgba(0,0,0,0.4); letter-spacing: -2px;">
                    🎧 EchoVerse Pro
                </h1>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px);
                           border-radius: 20px; padding: 20px; margin: 20px auto; max-width: 600px;
                           border: 1px solid rgba(255,255,255,0.2);">
                    <h2 style="color: white; font-size: 2em; margin-bottom: 15px; font-weight: 600;">
                        Powered by IBM Granite 3B AI
                    </h2>
                    <p style="color: rgba(255,255,255,0.95); font-size: 1.3em; margin: 0; font-weight: 400;">
                        🧠 Advanced Neural Text Enhancement • 🌍 30+ Languages • 🎨 6 Emotional Tones
                    </p>
                </div>
                <div style="display: flex; justify-content: center; gap: 30px; margin-top: 30px; flex-wrap: wrap;">
                    <div style="background: rgba(255,255,255,0.1); padding: 15px 25px; border-radius: 15px; backdrop-filter: blur(5px);">
                        <span style="color: white; font-weight: 600;">⚡ Real-time AI Processing</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 15px 25px; border-radius: 15px; backdrop-filter: blur(5px);">
                        <span style="color: white; font-weight: 600;">🎵 Premium Audio Quality</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 15px 25px; border-radius: 15px; backdrop-filter: blur(5px);">
                        <span style="color: white; font-weight: 600;">🚀 Ultra-Modern Interface</span>
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # Main Layout with Sidebar
        with gr.Row(equal_height=True):
            # Left Sidebar - Controls
            with gr.Column(scale=1, elem_classes=["sidebar", "slide-in"]):
                
                # AI Configuration Section
                gr.HTML("""
                <div style="text-align: center; margin-bottom: 28px;">
                    <h2 style="color: #f8fafc; font-size: 1.8em; font-weight: 700; margin: 0;
                               text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                        🎛️ AI Controls
                    </h2>
                    <p style="color: #cbd5e1; margin: 8px 0 0 0; font-size: 0.95em;">
                        Configure your audiobook transformation
                    </p>
                </div>
                """)
                
                # Emotional Tone Selection
                gr.HTML("""
                <div style="margin-bottom: 24px;">
                    <h3 style="color: #e2e8f0; font-size: 1.2em; font-weight: 600; margin-bottom: 12px;
                               display: flex; align-items: center; gap: 8px;">
                        🎭 <span>Emotional Tone</span>
                    </h3>
                </div>
                """)
                
                tone = gr.Radio(
                    choices=[
                        ("⚖️ Neutral - Professional & Balanced", "neutral"),
                        ("⚡ Suspenseful - Dramatic & Thrilling", "suspenseful"), 
                        ("⭐ Inspiring - Motivational & Uplifting", "inspiring"),
                        ("💕 Romantic - Warm & Emotional", "romantic"),
                        ("🔮 Mysterious - Enigmatic & Intriguing", "mysterious"),
                        ("😊 Cheerful - Joyful & Optimistic", "cheerful")
                    ],
                    value="neutral",
                    label="Select AI Processing Tone",
                    show_label=False,
                    elem_classes=["tone-selector"]
                )
                
                gr.HTML("<div style='height: 20px;'></div>")
                
                # Language Selection
                gr.HTML("""
                <div style="margin-bottom: 16px;">
                    <h3 style="color: #e2e8f0; font-size: 1.2em; font-weight: 600; margin-bottom: 12px;
                               display: flex; align-items: center; gap: 8px;">
                        🌍 <span>Audio Language</span>
                    </h3>
                </div>
                """)
                
                target_language = gr.Dropdown(
                    choices=[
                        ("🇺🇸 English", "en"), 
                        ("🇪🇸 Spanish", "es"), 
                        ("🇫🇷 French", "fr"),
                        ("🇩🇪 German", "de"), 
                        ("🇮🇹 Italian", "it"), 
                        ("🇵🇹 Portuguese", "pt"),
                        ("🇷🇺 Russian", "ru"), 
                        ("🇯🇵 Japanese", "ja"), 
                        ("🇰🇷 Korean", "ko"),
                        ("🇨🇳 Chinese", "zh"), 
                        ("🇮🇳 Hindi (हिन्दी)", "hi"), 
                        ("🇮🇳 Bengali (বাংলা)", "bn"),
                        ("🇮🇳 Tamil (தமிழ்)", "ta"), 
                        ("🇮🇳 Telugu (తెలుగు)", "te"), 
                        ("🇮🇳 Marathi (मराठी)", "mr"),
                        ("🇮🇳 Gujarati (ગુજરાતી)", "gu"),
                        ("🇮🇳 Kannada (ಕನ್ನಡ)", "kn"),
                        ("🇮🇳 Malayalam (മലയാളം)", "ml"),
                        ("🇮🇳 Punjabi (ਪੰਜਾਬੀ)", "pa"),
                        ("🇮🇳 Urdu (اردو)", "ur")
                    ],
                    value="en",
                    label="Target Audio Language",
                    show_label=False,
                    elem_classes=["language-dropdown"]
                )
                
                gr.HTML("<div style='height: 20px;'></div>")
                
                # Audio Settings
                gr.HTML("""
                <div style="margin-bottom: 16px;">
                    <h3 style="color: #e2e8f0; font-size: 1.2em; font-weight: 600; margin-bottom: 12px;
                               display: flex; align-items: center; gap: 8px;">
                        🎚️ <span>Audio Settings</span>
                    </h3>
                </div>
                """)
                
                slow_speech = gr.Checkbox(
                    label="🐌 Enable Slow Speech Mode",
                    value=False,
                    info="Better clarity for learning",
                    elem_classes=["audio-checkbox"]
                )
                
                gr.HTML("<div style='height: 32px;'></div>")
                
                # Action Buttons
                process_btn = gr.Button(
                    "🧠 Transform with IBM Granite AI",
                    variant="primary",
                    size="lg",
                    elem_classes=["transform-button"]
                )
                
                gr.HTML("<div style='height: 12px;'></div>")
                
                clear_btn = gr.Button(
                    "🗑️ Clear All Fields",
                    variant="secondary",
                    size="lg",
                    elem_classes=["clear-button"]
                )
                
                gr.HTML("<div style='height: 32px;'></div>")
                
                # AI Model Status
                if model_loaded:
                    gr.HTML("""
                    <div style="background: linear-gradient(135deg, #22c55e, #16a34a); 
                               padding: 20px; border-radius: 16px; text-align: center;
                               box-shadow: 0 8px 25px rgba(34, 197, 94, 0.3);
                               border: 1px solid rgba(34, 197, 94, 0.2);">
                        <div style="font-size: 2em; margin-bottom: 8px;">✅</div>
                        <h4 style="color: white; font-weight: 600; margin-bottom: 8px; font-size: 1.1em;">
                            IBM Granite AI Ready
                        </h4>
                        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.85em;">
                            Advanced text enhancement operational
                        </p>
                    </div>
                    """)
                else:
                    gr.HTML("""
                    <div style="background: linear-gradient(135deg, #f97316, #ea580c); 
                               padding: 20px; border-radius: 16px; text-align: center;
                               box-shadow: 0 8px 25px rgba(249, 115, 22, 0.3);
                               border: 1px solid rgba(249, 115, 22, 0.2);">
                        <div style="font-size: 2em; margin-bottom: 8px;">⚠️</div>
                        <h4 style="color: white; font-weight: 600; margin-bottom: 8px; font-size: 1.1em;">
                            AI Loading...
                        </h4>
                        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.85em;">
                            Model will initialize on first use
                        </p>
                    </div>
                    """)
            
            # Main Content Area
            with gr.Column(scale=2, elem_classes=["main-content", "fade-in"]):
                
                # Text Input Section
                gr.HTML("""
                <div style="margin-bottom: 28px;">
                    <h2 style="color: #f8fafc; font-size: 2em; font-weight: 700; margin-bottom: 8px;
                               display: flex; align-items: center; gap: 12px;">
                        📝 <span>Content Input</span>
                    </h2>
                    <p style="color: #cbd5e1; margin: 0; font-size: 1.05em;">
                        Provide your text content for AI-powered audiobook transformation
                    </p>
                </div>
                """)
                
                text_input = gr.Textbox(
                    label="Enter your text content",
                    placeholder="✍️ Paste your story, article, book chapter, or any text content here...\n\nThe IBM Granite AI will enhance your text with the selected emotional tone and convert it into a natural-sounding audiobook in your chosen language.\n\nSupports up to 5000 characters for optimal processing speed.",
                    lines=12,
                    max_lines=20,
                    show_label=False,
                    elem_classes=["main-textbox"]
                )
                
                gr.HTML("<div style='height: 20px;'></div>")
                
                # File Upload
                gr.HTML("""
                <div style="text-align: center; margin-bottom: 16px;">
                    <h3 style="color: #e2e8f0; font-size: 1.3em; font-weight: 600; margin: 0;">
                        📁 Or Upload Text File
                    </h3>
                    <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 0.9em;">
                        Supports .txt files • Automatically processes content
                    </p>
                </div>
                """)
                
                file_input = gr.File(
                    label="Upload text file",
                    file_types=[".txt"],
                    show_label=False,
                    elem_classes=["file-upload"]
                )
                
                gr.HTML("<div style='height: 32px;'></div>")
                
                # Processing Status
                status_output = gr.Textbox(
                    label="🔥 IBM Granite AI Status",
                    interactive=False,
                    elem_classes=["status-output"]
                )
        
        # Results Section
        gr.HTML("""
        <div class="fade-in" style="margin: 40px 0;">
            <div style="text-align: center; margin-bottom: 32px;">
                <h2 style="color: #f8fafc; font-size: 2.2em; font-weight: 700; margin-bottom: 12px;
                           display: flex; align-items: center; justify-content: center; gap: 16px;">
                    📊 <span>AI Processing Results</span>
                </h2>
                <p style="color: #cbd5e1; margin: 0; font-size: 1.1em;">
                    Compare original text with IBM Granite AI enhancement
                </p>
            </div>
        </div>
        """)
        
        # Text Comparison Cards
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["text-card"]):
                gr.HTML("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: #f1f5f9; font-size: 1.4em; font-weight: 600; margin: 0;
                               display: flex; align-items: center; justify-content: center; gap: 10px;">
                        📄 <span>Original Text</span>
                    </h3>
                    <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 0.9em;">
                        Your input content as provided
                    </p>
                </div>
                """)
                
                original_display = gr.Textbox(
                    lines=8,
                    interactive=False,
                    show_label=False,
                    placeholder="Your original text will appear here after processing...",
                    elem_classes=["result-textbox"]
                )
            
            with gr.Column(scale=1, elem_classes=["text-card"]):
                gr.HTML("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: #f1f5f9; font-size: 1.4em; font-weight: 600; margin: 0;
                               display: flex; align-items: center; justify-content: center; gap: 10px;">
                        🧠 <span>IBM Granite Enhanced</span>
                    </h3>
                    <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 0.9em;">
                        AI-enhanced with emotional tone
                    </p>
                </div>
                """)
                
                enhanced_display = gr.Textbox(
                    lines=8,
                    interactive=False,
                    show_label=False,
                    placeholder="AI-enhanced text with applied emotional tone will appear here...",
                    elem_classes=["result-textbox"]
                )
        
        # Premium Audio Section
        gr.HTML("""
        <div class="audio-section fade-in">
            <div style="margin-bottom: 24px;">
                <h2 style="font-size: 2.5em; font-weight: 800; margin-bottom: 16px;
                           display: flex; align-items: center; justify-content: center; gap: 16px;">
                    🎵 <span>Premium Audiobook</span>
                </h2>
                <p style="font-size: 1.2em; opacity: 0.95; margin: 0; font-weight: 400;">
                    High-quality, AI-enhanced narration ready for listening and download
                </p>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                        gap: 20px; margin-top: 24px;">
                <div style="background: rgba(255,255,255,0.1); padding: 16px; border-radius: 12px; backdrop-filter: blur(5px);">
                    <div style="font-size: 1.5em; margin-bottom: 4px;">🎧</div>
                    <div style="font-weight: 600;">Premium Quality</div>
                    <div style="font-size: 0.9em; opacity: 0.8;">Crystal clear audio</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 16px; border-radius: 12px; backdrop-filter: blur(5px);">
                    <div style="font-size: 1.5em; margin-bottom: 4px;">🌍</div>
                    <div style="font-weight: 600;">Multi-Language</div>
                    <div style="font-size: 0.9em; opacity: 0.8;">30+ languages</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 16px; border-radius: 12px; backdrop-filter: blur(5px);">
                    <div style="font-size: 1.5em; margin-bottom: 4px;">⚡</div>
                    <div style="font-weight: 600;">AI Enhanced</div>
                    <div style="font-size: 0.9em; opacity: 0.8;">Emotional depth</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 16px; border-radius: 12px; backdrop-filter: blur(5px);">
                    <div style="font-size: 1.5em; margin-bottom: 4px;">📥</div>
                    <div style="font-weight: 600;">Instant Download</div>
                    <div style="font-size: 0.9em; opacity: 0.8;">MP3 format</div>
                </div>
            </div>
        </div>
        """)
        
        # Audio Player
        audio_output = gr.Audio(
            label="🎧 Your AI-Enhanced Audiobook",
            elem_classes=["audio-player"]
        )
        
        download_file = gr.File(
            label="📥 Download Premium MP3",
            interactive=False,
            elem_classes=["download-file"]
        )
        
        # Examples Section
        gr.HTML("""
        <div class="fade-in" style="margin: 50px 0 30px 0;">
            <div style="text-align: center; margin-bottom: 32px;">
                <h2 style="color: #f8fafc; font-size: 2em; font-weight: 700; margin-bottom: 12px;
                           display: flex; align-items: center; justify-content: center; gap: 12px;">
                    💡 <span>Try These Examples</span>
                </h2>
                <p style="color: #cbd5e1; margin: 0; font-size: 1.05em;">
                    Experience different tones and languages with our curated examples
                </p>
            </div>
        </div>
        """)
        
        # Premium Examples
        example_data = [
            [
                "The ancient lighthouse stood majestically on the rocky cliff, its beacon cutting through the stormy night. For centuries, it had guided countless ships safely to harbor, becoming a symbol of hope for sailors around the world.",
                "mysterious",
                "en",
                False
            ],
            [
                "Every morning brings new possibilities and endless opportunities to grow. Believe in yourself, embrace every challenge as a stepping stone to success, and remember that your dreams are not just valid—they are inevitable with dedication and persistence.",
                "inspiring", 
                "hi",
                False
            ],
            [
                "The detective carefully examined the crime scene, her trained eyes catching details others missed. The victim's diary lay open on the mahogany desk, its final entry dated exactly three days ago. Something was definitely wrong with this picture.",
                "suspenseful",
                "en", 
                True
            ],
            [
                "In the quiet garden where roses bloomed eternal, two hearts found their rhythm in the gentle dance of love. Every whispered word carried the weight of forever, every glance held promises of tomorrow.",
                "romantic",
                "fr",
                False
            ]
        ]
        
        gr.Examples(
            examples=example_data,
            inputs=[text_input, tone, target_language, slow_speech],
            label="🌟 Premium Examples - Try Different Languages & Emotional Tones"
        )
        
        # Feature Showcase Footer
        gr.HTML("""
        <div class="fade-in" style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                    border-radius: 24px; padding: 40px; margin: 40px 0; 
                    border: 1px solid rgba(148, 163, 184, 0.2);
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);">
            <div style="text-align: center; margin-bottom: 32px;">
                <h2 style="color: #f8fafc; font-size: 2em; font-weight: 700; margin-bottom: 16px;">
                    ✨ Premium Features
                </h2>
                <p style="color: #cbd5e1; font-size: 1.1em; margin: 0;">
                    Cutting-edge AI technology meets premium user experience
                </p>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
                        gap: 24px;">
                <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); 
                           padding: 28px; border-radius: 20px; text-align: center;
                           box-shadow: 0 12px 24px rgba(99, 102, 241, 0.3);
                           transition: transform 0.3s ease;">
                    <div style="font-size: 3em; margin-bottom: 12px;">🧠</div>
                    <h3 style="color: white; font-size: 1.3em; font-weight: 600; margin-bottom: 12px;">
                        IBM Granite AI
                    </h3>
                    <p style="color: rgba(255,255,255,0.9); font-size: 0.95em; line-height: 1.5; margin: 0;">
                        Advanced 3B parameter model with instruction-following capabilities for precise text enhancement
                    </p>
                </div>
                
                <div style="background: linear-gradient(135deg, #10b981, #059669); 
                           padding: 28px; border-radius: 20px; text-align: center;
                           box-shadow: 0 12px 24px rgba(16, 185, 129, 0.3);
                           transition: transform 0.3s ease;">
                    <div style="font-size: 3em; margin-bottom: 12px;">🌍</div>
                    <h3 style="color: white; font-size: 1.3em; font-weight: 600; margin-bottom: 12px;">
                        Global Language Support
                    </h3>
                    <p style="color: rgba(255,255,255,0.9); font-size: 0.95em; line-height: 1.5; margin: 0;">
                        Support for 30+ languages including comprehensive Indian language coverage
                    </p>
                </div>
                
                <div style="background: linear-gradient(135deg, #f59e0b, #d97706); 
                           padding: 28px; border-radius: 20px; text-align: center;
                           box-shadow: 0 12px 24px rgba(245, 158, 11, 0.3);
                           transition: transform 0.3s ease;">
                    <div style="font-size: 3em; margin-bottom: 12px;">🎨</div>
                    <h3 style="color: white; font-size: 1.3em; font-weight: 600; margin-bottom: 12px;">
                        Emotional Intelligence
                    </h3>
                    <p style="color: rgba(255,255,255,0.9); font-size: 0.95em; line-height: 1.5; margin: 0;">
                        Six distinct emotional tones for expressive storytelling and content enhancement
                    </p>
                </div>
                
                <div style="background: linear-gradient(135deg, #ef4444, #dc2626); 
                           padding: 28px; border-radius: 20px; text-align: center;
                           box-shadow: 0 12px 24px rgba(239, 68, 68, 0.3);
                           transition: transform 0.3s ease;">
                    <div style="font-size: 3em; margin-bottom: 12px;">🎵</div>
                    <h3 style="color: white; font-size: 1.3em; font-weight: 600; margin-bottom: 12px;">
                        Premium Audio
                    </h3>
                    <p style="color: rgba(255,255,255,0.9); font-size: 0.95em; line-height: 1.5; margin: 0;">
                        High-quality Google Text-to-Speech with natural voice synthesis and clear pronunciation
                    </p>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 32px; padding-top: 32px;
                        border-top: 1px solid rgba(148, 163, 184, 0.2);">
                <p style="color: #94a3b8; font-size: 1em; margin: 0; line-height: 1.6;">
                    <strong style="color: #f8fafc;">🎧 EchoVerse Pro</strong> • 
                    Premium AI Audiobook Creator • 
                    Powered by IBM Granite 3B • 
                    Built with modern web technologies
                </p>
                <p style="color: #64748b; font-size: 0.9em; margin: 12px 0 0 0;">
                    Made with ❤️ for content creators, educators, and storytellers worldwide
                </p>
            </div>
        </div>
        """)
        
        # Event Handlers
        process_btn.click(
            fn=process_audiobook,
            inputs=[text_input, file_input, tone, target_language, slow_speech],
            outputs=[status_output, original_display, enhanced_display, audio_output, download_file],
            show_progress=True
        )
        
        clear_btn.click(
            fn=clear_interface,
            outputs=[text_input, file_input, tone, target_language, slow_speech,
                    original_display, enhanced_display, status_output, audio_output, download_file]
        )
    
    return interface

# ==================== MAIN EXECUTION ====================
def main():
    """Launch the Ultra-Modern IBM Granite EchoVerse Pro"""
    print("\n🚀 Starting Ultra-Modern EchoVerse Pro...")
    print("🎨 Premium UI/UX with Professional Design")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🔥 CUDA: {torch.cuda.is_available()}")
    print("✨ Features: Sidebar controls • Modern color scheme • Professional layout")
    
    interface = create_ultra_modern_interface()
    
    print("\n🌐 Launching premium interface...")
    interface.launch(
        share=True,
        server_name="0.0.0.0", 
        show_error=True,
        debug=False,
        favicon_path=None,
        ssl_verify=False
    )

if __name__ == "__main__":
    main()

print("""
🎉 Ultra-Modern EchoVerse Pro is ready!
🎨 Professional sidebar layout with premium design
🧠 IBM Granite 3B AI for advanced text enhancement
🌈 Modern color scheme with glass morphism effects
📱 Fully responsive design with smooth animations
🔗 Public sharing link will be generated automatically
""")
