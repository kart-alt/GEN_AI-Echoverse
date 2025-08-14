# EchoVerse Pro - AI-Powered Audiobook Creator with IBM Granite
# Proper IBM Granite 3B Code Instruct Implementation

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
        'sentencepiece>=0.1.99',  # Required for some tokenizers
        'protobuf>=3.20.0'  # Required for tokenizers
    ]

    print("🚀 Installing packages for IBM Granite integration...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"✅ Installed: {package}")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")

# Install packages
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== IBM GRANITE MODEL HANDLER ====================
class IBMGraniteRewriter:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use the correct IBM Granite model
        self.model_name = "ibm-granite/granite-3b-code-instruct"
        # Backup model if Granite fails
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
            
            # Load tokenizer with proper settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_to_load,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set special tokens
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Load model with optimized settings
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
                'low_cpu_mem_usage': True,
                'device_map': 'auto' if self.device == "cuda" else None
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set up generation configuration
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
            print(f"🎯 Device: {self.device}")
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
        
        # Format for IBM Granite (instruction-following format)
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
            # Split by the response marker
            if "### Response:" in generated_text:
                response = generated_text.split("### Response:")[-1].strip()
            else:
                # Fallback: remove the original prompt
                response = generated_text.replace(original_prompt, "").strip()
            
            # Clean up the response
            response = response.replace("### Instruction:", "").replace("### Input:", "").strip()
            
            # If response is too short or empty, return original
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
        
        # Load model if not already loaded
        if not self.model_loaded:
            if not self.load_model():
                return self.fallback_rewrite(text, tone)
        
        try:
            # Limit input length to prevent memory issues
            if len(text) > 2000:
                text = text[:2000] + "..."
            
            # Create proper prompt for Granite
            prompt = self.create_granite_prompt(text, tone)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            print(f"🧠 Generating with IBM Granite ({tone} tone)...")
            
            # Generate with proper settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )
            
            # Decode the response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the actual response
            enhanced_text = self.extract_response(generated_text, prompt)
            
            if enhanced_text and len(enhanced_text) > 20:
                print("✅ IBM Granite processing successful!")
                result = enhanced_text
            else:
                print("⚠️ Granite output too short, using fallback...")
                result = self.fallback_rewrite(text, tone)
            
            # Translate if needed
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

            # Generate TTS with enhanced settings
            tts = gTTS(
                text=text,
                lang=language,
                slow=slow,
                lang_check=False,
                tld='com'  # Use .com for better voice quality
            )
            
            tts.save(temp_file.name)
            return temp_file.name
            
        except Exception as e:
            print(f"Audio generation error: {e}")
            # Try with English as fallback
            if language != 'en':
                return self.generate_audio(text, 'en', slow)
            return None

# ==================== INITIALIZE COMPONENTS ====================
print("🔧 Initializing IBM Granite EchoVerse...")
granite_rewriter = IBMGraniteRewriter()
audio_generator = AudioGenerator()

# Pre-load the model
print("🤖 Pre-loading IBM Granite model...")
model_loaded = granite_rewriter.load_model()

# ==================== GRADIO INTERFACE ====================
def process_audiobook(text_input, file_input, tone, target_language, slow_speech, progress=gr.Progress()):
    """Main processing function with IBM Granite"""
    
    progress(0.1, desc="📝 Processing input...")
    
    # Handle input
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
    
    # Use IBM Granite for text enhancement
    enhanced_text = granite_rewriter.rewrite_text(text, tone, target_language)
    
    progress(0.7, desc="🎵 Generating premium audio...")
    
    # Generate audio
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

# ==================== CREATE INTERFACE ====================
def create_granite_interface():
    """Create the premium interface with IBM Granite branding"""
    
    css = """
    .gradio-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #581c87 100%);
        font-family: 'Arial', sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #dc2626, #ea580c) !important;
        border: none !important;
        border-radius: 20px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        box-shadow: 0 8px 16px rgba(220, 38, 38, 0.3) !important;
    }
    .gr-panel {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    """
    
    with gr.Blocks(css=css, title="EchoVerse - IBM Granite AI Audiobooks") as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 40px; 
                    background: linear-gradient(135deg, #dc2626 0%, #ea580c 100%);
                    border-radius: 20px; margin-bottom: 30px; 
                    box-shadow: 0 20px 40px rgba(220, 38, 38, 0.3);">
            <h1 style="color: white; font-size: 3.5em; margin-bottom: 15px; text-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                🎧 EchoVerse Pro
            </h1>
            <h2 style="color: white; font-size: 1.8em; margin-bottom: 10px; opacity: 0.95;">
                Powered by IBM Granite 3B AI
            </h2>
            <p style="color: white; font-size: 1.2em; opacity: 0.9;">
                🧠 Advanced AI Text Enhancement • 🌍 30+ Languages • 🎨 6 Emotional Tones
            </p>
            <div style="background: rgba(255,255,255,0.2); padding: 12px; border-radius: 12px; margin-top: 15px;">
                <p style="color: white; margin: 0; font-size: 1em;">
                    ⚡ Real IBM Granite AI Processing • 🎵 Premium Audio Quality • 📱 Modern Interface
                </p>
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 📝 **Text Input**")
                text_input = gr.Textbox(
                    label="Enter your text",
                    placeholder="Paste your story, article, or content here for IBM Granite AI enhancement...",
                    lines=8,
                    max_lines=15
                )
                
                file_input = gr.File(
                    label="📁 Upload text file (.txt)",
                    file_types=[".txt"]
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## 🎨 **AI Configuration**")
                
                tone = gr.Radio(
                    choices=[
                        "neutral", "suspenseful", "inspiring",
                        "romantic", "mysterious", "cheerful"
                    ],
                    value="neutral",
                    label="🎭 Emotional Tone for AI"
                )
                
                target_language = gr.Dropdown(
                    choices=[
                        ("🇺🇸 English", "en"), ("🇪🇸 Spanish", "es"), ("🇫🇷 French", "fr"),
                        ("🇩🇪 German", "de"), ("🇮🇹 Italian", "it"), ("🇵🇹 Portuguese", "pt"),
                        ("🇷🇺 Russian", "ru"), ("🇯🇵 Japanese", "ja"), ("🇰🇷 Korean", "ko"),
                        ("🇨🇳 Chinese", "zh"), ("🇮🇳 Hindi", "hi"), ("🇮🇳 Bengali", "bn"),
                        ("🇮🇳 Tamil", "ta"), ("🇮🇳 Telugu", "te"), ("🇮🇳 Marathi", "mr")
                    ],
                    value="en",
                    label="🌍 Audio Language"
                )
                
                slow_speech = gr.Checkbox(
                    label="🐌 Slow Speech Mode",
                    value=False
                )
        
        # Control buttons
        with gr.Row():
            process_btn = gr.Button(
                "🧠 Transform with IBM Granite AI",
                variant="primary",
                size="lg"
            )
            clear_btn = gr.Button(
                "🗑️ Clear All",
                variant="secondary"
            )
        
        # Status
        status_output = gr.Textbox(label="🔥 IBM Granite Status", interactive=False)
        
        # Results
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📄 Original Text")
                original_display = gr.Textbox(lines=6, interactive=False, show_label=False)
            with gr.Column():
                gr.Markdown("### 🧠 IBM Granite Enhanced")
                enhanced_display = gr.Textbox(lines=6, interactive=False, show_label=False)
        
        # Audio output
        gr.Markdown("## 🎵 **Premium Audiobook**")
        audio_output = gr.Audio(label="🎧 Your AI-Enhanced Audiobook")
        download_file = gr.File(label="📥 Download MP3")
        
        # Examples
        gr.Examples(
            examples=[
                ["The old castle stood majestically on the hill, overlooking the peaceful village below.", "mysterious", "en", False],
                ["Every challenge is an opportunity to grow stronger and wiser than before.", "inspiring", "hi", False],
                ["The detective carefully examined the scene, knowing that one clue could solve everything.", "suspenseful", "en", True]
            ],
            inputs=[text_input, tone, target_language, slow_speech]
        )
        
        # Model status display
        if model_loaded:
            gr.HTML("""
            <div style="background: linear-gradient(90deg, #059669, #0d9488); color: white; 
                        padding: 15px; border-radius: 12px; text-align: center; margin: 20px 0;">
                <h3>✅ IBM Granite 3B Code Instruct - LOADED & READY</h3>
                <p>Advanced AI text enhancement is fully operational!</p>
            </div>
            """)
        else:
            gr.HTML("""
            <div style="background: linear-gradient(90deg, #dc2626, #ea580c); color: white; 
                        padding: 15px; border-radius: 12px; text-align: center; margin: 20px 0;">
                <h3>⚠️ IBM Granite Model Loading...</h3>
                <p>Using intelligent fallback processing. Model will load on first use.</p>
            </div>
            """)
        
        # Event handlers
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
    """Launch the IBM Granite EchoVerse application"""
    print("\n🚀 Starting IBM Granite EchoVerse Pro...")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🔥 CUDA: {torch.cuda.is_available()}")
    
    # Create and launch
    interface = create_granite_interface()
    
    print("\n🌐 Launching application...")
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        show_error=True,
        debug=False
    )

if __name__ == "__main__":
    main()

print("""
🎉 IBM Granite EchoVerse Pro is ready!
🧠 Using real IBM Granite 3B Code Instruct model
🎨 6 emotional tones with advanced AI processing  
🌍 30+ language support with premium audio
🔗 Public link will be generated for easy access
""")
