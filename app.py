import os
import numpy as np
import torch
import gradio as gr  
import spaces
from typing import Optional, Tuple
import whisperx
import gc
from pathlib import Path
import os
import sys
from pathlib import Path

# Add the src directory to the Python path so we can import voxcpm without installing it
src_path = Path(__file__).parent / "src"
if not src_path.exists():
    print(f"Error: src directory not found at {src_path}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)
    
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Remove HF_REPO_ID to prevent automatic downloads
if "HF_REPO_ID" in os.environ:
    del os.environ["HF_REPO_ID"]

try:
    import voxcpm
except ImportError as e:
    print(f"Error importing voxcpm: {e}")
    print("Make sure you're running from the project root directory and the src/ folder exists")
    sys.exit(1)


class VoxCPMDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}")

        # ASR model for prompt text recognition - using WhisperX instead of SenseVoiceSmall
        self.asr_model = None  # Lazy initialization
        self.asr_model_name = "small"  # Using Whisper small model
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        self.batch_size = 16 if self.device == "cuda" else 4

        # TTS model (lazy init)
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self.default_local_model_dir = "./models/VoxCPM-0.5B"

    # ---------- Model helpers ----------
    def _resolve_model_dir(self) -> str:
        """
        Resolve model directory:
        1) Use local weights directory if exists
        2) Use default local model directory if exists
        3) Otherwise, raise error (no automatic downloads)
        """
        # First check if we have safetensors in the weights directory
        weights_dir = "./weights"
        if os.path.isdir(weights_dir):
            # Check if we have the required safetensors files
            required_files = ["model.safetensors", "audiovae.safetensors", "config.json"]
            has_all_files = all(os.path.exists(os.path.join(weights_dir, f)) for f in required_files)
            if has_all_files:
                print(f"Using local safetensors model from: {weights_dir}")
                return weights_dir
        
        # Fall back to default local model directory structure
        if os.path.isdir(self.default_local_model_dir):
            # Check if we have the required files in the traditional format
            required_files = ["pytorch_model.bin", "audiovae.pth", "config.json"]
            has_all_files = all(os.path.exists(os.path.join(self.default_local_model_dir, f)) for f in required_files)
            if has_all_files:
                print(f"Using local pytorch model from: {self.default_local_model_dir}")
                return self.default_local_model_dir
        
        # If we get here, no valid model directory was found
        raise FileNotFoundError(
            f"No local model found. Please ensure you have either:\n"
            f"1. Safetensors files in './weights/' directory: model.safetensors, audiovae.safetensors, config.json\n"
            f"2. OR PyTorch files in '{self.default_local_model_dir}' directory: pytorch_model.bin, audiovae.pth, config.json\n"
            f"The app will not automatically download models from HuggingFace."
        )

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        print("Model not loaded, initializing...")
        model_dir = self._resolve_model_dir()
        print(f"Using model dir: {model_dir}")
        self.voxcpm_model = voxcpm.VoxCPM(voxcpm_model_path=model_dir)
        print("Model loaded successfully.")
        return self.voxcpm_model

    def get_or_load_whisperx(self):
        """Lazy load WhisperX model"""
        if self.asr_model is None:
            print(f"Loading WhisperX {self.asr_model_name} model...")
            self.asr_model = whisperx.load_model(
                self.asr_model_name, 
                self.device, 
                compute_type=self.compute_type
            )
            print("WhisperX model loaded successfully.")
        return self.asr_model

    # ---------- Functional endpoints ----------
    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        
        # Load WhisperX model
        model = self.get_or_load_whisperx()
        
        # Load and transcribe audio
        audio = whisperx.load_audio(prompt_wav)
        result = model.transcribe(audio, batch_size=self.batch_size)
        
        # Extract text from segments
        text = " ".join([segment["text"] for segment in result["segments"]])
        
        # Clean up GPU memory
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        
        return text.strip()

    def generate_tts_audio(
        self,
        text_input: str,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text_input: Optional[str] = None,
        cfg_value_input: float = 2.0,
        inference_timesteps_input: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate speech from text using VoxCPM; optional reference audio for voice style guidance.
        Returns (sample_rate, waveform_numpy)
        """
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        prompt_wav_path = prompt_wav_path_input if prompt_wav_path_input else None
        prompt_text = prompt_text_input if prompt_text_input else None

        print(f"Generating audio for text: '{text[:60]}...'")
        wav = current_model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps_input),
            normalize=do_normalize,
            denoise=denoise,
        )
        return (16000, wav)


# ---------- UI Builders ----------

def create_demo_interface(demo: VoxCPMDemo):
    """Build the Gradio UI for VoxCPM demo."""
    # static assets (logo path)
    gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        ),
        css="""
        .logo-container {
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        }
        .logo-container img {
            height: 80px;
            width: auto;
            max-width: 200px;
            display: inline-block;
        }
        /* Bold accordion labels */
        #acc_quick details > summary,
        #acc_tips details > summary {
            font-weight: 600 !important;
            font-size: 1.1em !important;
        }
        /* Bold labels for specific checkboxes */
        #chk_denoise label,
        #chk_denoise span,
        #chk_normalize label,
        #chk_normalize span {
            font-weight: 600;
        }
        """
    ) as interface:
        # Header logo
        gr.HTML('<div class="logo-container"><img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo"></div>')

        # Quick Start
        with gr.Accordion("📋 Quick Start Guide ｜快速入门", open=False, elem_id="acc_quick"):
            gr.Markdown("""
            ### How to Use ｜使用说明
            1. **(Optional) Provide a Voice Prompt** - Upload or record an audio clip to provide the desired voice characteristics for synthesis.  
               **（可选）提供参考声音** - 上传或录制一段音频，为声音合成提供音色、语调和情感等个性化特征
            2. **(Optional) Enter prompt text** - If you provided a voice prompt, enter the corresponding transcript here (auto-recognition available).  
               **（可选项）输入参考文本** - 如果提供了参考语音，请输入其对应的文本内容（支持自动识别）。
            3. **Enter target text** - Type the text you want the model to speak.  
               **输入目标文本** - 输入您希望模型朗读的文字内容。
            4. **Generate Speech** - Click the "Generate" button to create your audio.  
               **生成语音** - 点击"生成"按钮，即可为您创造出音频。
            """)

        # Pro Tips
        with gr.Accordion("💡 Pro Tips ｜使用建议", open=False, elem_id="acc_tips"):
            gr.Markdown("""
            ### Prompt Speech Enhancement｜参考语音降噪
            - **Enable** to remove background noise for a clean, studio-like voice, with an external ZipEnhancer component.  
              **启用**：通过 ZipEnhancer 组件消除背景噪音，获得更好的音质。
            - **Disable** to preserve the original audio's background atmosphere.  
              **禁用**：保留原始音频的背景环境声，如果想复刻相应声学环境。

            ### Text Normalization｜文本正则化
            - **Enable** to process general text with an external WeTextProcessing component.  
              **启用**：使用 WeTextProcessing 组件，可处理常见文本。
            - **Disable** to use VoxCPM's native text understanding ability. For example, it supports phonemes input ({HH AH0 L OW1}), try it!  
              **禁用**：将使用 VoxCPM 内置的文本理解能力。如，支持音素输入（如 {da4}{jia1}好）和公式符号合成，尝试一下！

            ### CFG Value｜CFG 值
            - **Lower CFG** if the voice prompt sounds strained or expressive.  
              **调低**：如果提示语音听起来不自然或过于夸张。
            - **Higher CFG** for better adherence to the prompt speech style or input text.  
              **调高**：为更好地贴合提示音频的风格或输入文本。

            ### Inference Timesteps｜推理时间步
            - **Lower** for faster synthesis speed.  
              **调低**：合成速度更快。
            - **Higher** for better synthesis quality.  
              **调高**：合成质量更佳。

            ### Long Text (e.g., >5 min speech)｜长文本 (如 >5分钟的合成语音)
            While VoxCPM can handle long texts directly, we recommend using empty lines to break very long content into paragraphs; the model will then synthesize each paragraph individually.  
            虽然 VoxCPM 支持直接生成长文本，但如果目标文本过长，我们建议使用换行符将内容分段；模型将对每个段落分别合成。
            """)

        # Main controls
        with gr.Row():
            with gr.Column():
                prompt_wav = gr.Audio(
                    sources=["upload", 'microphone'],
                    type="filepath",
                    label="Prompt Speech",
                    value="./examples/example.wav",
                )
                DoDenoisePromptAudio = gr.Checkbox(
                    value=False,
                    label="Prompt Speech Enhancement",
                    elem_id="chk_denoise",
                    info="We use ZipEnhancer model to denoise the prompt audio."
                )
                with gr.Row():
                    prompt_text = gr.Textbox(
                        value="Just by listening a few minutes a day, you'll be able to eliminate negative thoughts by conditioning your mind to be more positive.",
                        label="Prompt Text",
                        placeholder="Please enter the prompt text. Automatic recognition is supported, and you can correct the results yourself..."
                    )
                run_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column():
                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="CFG Value (Guidance Scale)",
                    info="Higher values increase adherence to prompt, lower values allow more creativity"
                )
                inference_timesteps = gr.Slider(
                    minimum=4,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Inference Timesteps",
                    info="Number of inference timesteps for generation (higher values may improve quality but slower)"
                )
                with gr.Row():
                    text = gr.Textbox(
                        value="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly realistic speech.",
                        label="Target Text",
                        info="Default processing splits text on \\n into paragraphs; each is synthesized as a chunk and then concatenated into the final audio."
                    )
                with gr.Row():
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label="Text Normalization",
                        elem_id="chk_normalize",
                        info="We use WeTextPorcessing library to normalize the input text."
                    )
                audio_output = gr.Audio(label="Output Audio")

        # Wiring
        run_btn.click(
            fn=demo.generate_tts_audio,
            inputs=[text, prompt_wav, prompt_text, cfg_value, inference_timesteps, DoNormalizeText, DoDenoisePromptAudio],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )
        prompt_wav.change(fn=demo.prompt_wav_recognition, inputs=[prompt_wav], outputs=[prompt_text])

    return interface


def run_demo(server_name: str = "localhost", server_port: int = 7860, show_error: bool = True):
    demo = VoxCPMDemo()
    interface = create_demo_interface(demo)
    # Recommended to enable queue on Spaces for better throughput
    interface.queue(max_size=10).launch(server_name=server_name, server_port=server_port, show_error=show_error)


if __name__ == "__main__":
    run_demo()