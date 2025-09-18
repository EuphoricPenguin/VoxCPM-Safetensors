import torch
import torchaudio
import os
import tempfile
# Removed snapshot_download import since we're disabling automatic downloads
from .model.voxcpm import VoxCPMModel

class VoxCPM:
    def __init__(self,
            voxcpm_model_path : str,
            zipenhancer_model_path : str = "./models/zipenhancer_onnx",
            enable_denoiser : bool = True,
        ):
        """Initialize VoxCPM TTS pipeline.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets
                (weights, configs, etc.). Typically the directory returned by
                a prior download step.
            zipenhancer_model_path: Local path to ONNX model directory for ZipEnhancer.
                If None, denoiser will not be initialized.
            enable_denoiser: Whether to initialize the denoiser pipeline.
        """
        print(f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}")
        self.tts_model = VoxCPMModel.from_local(voxcpm_model_path)
        self.text_normalizer = None
        if enable_denoiser and zipenhancer_model_path is not None:
            try:
                from .zipenhancer import ZipEnhancer
                self.denoiser = ZipEnhancer(zipenhancer_model_path)
                print("ZipEnhancer denoiser initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize ZipEnhancer denoiser: {e}")
                print("Audio denoising will be disabled")
                self.denoiser = None
        else:
            self.denoiser = None
        print("Warm up VoxCPMModel...")
        self.tts_model.generate(
            target_text="Hello, this is the first test sentence.",
            max_len=10,
        )

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = None,
            load_denoiser: bool = True,
            zipenhancer_model_path: str = "./models/zipenhancer_onnx",
            cache_dir: str = None,
            local_files_only: bool = True,  # Changed to True by default to prevent downloads
        ):
        """Instantiate ``VoxCPM`` from a local model directory.

        Args:
            hf_model_id: Local filesystem path to the model directory.
            load_denoiser: Whether to initialize the denoiser pipeline.
            zipenhancer_model_id: Denoiser model id or path for ModelScope
                acoustic noise suppression.
            cache_dir: Custom cache directory (not used for local models).
            local_files_only: If True, only use local files and do not attempt
                to download.

        Returns:
            VoxCPM: Initialized instance whose ``voxcpm_model_path`` points to
            the local model directory.

        Raises:
            ValueError: If no valid local model directory is provided.
            FileNotFoundError: If the local model directory doesn't contain required files.
        """
        if not hf_model_id:
            # Try to auto-detect local model directories
            possible_paths = [
                "./weights",  # Safetensors directory
                "./models/VoxCPM-0.5B",  # Default local directory
                "./models",  # Fallback
            ]
            
            for path in possible_paths:
                if os.path.isdir(path):
                    # Check if this directory contains model files
                    safetensors_files = ["model.safetensors", "audiovae.safetensors", "config.json"]
                    pytorch_files = ["pytorch_model.bin", "audiovae.pth", "config.json"]
                    
                    has_safetensors = all(os.path.exists(os.path.join(path, f)) for f in safetensors_files)
                    has_pytorch = all(os.path.exists(os.path.join(path, f)) for f in pytorch_files)
                    
                    if has_safetensors or has_pytorch:
                        hf_model_id = path
                        print(f"Auto-detected model directory: {path}")
                        break
            
            if not hf_model_id:
                raise ValueError(
                    "No model directory provided and no local model found. "
                    "Please specify a local model directory or place model files in ./weights/ or ./models/VoxCPM-0.5B/"
                )
        
        # Only allow local paths - no automatic downloads
        if not os.path.isdir(hf_model_id):
            raise FileNotFoundError(
                f"Model directory '{hf_model_id}' not found. "
                "This app only supports local models. Please provide a valid local directory path."
            )
        
        # Validate that the directory contains required files
        safetensors_files = ["model.safetensors", "audiovae.safetensors", "config.json"]
        pytorch_files = ["pytorch_model.bin", "audiovae.pth", "config.json"]
        
        has_safetensors = all(os.path.exists(os.path.join(hf_model_id, f)) for f in safetensors_files)
        has_pytorch = all(os.path.exists(os.path.join(hf_model_id, f)) for f in pytorch_files)
        
        if not (has_safetensors or has_pytorch):
            raise FileNotFoundError(
                f"Model directory '{hf_model_id}' is missing required files. "
                f"Expected either safetensors files: {safetensors_files} OR pytorch files: {pytorch_files}"
            )
        
        return cls(
            voxcpm_model_path=hf_model_id,
            zipenhancer_model_path=zipenhancer_model_path if load_denoiser else None,
            enable_denoiser=load_denoiser,
        )

    def generate(self, 
            text : str,
            prompt_wav_path : str = None,
            prompt_text : str = None,
            cfg_value : float = 2.0,    
            inference_timesteps : int = 10,
            max_length : int = 4096,
            normalize : bool = True,
            denoise : bool = True,
            retry_badcase : bool = True,
            retry_badcase_max_times : int = 3,
            retry_badcase_ratio_threshold : float = 6.0,
        ):
        """Synthesize speech for the given text and return a single waveform.

        This method optionally builds and reuses a prompt cache. If an external
        prompt (``prompt_wav_path`` + ``prompt_text``) is provided, it will be
        used for all sub-sentences. Otherwise, the prompt cache is built from
        the first generated result and reused for the remaining text chunks.

        Args:
            text: Input text. Can include newlines; each non-empty line is
                treated as a sub-sentence.
            prompt_wav_path: Path to a reference audio file for prompting.
            prompt_text: Text content corresponding to the prompt audio.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            max_length: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            denoise: Whether to denoise the prompt audio if a denoiser is
                available.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
        Returns:
            numpy.ndarray: 1D waveform array (float32) on CPU.
        """
        texts = text.split("\n")
        texts = [t.strip() for t in texts if t.strip()]
        final_wav = []
        temp_prompt_wav_path = None 
        
        try:
            if prompt_wav_path is not None and prompt_text is not None:
                if denoise and self.denoiser is not None:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            temp_prompt_wav_path = tmp_file.name
                        self.denoiser.enhance(prompt_wav_path, output_path=temp_prompt_wav_path)
                        prompt_wav_path = temp_prompt_wav_path
                    except Exception as e:
                        print(f"Warning: Audio denoising failed: {e}. Using original audio.")
                        # Continue with the original prompt_wav_path
                fixed_prompt_cache = self.tts_model.build_prompt_cache(
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text
                )
            else:
                fixed_prompt_cache = None  # will be built from the first inference
            
            for sub_text in texts:
                if sub_text.strip() == "":
                    continue
                print("sub_text:", sub_text)
                if normalize:
                    if self.text_normalizer is None:
                        from .utils.text_normalize import TextNormalizer
                        self.text_normalizer = TextNormalizer()
                    sub_text = self.text_normalizer.normalize(sub_text)
                wav, target_text_token, generated_audio_feat = self.tts_model.generate_with_prompt_cache(
                                target_text=sub_text,
                                prompt_cache=fixed_prompt_cache,
                                min_len=2,
                                max_len=max_length,
                                inference_timesteps=inference_timesteps,
                                cfg_value=cfg_value,
                                retry_badcase=retry_badcase,
                                retry_badcase_max_times=retry_badcase_max_times,
                                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                            )
                if fixed_prompt_cache is None:
                    fixed_prompt_cache = self.tts_model.merge_prompt_cache(
                        original_cache=None,
                        new_text_token=target_text_token,
                        new_audio_feat=generated_audio_feat
                    )
                final_wav.append(wav)
        
            return torch.cat(final_wav, dim=1).squeeze(0).cpu().numpy()
        
        finally:
            if temp_prompt_wav_path and os.path.exists(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass  