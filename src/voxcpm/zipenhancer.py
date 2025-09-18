"""
ZipEnhancer Module - Audio Denoising Enhancer with ONNX support

Provides on-demand import ZipEnhancer functionality for audio denoising processing.
Uses ONNX format for better performance and compatibility.
"""

import os
import tempfile
from typing import Optional, Union
import torchaudio
import torch
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class ZipEnhancer:
    """ZipEnhancer Audio Denoising Enhancer with ONNX support"""
    def __init__(self, model_path: str = "iic/speech_zipenhancer_ans_multiloss_16k_base"):
        """
        Initialize ZipEnhancer with ONNX support
        Args:
            model_path: ModelScope model path or local path containing ONNX model
        """
        self.model_path = model_path
        self._session = None
        
        # Check if ONNX Runtime is available
        if ort is None:
            raise ImportError("ONNX Runtime is required. Please install with: pip install onnxruntime")
        
        # Initialize ONNX session
        self._initialize_onnx_session()
    
    def _initialize_onnx_session(self):
        """Initialize ONNX runtime session for the model"""
        try:
            # Try to load from local path first
            if os.path.isdir(self.model_path):
                # Look for ONNX model file in the directory (support multiple naming patterns)
                onnx_patterns = ["*.onnx", "*.onnxmodel", "model.onnx"]
                onnx_files = []
                
                for pattern in onnx_patterns:
                    onnx_files.extend(list(Path(self.model_path).glob(pattern)))
                    if onnx_files:
                        break
                
                if onnx_files:
                    model_file = str(onnx_files[0])
                    print(f"Loading ONNX model from: {model_file}")
                    
                    # Check if file is valid and not empty/corrupted
                    file_size = os.path.getsize(model_file)
                    if file_size == 0:
                        raise RuntimeError(f"ONNX model file is empty: {model_file}")
                    
                    # Configure ONNX Runtime session options
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    
                    # Use CUDA provider if available, otherwise CPU
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    
                    try:
                        self._session = ort.InferenceSession(model_file, sess_options=sess_options, providers=providers)
                        
                        # Print model info for debugging
                        print(f"ONNX model inputs: {[input.name for input in self._session.get_inputs()]}")
                        print(f"ONNX model outputs: {[output.name for output in self._session.get_outputs()]}")
                        
                        return
                    except Exception as inner_e:
                        raise RuntimeError(f"Failed to load ONNX model (file may be corrupted): {inner_e}")
            
            # If not found locally, provide helpful error message
            raise FileNotFoundError(
                f"ONNX model file not found in {self.model_path}. "
                "Please download the ONNX model using the download script:\n"
                "python download_models.py\n\n"
                "Or manually download from: "
                "https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base/resolve/master/onnx_model.onnx"
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX session: {e}")
        
    def _normalize_loudness(self, wav_path: str):
        """
        Audio loudness normalization
        
        Args:
            wav_path: Audio file path
        """
        audio, sr = torchaudio.load(wav_path)
        loudness = torchaudio.functional.loudness(audio, sr)
        normalized_audio = torchaudio.functional.gain(audio, -20-loudness)
        torchaudio.save(wav_path, normalized_audio, sr)
    
    def _preprocess_audio(self, audio_path: str):
        """Load and preprocess audio for ONNX inference"""
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed (typical for speech enhancement models)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            sr = 16000
        
        return audio, sr

    def _onnx_inference(self, audio: torch.Tensor):
        """Perform ONNX inference on audio data"""
        # Convert to numpy array and ensure correct shape
        audio_np = audio.numpy().astype(np.float32)
        
        # Get model input/output information
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        
        # Check if the model expects spectral inputs (magnitude/phase)
        input_names = [input.name for input in inputs]
        
        if 'noisy_mag' in input_names and 'noisy_pha' in input_names:
            # This model expects magnitude and phase spectral inputs
            # For now, we'll return the original audio since we don't have the spectral processing implemented
            print("Warning: ZipEnhancer model expects spectral inputs (magnitude/phase). Enhancement skipped.")
            return audio
            
        elif len(inputs) == 1:
            # Assume the model takes raw audio input
            input_info = inputs[0]
            output_info = outputs[0]
            
            # Prepare input based on the expected shape
            expected_shape = input_info.shape
            current_shape = audio_np.shape
            
            # Reshape if necessary (common for audio models)
            if len(expected_shape) == 3 and len(current_shape) == 2:
                # Add batch dimension if missing: [channels, samples] -> [1, channels, samples]
                audio_np = audio_np[np.newaxis, :, :]
            elif len(expected_shape) == 2 and len(current_shape) == 1:
                # Add channel dimension if missing: [samples] -> [1, samples]
                audio_np = audio_np[np.newaxis, :]
            
            # Run inference
            output = self._session.run([output_info.name], {input_info.name: audio_np})
            
            # Return the enhanced audio
            enhanced_audio = output[0]
            
            # Remove batch dimension if added
            if enhanced_audio.shape[0] == 1 and len(enhanced_audio.shape) > 1:
                enhanced_audio = enhanced_audio[0]
            
            return torch.from_numpy(enhanced_audio)
        else:
            # Unknown input format, return original audio
            print("Warning: Unknown input format for ZipEnhancer model. Enhancement skipped.")
            return audio

    def enhance(self, input_path: str, output_path: Optional[str] = None, 
                normalize_loudness: bool = True) -> str:
        """
        Audio denoising enhancement using ONNX model
        Args:
            input_path: Input audio file path
            output_path: Output audio file path (optional, creates temp file by default)
            normalize_loudness: Whether to perform loudness normalization
        Returns:
            str: Output audio file path
        Raises:
            RuntimeError: If ONNX session is not initialized or processing fails
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file does not exist: {input_path}")
        
        if self._session is None:
            raise RuntimeError("ONNX session not initialized")
        
        # Create temporary file if no output path is specified
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                output_path = tmp_file.name
        
        try:
            # Load and preprocess audio
            audio, sr = self._preprocess_audio(input_path)
            
            # Perform ONNX inference
            enhanced_audio = self._onnx_inference(audio)
            
            # Save enhanced audio
            torchaudio.save(output_path, enhanced_audio, sr)
            
            # Loudness normalization
            if normalize_loudness:
                self._normalize_loudness(output_path)
            
            return output_path
            
        except Exception as e:
            # Clean up possibly created temporary files
            if output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"Audio denoising processing failed: {e}")