#!/usr/bin/env python3
"""
Script to download required models for VoxCPM.
1. ZipEnhancer ONNX model from ModelScope
2. VoxCPM Safetensors model from HuggingFace
"""

import os
import sys
import requests
from pathlib import Path

def download_file(url, target_path, description=""):
    """Download a file with progress tracking."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"{description} progress: {progress:.1f}%", end='\r')
        
        print(f"\n✅ {description} downloaded: {target_path.name}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {description}: {e}")
        return False

def download_zipenhancer_onnx():
    """Download ZipEnhancer ONNX model."""
    model_url = "https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base/resolve/master/onnx_model.onnx"
    target_dir = Path("./models/zipenhancer_onnx")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading ZipEnhancer ONNX model...")
    onnx_path = target_dir / "onnx_model.onnx"
    
    if download_file(model_url, onnx_path, "ZipEnhancer ONNX"):
        print(f"File size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    return False

def download_voxcpm_safetensors():
    """Download VoxCPM Safetensors model from HuggingFace."""
    base_url = "https://huggingface.co/euphoricpenguin22/VoxCPM-0.5B-Safetensors/resolve/main/"
    target_dir = Path("./weights")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # List of required files
    files_to_download = [
        "model.safetensors",
        "audiovae.safetensors", 
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    print("Downloading VoxCPM Safetensors model...")
    
    success_count = 0
    for filename in files_to_download:
        file_url = base_url + filename
        file_path = target_dir / filename
        
        if download_file(file_url, file_path, f"VoxCPM {filename}"):
            success_count += 1
    
    return success_count == len(files_to_download)

def main():
    """Main download function."""
    print("VoxCPM Model Downloader")
    print("=" * 40)
    
    # Download ZipEnhancer ONNX
    onnx_success = download_zipenhancer_onnx()
    
    # Download VoxCPM Safetensors
    safetensors_success = download_voxcpm_safetensors()
    
    print("\n" + "=" * 40)
    
    if onnx_success and safetensors_success:
        print("🎉 All models downloaded successfully!")
        print("\nModels are ready in:")
        print("- ./models/zipenhancer_onnx/ (ZipEnhancer ONNX)")
        print("- ./weights/ (VoxCPM Safetensors)")
    else:
        print("❌ Some downloads failed.")
        if not onnx_success:
            print("- ZipEnhancer ONNX model download failed")
        if not safetensors_success:
            print("- VoxCPM Safetensors model download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
