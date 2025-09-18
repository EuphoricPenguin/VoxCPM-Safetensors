## 🎙️ VoxCPM-Safetensors: (More) Secure Fork of Tokenizer-Free TTS for Context-Aware Speech Generation



<div align="center">
  <img src="assets/voxcpm_logo.png" alt="VoxCPM Logo" width="40%">
</div>


## Overview

A fork focused on using model weights that are freely-licensed and reduce the risk of arbitrary code execution. The main weights [were converted to Safetensors](https://huggingface.co/euphoricpenguin22/VoxCPM-0.5B-Safetensors). SenseVoice was replaced with WhisperX Small to maintain free licensing. The ZipEnhancer model is now forcibly imported as ONNX, which offers similar security benefits to Safetensors. Triton was replaced with [triton-windows](https://github.com/woct0rdho/triton-windows) to get it running, although torch.compile might still need fixed. To be clear, I am not a security expert, but just a guy who was bugged with the prevelant usage of the [pickle](https://huggingface.co/docs/hub/en/security-pickle) format. WhisperX might still use pickle, but I consider it to be a more established and trusted model source than SenseVoice. This fork was created using Void Agent and DeepSeek. 

VoxCPM is a novel tokenizer-free Text-to-Speech (TTS) system that redefines realism in speech synthesis. By modeling speech in a continuous space, it overcomes the limitations of discrete tokenization and enables two flagship capabilities: context-aware speech generation and true-to-life zero-shot voice cloning.

Unlike mainstream approaches that convert speech to discrete tokens, VoxCPM uses an end-to-end diffusion autoregressive architecture that directly generates continuous speech representations from text. Built on [MiniCPM-4](https://huggingface.co/openbmb/MiniCPM4-0.5B) backbone, it achieves implicit semantic-acoustic decoupling through hierachical language modeling and FSQ constraints, greatly enhancing both expressiveness and generation stability.

<div align="center">
  <img src="assets/voxcpm_model.png" alt="VoxCPM Model Architecture" width="90%">
</div>


###  🚀 Key Features
- **Context-Aware, Expressive Speech Generation** - VoxCPM comprehends text to infer and generate appropriate prosody, delivering speech with remarkable expressiveness and natural flow. It spontaneously adapts speaking style based on content, producing highly fitting vocal expression trained on a massive 1.8 million-hour bilingual corpus.
- **True-to-Life Voice Cloning** - With only a short reference audio clip, VoxCPM performs accurate zero-shot voice cloning, capturing not only the speaker’s timbre but also fine-grained characteristics such as accent, emotional tone, rhythm, and pacing to create a faithful and natural replica.
- **High-Efficiency Synthesis** - VoxCPM supports streaming synthesis with a Real-Time Factor (RTF) as low as 0.17 on a consumer-grade NVIDIA RTX 4090 GPU, making it possible for real-time applications.





##  Quick Start

### 1. Installation

`pip install -r requirements.txt`

`python download_models.py`

### 2. Start web demo

You can start the UI interface by running `python app.py`, which allows you to perform Voice Cloning and Voice Creation.


## ⚠️ Risks and limitations
- General Model Behavior: While VoxCPM has been trained on a large-scale dataset, it may still produce outputs that are unexpected, biased, or contain artifacts.
- Potential for Misuse of Voice Cloning: VoxCPM's powerful zero-shot voice cloning capability can generate highly realistic synthetic speech. This technology could be misused for creating convincing deepfakes for purposes of impersonation, fraud, or spreading disinformation. Users of this model must not use it to create content that infringes upon the rights of individuals. It is strictly forbidden to use VoxCPM for any illegal or unethical purposes. We strongly recommend that any publicly shared content generated with this model be clearly marked as AI-generated.
- Current Technical Limitations: Although generally stable, the model may occasionally exhibit instability, especially with very long or expressive inputs. Furthermore, the current version offers limited direct control over specific speech attributes like emotion or speaking style.
- Bilingual Model: VoxCPM is trained primarily on Chinese and English data. Performance on other languages is not guaranteed and may result in unpredictable or low-quality audio.
- This model is released for research and development purposes only. We do not recommend its use in production or commercial applications without rigorous testing and safety evaluations. Please use VoxCPM responsibly.

## 📄 License
The VoxCPM model weights and code are open-sourced under the [Apache-2.0](LICENSE) license.

## 🙏 Acknowledgments

We extend our sincere gratitude to the following works and resources for their inspiration and contributions:

- [DiTAR](https://arxiv.org/abs/2502.03930) for the diffusion autoregressive backbone used in speech generation
- [MiniCPM-4](https://github.com/OpenBMB/MiniCPM) for serving as the language model foundation
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for the implementation of Flow Matching-based LocDiT
- [DAC](https://github.com/descriptinc/descript-audio-codec) for providing the Audio VAE backbone

## Institutions

This project is developed by the following institutions:
- <img src="assets/modelbest_logo.png" width="28px"> [ModelBest](https://modelbest.cn/)

- <img src="assets/thuhcsi_logo.png" width="28px"> [THUHCSI](https://github.com/thuhcsi)

## 📚 Citation

The techical report is coming soon, please wait for the release 😊

If you find our model helpful, please consider citing our projects 📝 and staring us ⭐️！

```bib
@misc{voxcpm2025,
  author       = {{Yixuan Zhou, Guoyang Zeng, Xin Liu, Xiang Li, Renjie Yu, Ziyang Wang, Runchuan Ye, Weiyue Sun, Jiancheng Gui, Kehan Li, Zhiyong Wu, Zhiyuan Liu}},
  title        = {{VoxCPM}},
  year         = {2025},
  publish = {\url{https://github.com/OpenBMB/VoxCPM}},
  note         = {GitHub repository}
}
```
