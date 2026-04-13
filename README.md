# Agentic Voice Pipeline

A real-time, Agentic Voice-to-Voice (V2V) pipeline built for low-latency interactions. This project integrates state-of-the-art open-source models for Speech-to-Text (STT), Large Language Model (LLM) reasoning, and Text-to-Speech (TTS).

![Project Header](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/LLM-Ollama-orange)
![Faster-Whisper](https://img.shields.io/badge/STT-Faster--Whisper-blueviolet)
![Piper](https://img.shields.io/badge/TTS-Piper-red)

---

## 🚀 Key Features

- **End-to-End Voice Interaction**: Speak to an AI agent directly and hear its response.
- **Low Latency Streaming**: Implementation of a sentence-buffered pipeline to minimize Time to First Byte (TTFB).
- **Multi-Modal Input/Output**: Toggle between Voice and Text modes dynamically.
- **Automated Model Management**: Automatic downloading and verification of STT, LLM, and TTS models.
- **Rich Performance Metrics**: Real-time tracking of TTFT (Time to First Token), synthesis duration, and processing overhead.

---

## 🛠️ Architecture & Streaming Mechanism

### The Streaming Pipeline
To ensure a natural conversation flow, the project uses a **Sentence-Buffered Streaming** approach:

1. **LLM Token Generation**: The LLM starts streaming tokens immediately.
2. **Dynamic Buffering**: Tokens are accumulated until a complete sentence is detected (using punctuation patterns like `.`, `?`, `!`, or `\n`).
3. **Parallel TTS Synthesis**: As soon as a sentence is buffered, it is sent to the TTS engine in a separate generator.
4. **Asynchronous Playback**: Synthesized audio chunks are pushed to a thread-safe `Playback Queue`. A dedicated background thread monitors this queue and plays audio via PyAudio directly to the hardware.

> [!TIP]
> This "look-ahead" buffering allows the user to hear the start of a response while the LLM is still generating the rest of it.

### Core Components
- **STT**: [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (using a `tiny` model for maximum speed).
- **LLM**: [Ollama](https://ollama.com/) running **Gemma 3 (270m)** for ultra-fast reasoning.
- **TTS**: [Piper](https://github.com/rhasspy/piper) (ONNX-based) for high-quality, local speech synthesis.

---

## 📦 Local Setup Guide

### 1. Prerequisites
- **Python**: 3.10 or higher.
- **Ollama**: [Download and Install Ollama](https://ollama.com/download).
- **PortAudio**: (Required for PyAudio)
  - **Windows**: Built-in, usually works out of the box with the provided wheel.
  - **Linux/Mac**: May require `libportaudio2` or `portaudio` via homebrew.

### 2. Installation Steps

```bash
# Clone the repository
git clone https://github.com/Hemanggour/Agentic-Voice-Pipeline.git
cd Agentic-Voice-Pipeline

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Ollama Preparation
Ensure the Ollama service is running. The pipeline will automatically attempt to pull the required model if it's missing, but you can also do it manually:
```bash
ollama pull gemma3:270m
```

---

## 🎮 How to Use

Simply run the main entry point:
```bash
python main.py
```

### Interactive Setup
Upon running, you will be prompted to select your modes:
1. **Input Mode**: Voice Input (1) or Text Input (2).
2. **Output Mode**: Voice Output (1) or Text Output (2).
3. **Debug Mode**: Enable (y) to see performance metrics and internal logs.

### Basic Commands
- In **Text Input** mode, type `exit` or `quit` to stop.
- In **Voice Input** mode, follow the console prompts to start/stop speaking.

---

## ⚙️ Configuration Management

All settings are centralized in `core/config.py`. You can modify this file to change:

- **LLM Model**: Switch to larger models (e.g., `llama3`, `gemma3:4b`).
- **TTS Voice**: Change the Piper voice profile by updating the `DOWNLOAD_URLS` and names.
- **STT Precision**: Switch from `tiny` to `base` or `small` for better accuracy.
- **Audio Parameters**: Adjust sample rates, chunk sizes, and hardware device settings.

```python
# core/config.py snippet
LLM = {
    "MODEL": "gemma3:270m",
    "TEMPERATURE": 0.6,
    "MEMORY_LIMIT": 10,
}
```

---

## 📊 Performance Monitoring
If Debug mode is enabled, the pipeline prints a detailed breakdown for every interaction:
- **STT TTFT**: Time from end of speech to first transcribed word.
- **LLM TTFT**: Time until the LLM generates its first token.
- **TTS TTFB**: Time until the first audio chunk is synthesized.
- **Overall Latency**: Total time between user input and assistant response.

---

## 🤝 Contributing
Feel free to open issues or submit PRs for:
- Support for additional STT/TTS engines.
- Refined sentence detection logic.
- UI/Web interface integration.
