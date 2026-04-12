import io
import pyaudio
import queue
import re
import sys
import threading
import time

from core.stt import STTAgent
from core.llm import ChatAgent
from core.tts import TTSAgent
from core.utils import time_it
from core.audio_utils import record_audio


# Force stdout to use UTF-8 to avoid UnicodeEncodeError on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ANSI colors for better visibility
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"

# Audio setup
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=22050,
    output=True
)

audio_queue = queue.Queue()


def playback_worker():
    """Background thread to play audio chunks from the queue."""
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        stream.write(chunk)
        audio_queue.task_done()


# Start playback thread
playback_thread = threading.Thread(target=playback_worker, daemon=True)
playback_thread.start()


def is_sentence_end(text):
    """Check if the text contains a sentence ending punctuation."""
    return bool(re.search(r'[.!?\n]', text))


def get_modes():
    """Prompt the user for input and output modes."""
    print(f"\n{Colors.BOLD}--- AGENTIC VOICE PIPELINE SETUP ---{Colors.END}")
    
    print(f"\n{Colors.YELLOW}Select Input Mode:{Colors.END}")
    print("1. Voice Input")
    print("2. Text Input")
    input_choice = input("Choice (1/2): ").strip()
    input_mode = "voice" if input_choice == "1" else "text"
    
    print(f"\n{Colors.YELLOW}Select Output Mode:{Colors.END}")
    print("1. Voice Output")
    print("2. Text Output")
    output_choice = input("Choice (1/2): ").strip()
    output_mode = "voice" if output_choice == "1" else "text"
    
    return input_mode, output_mode


@time_it
def run_pipeline(input_mode, output_mode, stt, llm, tts):
    print(f"\n{Colors.BOLD}--- PIPELINE ANALYSIS ---{Colors.END}")
    
    user_text = ""
    stt_metrics = None

    # 1. INPUT STAGE
    if input_mode == "voice":
        print(f"{Colors.CYAN}[STT] STARTING...{Colors.END}")
        record_audio("user_audio.wav")
        stt_response = stt.generate("user_audio.wav")
        stt_metrics = stt_response.get('metrics')
        user_text = stt_response.get('text')
        print(f"{Colors.CYAN}[STT] ENDED | Total: {stt_metrics['total_time']:.3f}s | TTFT: {stt_metrics['ttft']:.3f}s{Colors.END}")
        print(f"User: {user_text}\n")
    else:
        user_text = input(f"\n{Colors.BOLD}User:{Colors.END} ").strip()
        if not user_text:
            return
        if user_text.lower() in ['exit', 'quit']:
            sys.exit(0)

    # 2. LLM & OUTPUT STAGE
    print(f"{Colors.GREEN}[LLM] STARTING...{Colors.END}")
    
    sentence_buffer = ""
    first_token_received = False
    first_tts_started = False
    
    tts_total_time = 0
    tts_ttfb_initial = 0

    print(f"{Colors.BOLD}Assistant:{Colors.END} ", end="", flush=True)
    
    # Stream LLM tokens
    llm_metrics = None
    for llm_chunk in llm.stream(user_text):
        if llm_chunk['type'] == 'token':
            if not first_token_received:
                first_token_received = True
            
            token = llm_chunk['text']
            print(token, end="", flush=True)
            
            if output_mode == "voice":
                sentence_buffer += token
                # If we hit a sentence boundary, send to TTS
                if is_sentence_end(token):
                    sentence = sentence_buffer.strip()
                    if sentence:
                        if not first_tts_started:
                            first_tts_started = True

                        for tts_chunk in tts.stream(sentence):
                            if tts_chunk['type'] == 'chunk':
                                audio_queue.put(tts_chunk['audio'])
                            elif tts_chunk['type'] == 'end':
                                tts_total_time += tts_chunk['metrics']['total_time']
                                if tts_ttfb_initial == 0:
                                    tts_ttfb_initial = tts_chunk['metrics']['ttfb']
                        sentence_buffer = ""

        elif llm_chunk['type'] == 'end':
            # Handle remaining buffer for TTS
            if output_mode == "voice":
                sentence = sentence_buffer.strip()
                if sentence:
                    for tts_chunk in tts.stream(sentence):
                        if tts_chunk['type'] == 'chunk':
                            audio_queue.put(tts_chunk['audio'])
                        elif tts_chunk['type'] == 'end':
                            tts_total_time += tts_chunk['metrics']['total_time']
            
            llm_metrics = llm_chunk.get('metrics')
            print(f"\n\n{Colors.GREEN}[LLM] ENDED | Total: {llm_metrics['total_time']:.3f}s | TTFT: {llm_metrics['ttft']:.3f}s{Colors.END}")
            if output_mode == "voice":
                 print(f"{Colors.YELLOW}[TTS] FLOW SUMMARY | First TTFB: {tts_ttfb_initial:.3f}s | Total Synthesis: {tts_total_time:.3f}s{Colors.END}")

    # 3. FINAL SUMMARY
    print(f"\n{Colors.BOLD}{'='*30}")
    print(f"{'PERFORMANCE SUMMARY':^30}")
    print(f"{'='*30}{Colors.END}")
    if stt_metrics:
        print(f"STT: {stt_metrics['total_time']:.3f}s (TTFT: {stt_metrics['ttft']:.3f}s)")
    if llm_metrics:
        print(f"LLM: {llm_metrics['total_time']:.3f}s (TTFT: {llm_metrics['ttft']:.3f}s)")
    if output_mode == "voice":
        print(f"TTS: {tts_total_time:.3f}s (First TTFB: {tts_ttfb_initial:.3f}s)")
    print(f"{Colors.BOLD}{'-'*30}{Colors.END}")


if __name__ == "__main__":
    try:
        input_mode, output_mode = get_modes()
        
        print(f"\n{Colors.BOLD}Loading required models...{Colors.END}")
        
        stt = None
        llm = None
        tts = None
        
        # Selectively load models
        if input_mode == "voice":
            print("- Initializing STT...")
            stt = STTAgent()
            
        print("- Initializing LLM...")
        llm = ChatAgent()
        
        if output_mode == "voice":
            print("- Initializing TTS...")
            tts = TTSAgent()
            
        print(f"{Colors.GREEN}Ready!{Colors.END}")
        
        # Main Loop
        while True:
            try:
                run_pipeline(input_mode, output_mode, stt, llm, tts)
                
                # If voice output, wait for it to finish before next loop
                if output_mode == "voice":
                    print("\nWaiting for audio to complete...")
                    audio_queue.join()
                
            except KeyboardInterrupt:
                print("\nInteraction interrupted.")
                break
            except Exception as e:
                print(f"\nError in pipeline: {e}")
                break

    except KeyboardInterrupt:
        pass

    finally:
        print("\nShutting down...")
        audio_queue.put(None)  # Stop playback thread
        playback_thread.join(timeout=1)
        stream.stop_stream()
        stream.close()
        p.terminate()
