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

stt = STTAgent()
llm = ChatAgent()
tts = TTSAgent()

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


@time_it
def generate_pipeline():
    stt_response = stt.generate("user_audio.wav")
    print(f"STT [{stt_response.get('metrics').get('ttft'):.3f}s]: {stt_response.get('text')}")

    llm_response = llm.generate(stt_response.get('text'))
    print(f"LLM [{llm_response.get('metrics').get('ttft'):.3f}s]: {llm_response.get('text')}")

    tts_response = tts.generate(llm_response.get('text'))
    audio_queue.put(tts_response.get('audio'))
    print(f"TTS [{tts_response.get('metrics').get('ttfb'):.3f}s]")


def is_sentence_end(text):
    """Check if the text contains a sentence ending punctuation."""
    return bool(re.search(r'[.!?\n]', text))


@time_it
def stream_pipeline():
    print(f"\n{Colors.BOLD}--- PIPELINE ANALYSIS ---{Colors.END}")
    
    # 1. STT STAGE
    print(f"{Colors.CYAN}[STT] STARTING...{Colors.END}")

    stt_response = stt.generate("user_audio.wav")
    stt_metrics = stt_response.get('metrics')

    print(f"{Colors.CYAN}[STT] ENDED | Total: {stt_metrics['total_time']:.3f}s | TTFT: {stt_metrics['ttft']:.3f}s{Colors.END}")
    print(f"User: {stt_response.get('text')}\n")

    # 2. LLM & TTS STAGE
    print(f"{Colors.GREEN}[LLM] STARTING...{Colors.END}")
    
    sentence_buffer = ""
    first_token_received = False
    first_tts_started = False
    
    tts_total_time = 0
    tts_ttfb_initial = 0

    print(f"{Colors.BOLD}Assistant:{Colors.END} ", end="", flush=True)
    
    # Stream LLM tokens
    for llm_chunk in llm.stream(stt_response.get('text')):
        if llm_chunk['type'] == 'token':
            if not first_token_received:
                first_token_received = True
                # Print TTFT for LLM silently or as a tag
            
            token = llm_chunk['text']
            print(token, end="", flush=True)
            sentence_buffer += token

            # If we hit a sentence boundary, send to TTS
            if is_sentence_end(token):
                sentence = sentence_buffer.strip()
                if sentence:
                    if not first_tts_started:
                        # Log when first TTS starts working
                        first_tts_started = True

                    # Synthesize sentence and put chunks in playback queue
                    for tts_chunk in tts.stream(sentence):
                        if tts_chunk['type'] == 'chunk':
                            audio_queue.put(tts_chunk['audio'])
                        elif tts_chunk['type'] == 'end':
                            tts_total_time += tts_chunk['metrics']['total_time']
                            if tts_ttfb_initial == 0:
                                tts_ttfb_initial = tts_chunk['metrics']['ttfb']
                    sentence_buffer = ""

        elif llm_chunk['type'] == 'end':
            # Handle remaining buffer
            sentence = sentence_buffer.strip()
            if sentence:
                for tts_chunk in tts.stream(sentence):
                    if tts_chunk['type'] == 'chunk':
                        audio_queue.put(tts_chunk['audio'])
                    elif tts_chunk['type'] == 'end':
                        tts_total_time += tts_chunk['metrics']['total_time']
            
            llm_metrics = llm_chunk.get('metrics')
            print(f"\n\n{Colors.GREEN}[LLM] ENDED | Total: {llm_metrics['total_time']:.3f}s | TTFT: {llm_metrics['ttft']:.3f}s{Colors.END}")
            print(f"{Colors.YELLOW}[TTS] FLOW SUMMARY | First TTFB: {tts_ttfb_initial:.3f}s | Total Synthesis: {tts_total_time:.3f}s{Colors.END}")

    # 3. FINAL SUMMARY
    print(f"\n{Colors.BOLD}{'='*30}")
    print(f"{'PERFORMANCE SUMMARY':^30}")
    print(f"{'='*30}{Colors.END}")
    print(f"STT: {stt_metrics['total_time']:.3f}s (TTFT: {stt_metrics['ttft']:.3f}s)")
    print(f"LLM: {llm_metrics['total_time']:.3f}s (TTFT: {llm_metrics['ttft']:.3f}s)")
    print(f"TTS: {tts_total_time:.3f}s (First TTFB: {tts_ttfb_initial:.3f}s)")
    print(f"{Colors.BOLD}{'-'*30}{Colors.END}")


if __name__ == "__main__":
    try:
        stream_pipeline()

        print("\nWaiting for audio to complete...")
        audio_queue.join()
        print("Done.")

    except KeyboardInterrupt:
        pass

    finally:
        audio_queue.put(None)  # Stop playback thread
        playback_thread.join(timeout=1)
        stream.stop_stream()
        stream.close()
        p.terminate()
