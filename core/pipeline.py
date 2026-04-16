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
from core.audio_utils import record_audio

class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"

from core.config import Config

class VoicePipeline:
    def __init__(self, input_mode="text", output_mode="text", debug=None):
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.debug = debug if debug is not None else Config.DEBUG
        
        # Audio setup
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=Config.AUDIO["CHANNELS"],
            rate=Config.AUDIO["SAMPLE_RATE"],
            output=True
        )
        self.audio_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        # Model initialization
        self.stt = None
        self.llm = None
        self.tts = None
        
        self._initialize_agents()

    def _initialize_agents(self):
        if self.debug:
            print(f"\n{Colors.BOLD}Loading required models...{Colors.END}")
        
        if self.input_mode == "voice":
            if self.debug: print("- Initializing STT...")
            self.stt = STTAgent()
            
        if self.debug: print("- Initializing LLM...")
        self.llm = ChatAgent()
        
        if self.output_mode == "voice":
            if self.debug: print("- Initializing TTS...")
            self.tts = TTSAgent()
            
        if self.debug:
            print(f"{Colors.GREEN}Ready!{Colors.END}")

    def _playback_worker(self):
        """Background thread to play audio chunks from the queue."""
        while True:
            chunk = self.audio_queue.get()
            if chunk is None:
                break
            self.stream.write(chunk)
            self.audio_queue.task_done()

    def _is_sentence_end(self, text):
        """Check if the text contains a sentence ending punctuation."""
        return bool(re.search(r'[.!?\n]', text))

    def _debug_print(self, message):
        if self.debug:
            print(message)

    def run(self):
        if self.debug:
            print(f"\n{Colors.BOLD}--- PIPELINE ANALYSIS ---{Colors.END}")
        
        user_text = ""
        stt_metrics = None

        # 1. INPUT STAGE
        if self.input_mode == "voice":
            self._debug_print(f"{Colors.CYAN}[STT] STARTING...{Colors.END}")
            record_audio("user_audio.wav")
            stt_response = self.stt.generate("user_audio.wav")
            stt_metrics = stt_response.get('metrics')
            user_text = stt_response.get('text')
            self._debug_print(f"{Colors.CYAN}[STT] ENDED | Total: {stt_metrics['total_time']:.3f}s | TTFT: {stt_metrics['ttft']:.3f}s{Colors.END}")
            print(f"User: {user_text}\n")
        else:
            user_text = input(f"\n{Colors.BOLD}User:{Colors.END} ").strip()
            if not user_text:
                return True
            if user_text.lower() in ['exit', 'quit']:
                return False

        # 2. LLM & OUTPUT STAGE
        self._debug_print(f"{Colors.GREEN}[LLM] STARTING...{Colors.END}")
        
        sentence_buffer = ""
        first_token_received = False
        first_tts_started = False
        
        tts_total_time = 0
        tts_ttfb_initial = 0

        print(f"{Colors.BOLD}Assistant:{Colors.END} ", end="", flush=True)
        
        # Stream LLM tokens
        llm_metrics = None
        token_counter = 0
        for llm_chunk in self.llm.stream(user_text):
            if llm_chunk['type'] == 'token':
                if not first_token_received:
                    first_token_received = True
                
                token = llm_chunk['text']
                token_counter += 1
                print(token, end="", flush=True)
                
                if self.output_mode == "voice":
                    sentence_buffer += token
                    
                    # Check for punctuation OR token threshold
                    trigger_punct = self._is_sentence_end(token)
                    trigger_limit = token_counter >= Config.LLM["CHUNK_TOKEN_THRESHOLD"]
                    
                    if trigger_punct or trigger_limit:
                        reason = "PUNCTUATION" if trigger_punct else "THRESHOLD"
                        self._debug_print(f"\n{Colors.YELLOW}[PIPELINE] Chunking trigger: {reason} ({token_counter} tokens){Colors.END}")
                        
                        sentence = sentence_buffer.strip()
                        if sentence:
                            if not first_tts_started:
                                first_tts_started = True

                            for tts_chunk in self.tts.stream(sentence):
                                if tts_chunk['type'] == 'chunk':
                                    self.audio_queue.put(tts_chunk['audio'])
                                elif tts_chunk['type'] == 'end':
                                    tts_total_time += tts_chunk['metrics']['total_time']
                                    if tts_ttfb_initial == 0:
                                        tts_ttfb_initial = tts_chunk['metrics']['ttfb']
                            
                            sentence_buffer = ""
                            token_counter = 0

            elif llm_chunk['type'] == 'end':
                if self.output_mode == "voice":
                    sentence = sentence_buffer.strip()
                    if sentence:
                        for tts_chunk in self.tts.stream(sentence):
                            if tts_chunk['type'] == 'chunk':
                                self.audio_queue.put(tts_chunk['audio'])
                            elif tts_chunk['type'] == 'end':
                                tts_total_time += tts_chunk['metrics']['total_time']
                
                llm_metrics = llm_chunk.get('metrics')
                self._debug_print(f"\n\n{Colors.GREEN}[LLM] ENDED | Total: {llm_metrics['total_time']:.3f}s | TTFT: {llm_metrics['ttft']:.3f}s{Colors.END}")
                if self.output_mode == "voice":
                     self._debug_print(f"{Colors.YELLOW}[TTS] FLOW SUMMARY | First TTFB: {tts_ttfb_initial:.3f}s | Total Synthesis: {tts_total_time:.3f}s{Colors.END}")

        # 3. FINAL SUMMARY
        if self.debug:
            print(f"\n{Colors.BOLD}{'='*30}")
            print(f"{'PERFORMANCE SUMMARY':^30}")
            print(f"{'='*30}{Colors.END}")
            if stt_metrics:
                print(f"STT: {stt_metrics['total_time']:.3f}s (TTFT: {stt_metrics['ttft']:.3f}s)")
            if llm_metrics:
                print(f"LLM: {llm_metrics['total_time']:.3f}s (TTFT: {llm_metrics['ttft']:.3f}s)")
            if self.output_mode == "voice":
                print(f"TTS: {tts_total_time:.3f}s (First TTFB: {tts_ttfb_initial:.3f}s)")
            print(f"{Colors.BOLD}{'-'*30}{Colors.END}")

        if self.output_mode == "voice":
            self._debug_print("\nWaiting for audio to complete...")
            self.audio_queue.join()
            
        return True

    def close(self):
        """Cleanup resources."""
        if self.debug:
            print("\nShutting down pipeline...")
        self.audio_queue.put(None)
        if self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
