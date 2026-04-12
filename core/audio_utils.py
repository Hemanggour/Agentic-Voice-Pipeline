import pyaudio
import wave
import threading
import sys

def record_audio(output_filename="user_audio.wav", rate=16000, chunk=1024, channels=1):
    """
    Records audio from the microphone until the user presses Enter.
    Saves the output to a WAV file.
    Default rate is 16000Hz as it's common for Whisper (STT).
    """
    p = pyaudio.PyAudio()
    
    # Try to find a working input device
    try:
        device_info = p.get_default_input_device_info()
    except Exception as e:
        print(f"Error: No default input device found. {e}")
        p.terminate()
        return False

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []
    stop_event = threading.Event()

    def record_thread():
        while not stop_event.is_set():
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

    print("\n[REC] Recording... Press ENTER to stop.", end="", flush=True)
    
    thread = threading.Thread(target=record_thread)
    thread.start()
    
    try:
        input() # Wait for Enter
    except KeyboardInterrupt:
        pass
    
    stop_event.set()
    thread.join()

    print("[REC] Done.\n")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return True
