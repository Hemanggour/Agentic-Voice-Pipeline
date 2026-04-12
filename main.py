import io
import sys
from core.pipeline import VoicePipeline, Colors

# Force stdout to use UTF-8 to avoid UnicodeEncodeError on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_modes():
    """Prompt the user for input and output modes and debug preference."""
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

    debug_choice = input(f"\nEnable debug mode? (y/n, default: n): ").strip().lower()
    debug = True if debug_choice == 'y' else False
    
    return input_mode, output_mode, debug


if __name__ == "__main__":
    pipeline = None
    try:
        input_mode, output_mode, debug = get_modes()
        
        # Initialize the pipeline
        pipeline = VoicePipeline(input_mode=input_mode, output_mode=output_mode, debug=debug)
        
        # Main Interaction Loop
        while True:
            try:
                # pipeline.run() returns False if the user wants to exit
                if not pipeline.run():
                    break
                
            except KeyboardInterrupt:
                print("\nInteraction interrupted.")
                break
            except Exception as e:
                print(f"\nError in pipeline: {e}")
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
    finally:
        if pipeline:
            pipeline.close()
        else:
            print("\nShutting down...")
