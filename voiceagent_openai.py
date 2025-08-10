### check audio input device
  
#    python voiceagent_openai.py --list-devices
#    python voiceagent_openai.py --device 1
    
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, try to load manually
    pass

import os
# Verify required environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY environment variable is required. "
        "Please set it in your .env file or environment. "
        "See env_template.txt for reference."
    )

import numpy as np
import sounddevice as sd
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Button, RichLog, Static
from typing_extensions import override

from agents.voice import StreamedAudioInput, VoicePipeline

# Import MyWorkflow class - handle both module and package use cases
if TYPE_CHECKING:
    # For type checking, use the relative import
    from .my_workflow import MyWorkflow
else:
    # At runtime, try both import styles
    try:
        # Try relative import first (when used as a package)
        from .my_workflow import MyWorkflow
    except ImportError:
        # Fall back to direct import (when run as a script)
        from my_workflow import MyWorkflow

CHUNK_LENGTH_S = 0.05  # 100ms
SAMPLE_RATE = 24000
FORMAT = np.int16
CHANNELS = 1


class Header(Static):
    """A header widget."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return "Speak to the agent. When you stop speaking, it will respond."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)"
            if self.is_recording
            else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #input-container {
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
        }

        #bottom-pane {
            width: 100%;
            height: 82%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
    """

    should_send_audio: asyncio.Event
    audio_player: sd.OutputStream
    last_audio_item_id: str | None
    connected: asyncio.Event
    manual_device_id: int | None = None

    def __init__(self) -> None:
        super().__init__()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.manual_device_id = None
        self.pipeline = VoicePipeline(
            workflow=MyWorkflow(secret_word="dog", on_start=self._on_transcription)
        )
        self._audio_input = StreamedAudioInput()
        self.audio_player = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=FORMAT,
        )

    def _on_transcription(self, transcription: str) -> None:
        try:
            self.query_one("#bottom-pane", RichLog).write(f"Transcription: {transcription}")
        except Exception:
            pass

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield Header(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        # Test audio input first
        await self.test_audio_input()
        
        self.run_worker(self.start_voice_pipeline())
        self.run_worker(self.send_mic_audio())

    async def test_audio_input(self) -> None:
        """Test if audio input is working properly."""
        bottom_pane = self.query_one("#bottom-pane", RichLog)
        bottom_pane.write("🧪 Testing audio input...")
        
        try:
            # Test microphone access
            devices = sd.query_devices()
            bottom_pane.write(f"📱 Found {len(devices)} total audio devices")
            
            # Find input devices (devices that can record audio)
            input_devices = []
            for i, device in enumerate(devices):
                device_name = device.get('name', 'Unknown')
                # Check various properties that indicate input capability
                is_input = (
                    device.get('max_inputs', 0) > 0 or  # Some systems have this
                    device.get('channels', 0) > 0 or     # Some systems have this
                    'input' in device_name.lower() or    # Name contains 'input'
                    'mic' in device_name.lower() or      # Name contains 'mic'
                    'recording' in device_name.lower()   # Name contains 'recording'
                )
                
                if is_input:
                    input_devices.append((i, device))
                    bottom_pane.write(f"  🎤 Device {i}: {device_name}")
            
            if not input_devices:
                # Show all devices for manual inspection
                bottom_pane.write("⚠️ Could not auto-detect input devices, showing all devices:")
                for i, device in enumerate(devices):
                    device_name = device.get('name', 'Unknown')
                    device_info = f"  Device {i}: {device_name}"
                    
                    # Show additional device properties
                    if 'max_inputs' in device:
                        device_info += f" (inputs: {device['max_inputs']})"
                    if 'channels' in device:
                        device_info += f" (channels: {device['channels']})"
                    
                    bottom_pane.write(device_info)
                
                bottom_pane.write("🎤 Please identify which device is your microphone")
            
            # Test default device
            try:
                default_input = sd.default.device[0]
                bottom_pane.write(f"🎤 Default input device: {default_input}")
                
                # Test if we can create a stream with the default device
                test_stream = sd.InputStream(
                    device=default_input,
                    channels=1,
                    samplerate=16000,
                    dtype="int16",
                    blocksize=1600,
                )
                
                bottom_pane.write("✅ Audio stream creation successful")
                test_stream.close()
                
            except Exception as e:
                bottom_pane.write(f"⚠️ Default device test failed: {e}")
                # Try with device 0 as fallback
                try:
                    test_stream = sd.InputStream(
                        device=0,
                        channels=1,
                        samplerate=16000,
                        dtype="int16",
                        blocksize=1600,
                    )
                    bottom_pane.write("✅ Fallback device 0 stream creation successful")
                    test_stream.close()
                except Exception as e2:
                    bottom_pane.write(f"❌ Fallback device also failed: {e2}")
            
        except Exception as e:
            bottom_pane.write(f"❌ Audio test failed: {e}")
            import traceback
            bottom_pane.write(f"🔍 Traceback: {traceback.format_exc()}")
        
        bottom_pane.write("🧪 Audio test completed")

    async def start_voice_pipeline(self) -> None:
        bottom_pane = self.query_one("#bottom-pane", RichLog)
        bottom_pane.write("🚀 Starting voice pipeline...")
        
        try:
            bottom_pane.write("🔊 Starting audio player...")
            self.audio_player.start()
            bottom_pane.write("✅ Audio player started")
            
            bottom_pane.write("🎯 Initializing voice pipeline...")
            self.result = await self.pipeline.run(self._audio_input)
            bottom_pane.write("✅ Voice pipeline initialized")

            bottom_pane.write("🔄 Starting event stream...")
            async for event in self.result.stream():
                if event.type == "voice_stream_event_audio":
                    if event.data is not None:
                        self.audio_player.write(event.data)
                        bottom_pane.write(
                            f"🔊 Received audio: {len(event.data)} bytes"
                        )
                    else:
                        bottom_pane.write("⚠️ Received null audio data")
                elif event.type == "voice_stream_event_lifecycle":
                    bottom_pane.write(f"📊 Lifecycle event: {event.event}")
                else:
                    bottom_pane.write(f"📝 Unknown event: {event.type}")
                    
        except Exception as e:
            bottom_pane.write(f"❌ Voice pipeline error: {e}")
            import traceback
            bottom_pane.write(f"🔍 Traceback: {traceback.format_exc()}")
        finally:
            bottom_pane.write("🛑 Stopping audio player...")
            self.audio_player.close()
            bottom_pane.write("✅ Audio player stopped")

    async def send_mic_audio(self) -> None:
        device_info = sd.query_devices()
        print("Available audio devices:")
        print(device_info)
        
        # Use manual device ID if specified, otherwise auto-detect
        if self.manual_device_id is not None:
            working_device = self.manual_device_id
            print(f"Using manually specified device: {working_device}")
            
            # Test the manual device
            try:
                test_stream = sd.InputStream(
                    device=working_device,
                    channels=CHANNELS,
                    samplerate=SAMPLE_RATE,
                    dtype="int16",
                    blocksize=int(SAMPLE_RATE * 0.02),
                )
                test_stream.close()
                print(f"✅ Manual device {working_device} test successful")
            except Exception as e:
                print(f"❌ Manual device {working_device} test failed: {e}")
                return
        else:
            # Get default input device with fallback
            try:
                default_input = sd.default.device[0]
                print(f"Using default input device: {default_input}")
            except (IndexError, KeyError):
                default_input = 0
                print(f"Default device not found, using device 0")
            
            # Try to find a working input device
            working_device = None
            for device_id in [default_input, 0, 1, 2]:  # Try common device IDs
                try:
                    test_stream = sd.InputStream(
                        device=device_id,
                        channels=CHANNELS,
                        samplerate=SAMPLE_RATE,
                        dtype="int16",
                        blocksize=int(SAMPLE_RATE * 0.02),
                    )
                    test_stream.close()
                    working_device = device_id
                    print(f"✅ Found working input device: {device_id}")
                    break
                except Exception as e:
                    print(f"❌ Device {device_id} failed: {e}")
                    continue
            
            if working_device is None:
                print("❌ No working input device found!")
                return
        
        read_size = int(SAMPLE_RATE * 0.02)  # 20ms chunks
        print(f"Reading {read_size} samples per chunk from device {working_device}")

        stream = sd.InputStream(
            device=working_device,  # Use the working device
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            blocksize=read_size,  # Set block size
        )
        
        print("Starting audio stream...")
        stream.start()
        print("Audio stream started successfully")

        status_indicator = self.query_one(AudioStatusIndicator)
        bottom_pane = self.query_one("#bottom-pane", RichLog)

        try:
            while True:
                # Always read audio data to keep buffer flowing
                try:
                    data, overflowed = stream.read(read_size)
                    if overflowed:
                        bottom_pane.write("⚠️ Audio overflow detected")
                    
                    # Only process audio when recording is active
                    if status_indicator.is_recording:
                        await self._audio_input.add_audio(data)
                        bottom_pane.write(f"🎤 Recording: {len(data)} samples")
                    
                    await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
                    
                except Exception as e:
                    bottom_pane.write(f"❌ Audio read error: {e}")
                    await asyncio.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("Audio recording interrupted")
        except Exception as e:
            bottom_pane.write(f"❌ Fatal audio error: {e}")
        finally:
            print("Stopping audio stream...")
            stream.stop()
            stream.close()
            print("Audio stream stopped")

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            self.query_one(Button).press()
            return

        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            if status_indicator.is_recording:
                self.should_send_audio.clear()
                status_indicator.is_recording = False
            else:
                self.should_send_audio.set()
                status_indicator.is_recording = True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Agent with OpenAI")
    parser.add_argument(
        "--device", "-d", 
        type=int, 
        help="Audio input device ID (use --list-devices to see available devices)"
    )
    parser.add_argument(
        "--list-devices", "-l", 
        action="store_true", 
        help="List available audio devices and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            device_name = device.get('name', 'Unknown')
            device_info = f"Device {i}: {device_name}"
            
            # Show additional device properties
            if 'max_inputs' in device:
                device_info += f" (inputs: {device['max_inputs']})"
            if 'channels' in device:
                device_info += f" (channels: {device['channels']})"
            
            print(device_info)
        exit(0)
    
    # Create app with optional device override
    app = RealtimeApp()
    if args.device is not None:
        app.manual_device_id = args.device
        print(f"Using manually specified device: {args.device}")
    
    app.run()