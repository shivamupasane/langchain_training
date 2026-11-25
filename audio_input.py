import base64
import io
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from pprint import pprint
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()

# 1. Define a sample tool for the agent to take action
@tool
def check_order_status(order_id: str) -> str:
    """Checks the status of a customer order given an order ID."""
    return f"Order {order_id} is packed and out for delivery today."

# 2. Record audio from your microphone
DURATION = 5  # Seconds to record
SAMPLE_RATE = 16000  # 16kHz keeps the file small (under LangSmith's 20MB limit)

print(f"🎤 Recording for {DURATION} seconds... Speak now!")
audio_data = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)
sd.wait()  # Wait until recording finishes
print("✅ Recording complete!")

# 3. Convert the NumPy array into WAV bytes in memory (no need to save to disk)
wav_buffer = io.BytesIO()
write(wav_buffer, SAMPLE_RATE, audio_data)
audio_b64 = base64.b64encode(wav_buffer.getvalue()).decode("utf-8")

# 4. Build the multimodal audio message
multimodal_question = HumanMessage(content=[
    {
        "type": "text",
        "text": "Listen to the user's audio request and use your tools to take action."
    },
    {
        "type": "audio",
        "base64": audio_b64,
        "mime_type": "audio/wav"
    }
])

# 5. Create the agent with an audio-capable model and invoke it
audio_agent = create_agent(
    model="gpt-audio",  # Must use an audio-capable model (e.g., gpt-audio or gemini-2.5-flash)
    tools=[check_order_status]
)

response = audio_agent.invoke({
    "messages": [multimodal_question]
})

pprint(response['messages'][-1].text)
