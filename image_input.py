import base64
from dotenv import load_dotenv
from pprint import pprint
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()

# 1. Read the local image in the same folder as raw bytes
image_path = "Singapore_skyline_at_sunset_viewed_from_Gardens_by_the_Bay_East_-_20120426.jpg"  # Replace with your exact PNG filename
with open(image_path, "rb") as image_file:
    img_bytes = image_file.read()

# 2. Base64 encode the bytes
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# 3. Build the multimodal message
multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Tell me about this image"},
    {"type": "image", "base64": img_b64, "mime_type": "image/jpg"}
])

# 4. Run the agent
multi_agent = create_agent(
    model="gpt-5-nano"
)

response = multi_agent.invoke({
    "messages": [multimodal_question]
})

pprint(response['messages'][-1].content)
