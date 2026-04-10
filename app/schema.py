from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID

# 1️⃣ Agent Create Request
class AgentCreate(BaseModel):
    name: str = Field(..., example="Ahmed Support Bot")
    description: str = Field(..., example="Frontend developer helping clients with UI, bugs, and performance")
    tone: str = Field(..., example="friendly")
    type: str = Field(default="general", example="sales")


# 2️⃣ Chat Request
class ChatRequest(BaseModel):
    message: str = Field(default="", example="How can you help me?")
    image: Optional[str] = Field(default=None, description="Base64 encoded image for payment screenshot verification")


# 3️⃣ Chat Response
class ChatResponse(BaseModel):
    reply: str


# 4️⃣ Agent Response (Optional but professional)
class AgentResponse(BaseModel):
    agent_id: UUID
    embed_url: str
