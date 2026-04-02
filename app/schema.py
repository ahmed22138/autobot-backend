from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID

# 1️⃣ Agent Create Request
class AgentCreate(BaseModel):
    name: str = Field(..., example="Ahmed Support Bot", min_length=3)
    description: str = Field(
        ...,
        example="Frontend developer helping clients with UI, bugs, and performance",
        min_length=10
    )
    tone: str = Field(..., example="friendly", min_length=3)


# 2️⃣ Chat Request
class ChatRequest(BaseModel):
    message: str = Field(..., example="How can you help me?")


# 3️⃣ Chat Response
class ChatResponse(BaseModel):
    reply: str


# 4️⃣ Agent Response (Optional but professional)
class AgentResponse(BaseModel):
    agent_id: UUID
    embed_url: str
