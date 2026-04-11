import os
from fastapi import FastAPI, HTTPException
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
from app.database import supabase
from app.agent import run_agent
from app.schema import (
    AgentCreate,
    AgentResponse,
    ChatRequest,
    ChatResponse
)

app = FastAPI(title="Autonomous AI Agent Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory agent store (primary storage - no auth required)
# Also tries to sync to Supabase
agents_store: dict = {}


# 🔹 CREATE AGENT
@app.post("/create-agent", response_model=AgentResponse)
def create_agent(data: AgentCreate):
    agent_id = str(uuid4())

    agent_data = {
        "agent_id": agent_id,
        "name": data.name,
        "description": data.description,
        "tone": data.tone,
        "type": data.type,
    }

    # Save to in-memory store (always works)
    agents_store[agent_id] = agent_data

    # Also try to save to Supabase (may fail if RLS/auth not configured)
    try:
        supabase.table("agents").insert(agent_data).execute()
    except Exception:
        pass  # In-memory store is the fallback

    return AgentResponse(
        agent_id=agent_id,
        embed_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/chatbot/{agent_id}"
    )


# 🔹 CHAT WITH AGENT
@app.post("/chat/{agent_id}", response_model=ChatResponse)
def chat(agent_id: str, req: ChatRequest):

    # Check in-memory store first (fast)
    agent_data = agents_store.get(agent_id)

    # Fallback: try Supabase
    if not agent_data:
        try:
            result = (
                supabase
                .table("agents")
                .select("*")
                .eq("agent_id", agent_id)
                .single()
                .execute()
            )
            agent_data = result.data
        except Exception:
            agent_data = None

    if not agent_data:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Always fetch latest status from Supabase (dashboard deactivate/activate)
    try:
        status_result = (
            supabase
            .table("agents")
            .select("status")
            .eq("agent_id", agent_id)
            .single()
            .execute()
        )
        if status_result.data:
            agent_data["status"] = status_result.data.get("status", "active")
    except Exception:
        pass

    if agent_data.get("status") == "inactive":
        raise HTTPException(status_code=403, detail="Agent is inactive")

    reply = run_agent(agent_data, req.message, req.image, req.conversation_history or [])

    # Count message against agent owner's usage
    try:
        owner_id = agent_data.get("user_id")
        if not owner_id:
            # Fetch user_id from Supabase if not in memory
            owner_result = supabase.table("agents").select("user_id").eq("agent_id", agent_id).single().execute()
            owner_id = owner_result.data.get("user_id") if owner_result.data else None

        if owner_id:
            supabase.table("messages").insert({
                "user_id": owner_id,
                "agent_id": agent_id,
                "role": "user",
                "text": req.message or "[image]",
            }).execute()
    except Exception:
        pass  # Don't fail chat if message counting fails

    return ChatResponse(reply=reply)
