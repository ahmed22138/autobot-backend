import os
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from uuid import uuid4
from app.database import supabase
from app.agent import run_agent
from app.schema import AgentCreate, AgentResponse, ChatRequest, ChatResponse

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="AutoBot Studio API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — only allow our frontend ────────────────────────────────────────────
ALLOWED_ORIGINS = [
    os.getenv("FRONTEND_URL", "https://autobot-studio.vercel.app"),
    "https://autobot-studio.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Plan message limits
PLAN_LIMITS = {
    "basic":   100,
    "medium":  1000,
    "premium": None,   # unlimited
}


# ── Helper: check owner message limit ─────────────────────────────────────────
def check_message_limit(owner_id: str) -> tuple[bool, str]:
    """Returns (allowed, reason). True = can proceed."""
    try:
        # Get owner's plan
        sub = supabase.table("subscriptions").select("plan").eq("user_id", owner_id).single().execute()
        plan = sub.data.get("plan", "basic") if sub.data else "basic"
        limit = PLAN_LIMITS.get(plan)

        if limit is None:
            return True, ""  # premium = unlimited

        # Count messages this month
        start_of_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        count_result = (
            supabase.table("messages")
            .select("*", count="exact", head=True)
            .eq("user_id", owner_id)
            .eq("role", "user")
            .gte("created_at", start_of_month)
            .execute()
        )
        used = count_result.count or 0

        if used >= limit:
            return False, f"Message limit reached ({used}/{limit}). Owner must upgrade plan."
        return True, ""
    except Exception:
        return True, ""  # don't block on error


# ── CREATE AGENT ──────────────────────────────────────────────────────────────
@app.post("/create-agent", response_model=AgentResponse)
@limiter.limit("20/minute")
def create_agent(request: Request, data: AgentCreate):
    agent_id = str(uuid4())

    agent_data = {
        "agent_id": agent_id,
        "name": data.name,
        "description": data.description,
        "tone": data.tone,
        "type": data.type,
    }

    try:
        supabase.table("agents").insert(agent_data).execute()
    except Exception:
        pass

    return AgentResponse(
        agent_id=agent_id,
        embed_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/chatbot/{agent_id}"
    )


# ── CHAT ──────────────────────────────────────────────────────────────────────
@app.post("/chat/{agent_id}", response_model=ChatResponse)
@limiter.limit("30/minute")
def chat(agent_id: str, req: ChatRequest, request: Request):

    # Fetch agent from Supabase (single source of truth — no in-memory store)
    try:
        result = (
            supabase.table("agents")
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

    # Check agent status
    if agent_data.get("status") == "inactive":
        raise HTTPException(status_code=403, detail="Agent is inactive")

    # Check owner's message limit
    owner_id = agent_data.get("user_id")
    if owner_id:
        allowed, reason = check_message_limit(owner_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="This chatbot has reached its monthly message limit. Please try again next month."
            )

    # Run AI
    reply = run_agent(agent_data, req.message, req.image, req.conversation_history or [])

    # Save message for usage tracking
    if owner_id:
        try:
            supabase.table("messages").insert({
                "user_id": owner_id,
                "agent_id": agent_id,
                "role": "user",
                "text": req.message or "[image]",
            }).execute()
        except Exception:
            pass

    return ChatResponse(reply=reply)
