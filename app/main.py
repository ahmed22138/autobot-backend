import os
import re
import httpx
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

# ── CORS ──────────────────────────────────────────────────────────────────────
# /chat/ must allow all origins — embed chatbot runs on external websites
# /create-agent is protected by rate limiting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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


# ── Email helper ──────────────────────────────────────────────────────────────
async def send_order_email(owner_email: str, order_info: dict):
    """Send new order notification to owner via Resend"""
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        return  # silently skip if not configured

    product  = order_info.get("product", "Unknown Product")
    customer = order_info.get("customer", "Unknown Customer")
    amount   = order_info.get("amount", "")
    method   = order_info.get("payment_method", "")

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "from":    "AutoBot Studio <orders@autobotstudio.com>",
                    "to":      [owner_email],
                    "subject": f"🛒 New Order: {product}",
                    "html": f"""
<div style="font-family:sans-serif;max-width:500px;margin:auto;background:#0a0a0f;color:#fff;padding:32px;border-radius:12px">
  <h2 style="color:#22c55e">✅ New Order Received!</h2>
  <table style="width:100%;border-collapse:collapse;margin-top:16px">
    <tr><td style="color:#a1a1aa;padding:8px 0">Product</td><td style="color:#fff;font-weight:600">{product}</td></tr>
    <tr><td style="color:#a1a1aa;padding:8px 0">Amount</td><td style="color:#22c55e;font-weight:700">{amount}</td></tr>
    <tr><td style="color:#a1a1aa;padding:8px 0">Customer</td><td style="color:#fff">{customer}</td></tr>
    <tr><td style="color:#a1a1aa;padding:8px 0">Payment</td><td style="color:#fff">{method}</td></tr>
  </table>
  <a href="https://autobot-studio.vercel.app/dashboard/orders"
     style="display:inline-block;margin-top:24px;padding:12px 24px;background:linear-gradient(135deg,#3b82f6,#7c3aed);color:#fff;border-radius:8px;text-decoration:none;font-weight:600">
    View Order →
  </a>
  <p style="color:#52525b;font-size:12px;margin-top:24px">AutoBot Studio — Your AI Sales Platform</p>
</div>"""
                }
            )
    except Exception:
        pass  # never fail chat because of email


def extract_order_from_history(conversation_history: list, agent_id: str) -> dict | None:
    """Try to extract order details from conversation for saving"""
    if not conversation_history:
        return None
    full_text = " ".join(
        m.get("content", "") for m in conversation_history if isinstance(m.get("content"), str)
    )

    # Simple regex extractions
    email_match   = re.search(r'[\w.-]+@[\w.-]+\.\w+', full_text)
    phone_match   = re.search(r'03\d{2}[\s-]?\d{7}', full_text)
    amount_match  = re.search(r'PKR\s*([\d,]+)', full_text)
    name_match    = re.search(r'(?:name[:\s]+|I am\s+|my name is\s+)([A-Za-z ]{3,30})', full_text, re.IGNORECASE)

    if not email_match and not phone_match:
        return None  # not enough info

    order_number = f"ORD-{int(datetime.now().timestamp())}"
    return {
        "agent_id":       agent_id,
        "order_number":   order_number,
        "customer_name":  name_match.group(1).strip() if name_match else "Customer",
        "customer_email": email_match.group(0) if email_match else "",
        "customer_phone": phone_match.group(0) if phone_match else "",
        "product_name":   "Order via chatbot",
        "product_price":  float(amount_match.group(1).replace(",","")) if amount_match else 0,
        "payment_method": "chatbot",
        "payment_status": "screenshot_received",
        "order_status":   "new",
    }


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

    # If screenshot verified → save order + email owner
    if req.image and reply.startswith("✅ VERIFIED"):
        try:
            order = extract_order_from_history(req.conversation_history or [], agent_id)
            if order and owner_id:
                supabase.table("orders").insert(order).execute()

                # Get owner email
                owner_result = supabase.auth.admin.get_user_by_id(owner_id)
                owner_email  = owner_result.user.email if owner_result and owner_result.user else None
                if owner_email:
                    import asyncio
                    asyncio.create_task(send_order_email(owner_email, {
                        "product":        order.get("product_name"),
                        "customer":       order.get("customer_name"),
                        "amount":         f"PKR {order.get('product_price', 0):,.0f}",
                        "payment_method": order.get("payment_method"),
                    }))
        except Exception:
            pass

    return ChatResponse(reply=reply)
