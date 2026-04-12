import os
import re
import httpx
import logging
import time
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Simple in-memory agent cache (TTL = 5 min) ────────────────────────────────
_agent_cache: dict = {}   # agent_id → {"data": {...}, "ts": float}
CACHE_TTL = 300           # seconds

def get_cached_agent(agent_id: str):
    entry = _agent_cache.get(agent_id)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None

def set_cached_agent(agent_id: str, data: dict):
    _agent_cache[agent_id] = {"data": data, "ts": time.time()}

def invalidate_agent_cache(agent_id: str):
    _agent_cache.pop(agent_id, None)

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
def get_owner_email(owner_id: str) -> str | None:
    """Get owner email from Supabase auth"""
    try:
        result = supabase.auth.admin.get_user_by_id(owner_id)
        if result and result.user and result.user.email:
            return result.user.email
        logger.warning(f"Could not get email for owner {owner_id}: result={result}")
        return None
    except Exception as e:
        logger.error(f"Error fetching owner email for {owner_id}: {e}")
        return None


def send_order_email(owner_email: str, order_info: dict):
    """Send new order notification to owner via Resend"""
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logger.warning("RESEND_API_KEY not set — skipping email")
        return

    product  = order_info.get("product", "Unknown Product")
    customer = order_info.get("customer", "Unknown Customer")
    amount   = order_info.get("amount", "")
    method   = order_info.get("payment_method", "")

    logger.info(f"Sending order email to {owner_email} for product: {product}")

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "from":    "AutoBot Studio <onboarding@resend.dev>",
                    "to":      [owner_email],
                    "subject": f"New Order: {product}",
                    "html": f"""
<div style="font-family:sans-serif;max-width:500px;margin:auto;background:#0a0a0f;color:#fff;padding:32px;border-radius:12px">
  <h2 style="color:#22c55e">New Order Received!</h2>
  <table style="width:100%;border-collapse:collapse;margin-top:16px">
    <tr><td style="color:#a1a1aa;padding:8px 0">Product</td><td style="color:#fff;font-weight:600">{product}</td></tr>
    <tr><td style="color:#a1a1aa;padding:8px 0">Amount</td><td style="color:#22c55e;font-weight:700">{amount}</td></tr>
    <tr><td style="color:#a1a1aa;padding:8px 0">Customer</td><td style="color:#fff">{customer}</td></tr>
    <tr><td style="color:#a1a1aa;padding:8px 0">Payment</td><td style="color:#fff">{method}</td></tr>
  </table>
  <a href="https://autobot-studio.vercel.app/dashboard/orders"
     style="display:inline-block;margin-top:24px;padding:12px 24px;background:linear-gradient(135deg,#3b82f6,#7c3aed);color:#fff;border-radius:8px;text-decoration:none;font-weight:600">
    View Order
  </a>
  <p style="color:#52525b;font-size:12px;margin-top:24px">AutoBot Studio — Your AI Sales Platform</p>
</div>"""
                }
            )
            logger.info(f"Resend response: {resp.status_code} — {resp.text}")
    except Exception as e:
        logger.error(f"Email send failed: {e}")


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


# ── AGENT CONFIG (public — used by chatbot.js to get welcome message etc.) ────
@app.get("/agent/{agent_id}/config")
def get_agent_config(agent_id: str):
    try:
        result = (
            supabase.table("agents")
            .select("name, welcome_message, status, type")
            .eq("agent_id", agent_id)
            .single()
            .execute()
        )
        data = result.data
    except Exception:
        data = None

    if not data:
        raise HTTPException(status_code=404, detail="Agent not found")

    if data.get("status") == "inactive":
        raise HTTPException(status_code=403, detail="Agent is inactive")

    return {
        "name": data.get("name", "AI Assistant"),
        "welcome_message": data.get("welcome_message") or "",
        "type": data.get("type", "general"),
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
        invalidate_agent_cache(agent_id)
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

    # Fetch agent — use cache to avoid hitting DB on every message
    agent_data = get_cached_agent(agent_id)
    if not agent_data:
        try:
            result = (
                supabase.table("agents")
                .select("*")
                .eq("agent_id", agent_id)
                .single()
                .execute()
            )
            agent_data = result.data
            if agent_data:
                set_cached_agent(agent_id, agent_data)
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
        logger.info(f"Payment verified for agent {agent_id} — saving order and sending email")
        try:
            order = extract_order_from_history(req.conversation_history or [], agent_id)
            logger.info(f"Extracted order: {order}")
            if order and owner_id:
                # Deduplication: skip if same agent + phone/email got an order in last 5 min
                try:
                    from datetime import timedelta
                    five_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
                    dup_query = supabase.table("orders").select("id", count="exact", head=True)\
                        .eq("agent_id", agent_id)\
                        .gte("created_at", five_min_ago)
                    if order.get("customer_phone"):
                        dup_query = dup_query.eq("customer_phone", order["customer_phone"])
                    elif order.get("customer_email"):
                        dup_query = dup_query.eq("customer_email", order["customer_email"])
                    dup_result = dup_query.execute()
                    if (dup_result.count or 0) > 0:
                        logger.warning(f"Duplicate order skipped for agent {agent_id}")
                        return ChatResponse(reply=reply)
                except Exception as e:
                    logger.warning(f"Dedup check failed: {e}")

                supabase.table("orders").insert(order).execute()
                logger.info(f"Order saved to DB: {order.get('order_number')}")

                owner_email = get_owner_email(owner_id)
                if owner_email:
                    send_order_email(owner_email, {
                        "product":        order.get("product_name"),
                        "customer":       order.get("customer_name"),
                        "amount":         f"PKR {order.get('product_price', 0):,.0f}",
                        "payment_method": order.get("payment_method"),
                    })
                else:
                    logger.warning(f"No email found for owner {owner_id}")
            else:
                logger.warning(f"Order not saved — order={order}, owner_id={owner_id}")
        except Exception as e:
            logger.error(f"Order/email error: {e}")

    return ChatResponse(reply=reply)


# ── WHATSAPP WEBHOOK (Twilio) ─────────────────────────────────────────────────
@app.post("/webhook/whatsapp/{agent_id}")
async def whatsapp_webhook(agent_id: str, request: Request):
    """Receive WhatsApp message from Twilio, reply via agent"""
    from twilio.rest import Client as TwilioClient

    form = await request.form()
    incoming_msg = form.get("Body", "").strip()
    from_number  = form.get("From", "")

    logger.info(f"WhatsApp msg for agent {agent_id} from {from_number}: {incoming_msg}")

    if not incoming_msg or not from_number:
        return {"status": "ignored"}

    # Fetch agent
    agent_data = get_cached_agent(agent_id)
    if not agent_data:
        try:
            result = supabase.table("agents").select("*").eq("agent_id", agent_id).single().execute()
            agent_data = result.data
            if agent_data:
                set_cached_agent(agent_id, agent_data)
        except Exception:
            agent_data = None

    if not agent_data or agent_data.get("status") == "inactive":
        return {"status": "agent_unavailable"}

    # Check message limit
    owner_id = agent_data.get("user_id")
    if owner_id:
        allowed, _ = check_message_limit(owner_id)
        if not allowed:
            _send_whatsapp(from_number, "Sorry, service temporarily unavailable.")
            return {"status": "limit_reached"}

    # Run agent
    try:
        reply_text = run_agent(agent_data, incoming_msg, None, [])
    except Exception as e:
        logger.error(f"WhatsApp agent error: {e}")
        reply_text = "Sorry, kuch masla aa gaya. Thodi der baad dobara try karein."

    # Save message for usage tracking
    if owner_id:
        try:
            supabase.table("messages").insert({
                "user_id":  owner_id,
                "agent_id": agent_id,
                "role":     "user",
                "text":     incoming_msg,
            }).execute()
        except Exception:
            pass

    _send_whatsapp(from_number, reply_text)
    return {"status": "ok"}


def _send_whatsapp(to: str, message: str):
    """Send WhatsApp reply via Twilio"""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token  = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

    if not account_sid or not auth_token:
        logger.error("Twilio credentials missing")
        return
    try:
        from twilio.rest import Client as TwilioClient
        client = TwilioClient(account_sid, auth_token)
        client.messages.create(from_=from_number, to=to, body=message[:1600])
        logger.info(f"WhatsApp reply sent to {to}")
    except Exception as e:
        logger.error(f"Twilio send failed: {e}")
