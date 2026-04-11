from openai import OpenAI
import os
from supabase import create_client

client = OpenAI()

def get_supabase():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

TYPE_PROMPTS = {
    "sales": """
You are a sales assistant. Your job is to help customers buy products.

CONVERSATION FLOW:
1. Greet the customer and show available products with prices
2. Help them choose a product
3. Collect: Full Name, Email, Phone Number, Delivery Address
4. Ask which payment method they prefer
5. Show FULL payment details for that method (name + number/account) exactly like this:

   ═══════════════════════════
   💳 Payment Details
   ═══════════════════════════
   [Method Name e.g. EasyPaisa]
   Name:   [Account Holder Name]
   Number: [Account Number]
   Amount: PKR [product price]
   ═══════════════════════════

6. Tell customer: "Please send payment and share screenshot"
7. After screenshot received and verified, confirm order

RULES:
- Always show Name AND Number/Account together — never just the number alone
- Show PKR amount clearly
- If COD: confirm delivery address and say cash will be collected on delivery
- Be helpful and friendly throughout
""",
    "support": """
Your goal is to resolve customer issues quickly.
- Listen carefully to the problem
- Provide clear step-by-step solutions
- If you cannot solve it, collect their name, email, and issue details to escalate
- Be patient and empathetic
- Always confirm if the issue is resolved
""",
    "general": """
Your goal is to answer questions ONLY from the knowledge base provided.

RULES:
- Answer ONLY based on the knowledge base content below
- If the question is not covered in the knowledge base, say: "I don't have information about that. Please contact us directly."
- Do NOT make up answers or use outside knowledge
- Be helpful, clear, and concise
- If user asks something related, find the closest answer in the knowledge base
"""
}

def get_general_context(agent_id: str) -> str:
    """Fetch knowledge base for general bot"""
    try:
        supabase = get_supabase()
        result = supabase.table("agents").select("knowledge_base").eq("agent_id", agent_id).single().execute()
        kb = result.data.get("knowledge_base") if result.data else None
        if kb:
            return f"\n\nKNOWLEDGE BASE:\n{kb}\n"
        return ""
    except Exception:
        return ""


def get_sales_context(agent_id: str) -> str:
    """Fetch products and payment config for sales bot"""
    try:
        supabase = get_supabase()

        # Fetch products
        products_result = supabase.table("agent_products").select("*").eq("agent_id", agent_id).eq("in_stock", True).execute()
        products = products_result.data or []

        # Fetch payment config
        payment_result = supabase.table("agent_payment_config").select("*").eq("agent_id", agent_id).single().execute()
        payment = payment_result.data or {}

        context = ""

        # Products list
        if products:
            context += "\n\nAVAILABLE PRODUCTS:\n"
            for p in products:
                context += f"- {p['name']}: PKR {p['price']}"
                if p.get('description'):
                    context += f" ({p['description']})"
                context += "\n"
        else:
            context += "\n\nNo products added yet.\n"

        # Payment methods — formatted exactly as bot should show to customer
        if payment:
            context += "\nPAYMENT METHODS (show EXACTLY like this to customer):\n"

            if payment.get('easypaisa_number'):
                name = payment.get('easypaisa_account_name', '')
                context += f"\n📱 EasyPaisa\n"
                if name:
                    context += f"   Name:   {name}\n"
                context += f"   Number: {payment['easypaisa_number']}\n"

            if payment.get('jazzcash_number'):
                name = payment.get('jazzcash_account_name', '')
                context += f"\n📱 JazzCash\n"
                if name:
                    context += f"   Name:   {name}\n"
                context += f"   Number: {payment['jazzcash_number']}\n"

            if payment.get('bank_account'):
                context += f"\n🏦 Bank Transfer\n"
                if payment.get('bank_name'):
                    context += f"   Bank:    {payment['bank_name']}\n"
                if payment.get('bank_account_name'):
                    context += f"   Name:    {payment['bank_account_name']}\n"
                context += f"   Account: {payment['bank_account']}\n"

            if payment.get('cash_on_delivery'):
                context += f"\n🚚 Cash on Delivery (COD) available\n"

        context += "\nIMPORTANT: When customer selects a payment method, show ALL details of that method (name + number/account) clearly so they can make the payment easily.\n"

        return context
    except Exception:
        return ""


def verify_payment_screenshot(image_base64: str, payment_config: dict, expected_amount: float = None) -> str:
    """Use GPT-4o Vision to verify payment screenshot against owner's account details"""

    # Build what to look for
    valid_accounts = []
    if payment_config.get("easypaisa_number"):
        name = payment_config.get("easypaisa_account_name", "")
        valid_accounts.append(f"EasyPaisa: {payment_config['easypaisa_number']}" + (f" ({name})" if name else ""))
    if payment_config.get("jazzcash_number"):
        name = payment_config.get("jazzcash_account_name", "")
        valid_accounts.append(f"JazzCash: {payment_config['jazzcash_number']}" + (f" ({name})" if name else ""))
    if payment_config.get("bank_account"):
        valid_accounts.append(f"Bank: {payment_config.get('bank_name','')} | {payment_config['bank_account']} | {payment_config.get('bank_account_name','')}")

    accounts_str = "\n".join(valid_accounts) if valid_accounts else "No payment accounts configured"
    amount_check = f"\n- Amount should be PKR {expected_amount}" if expected_amount else ""

    verification_prompt = f"""Analyze this payment screenshot carefully.

OWNER'S VALID PAYMENT ACCOUNTS:
{accounts_str}{amount_check}

Check the screenshot and tell me:
1. Was payment made to one of the owner's accounts listed above? (yes/no)
2. What account number/name appears in the screenshot?
3. What amount was transferred?
4. Is this a valid completed transaction? (not pending/failed)

If everything matches → reply starting with "✅ VERIFIED:"
If payment went to wrong account → reply starting with "❌ WRONG ACCOUNT:"
If amount is wrong → reply starting with "❌ WRONG AMOUNT:"
If screenshot is unclear/fake → reply starting with "❌ INVALID:"

Be strict — only verify if the account number exactly matches."""

    # Clean base64 if it has data URI prefix
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": verification_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }],
        max_tokens=300
    )
    return response.choices[0].message.content


def run_agent(agent_data, user_message, image_base64: str = None, conversation_history: list = []):
    agent_type = agent_data.get("type", "general")
    type_instructions = TYPE_PROMPTS.get(agent_type, TYPE_PROMPTS["general"])

    # Fetch extra context based on type
    sales_context = ""
    general_context = ""
    payment_config = {}

    if agent_type == "sales":
        sales_context = get_sales_context(agent_data.get("agent_id", ""))
    elif agent_type == "general":
        general_context = get_general_context(agent_data.get("agent_id", ""))
        # Also fetch payment config for verification
        try:
            supabase = get_supabase()
            result = supabase.table("agent_payment_config").select("*").eq("agent_id", agent_data.get("agent_id", "")).single().execute()
            payment_config = result.data or {}
        except Exception:
            pass

    # Handle screenshot verification
    if image_base64 and agent_type == "sales" and payment_config:
        return verify_payment_screenshot(image_base64, payment_config)

    system_prompt = f"""You are {agent_data['name']}, an AI assistant.

Description: {agent_data['description']}
Tone: {agent_data['tone']}

Your Role ({agent_type} bot):
{type_instructions}
{sales_context}{general_context}"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (only role + content, skip images in history)
    for msg in conversation_history[:-1]:  # exclude last message (current one)
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": str(msg["content"])})

    if image_base64:
        # Non-sales bot with image — just describe it
        if "," in image_base64:
            clean_b64 = image_base64.split(",")[1]
        else:
            clean_b64 = image_base64
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message or "What is in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"}}
            ]
        })
        model = "gpt-4o"
    else:
        messages.append({"role": "user", "content": user_message})
        model = "gpt-4o-mini"

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content
