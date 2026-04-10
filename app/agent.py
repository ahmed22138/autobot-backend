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
4. Show payment options and give payment details
5. Ask for payment screenshot as proof
6. Confirm order and thank the customer

RULES:
- Always be helpful and friendly
- Show product prices in PKR
- If customer asks about a product not in the list, say it's not available
- After collecting all customer info, create the order
- For COD orders, confirm address and estimated delivery
- For EasyPaisa/JazzCash/Bank orders, give exact account details and ask for screenshot
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
Your goal is to be a helpful AI assistant.
- Answer questions clearly and concisely
- Stay on topic based on your description
- If unsure, say you will follow up
"""
}

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

        # Payment methods
        if payment:
            context += "\nPAYMENT METHODS ACCEPTED:\n"
            if payment.get('cash_on_delivery'):
                context += "- Cash on Delivery (COD)\n"
            if payment.get('easypaisa_number'):
                context += f"- EasyPaisa: {payment['easypaisa_number']}\n"
            if payment.get('jazzcash_number'):
                context += f"- JazzCash: {payment['jazzcash_number']}\n"
            if payment.get('bank_account'):
                context += f"- Bank Transfer: {payment.get('bank_name', '')} | Account: {payment['bank_account']} | Name: {payment.get('bank_account_name', '')}\n"

        return context
    except Exception:
        return ""


def run_agent(agent_data, user_message):
    agent_type = agent_data.get("type", "general")
    type_instructions = TYPE_PROMPTS.get(agent_type, TYPE_PROMPTS["general"])

    # For sales bot, fetch products + payment info
    sales_context = ""
    if agent_type == "sales":
        sales_context = get_sales_context(agent_data.get("agent_id", ""))

    system_prompt = f"""You are {agent_data['name']}, an AI assistant.

Description: {agent_data['description']}
Tone: {agent_data['tone']}

Your Role ({agent_type} bot):
{type_instructions}
{sales_context}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content
