from openai import OpenAI
client = OpenAI()

TYPE_PROMPTS = {
    "sales": """
Your goal is to convert visitors into customers.
- Warmly introduce the product/service
- Ask about their needs and budget
- Recommend the best plan or product
- Capture their name and email for follow-up
- Handle objections confidently
- Always end with a call to action (sign up, book demo, etc.)
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

def run_agent(agent_data, user_message):
    agent_type = agent_data.get("type", "general")
    type_instructions = TYPE_PROMPTS.get(agent_type, TYPE_PROMPTS["general"])

    system_prompt = f"""
You are {agent_data['name']}, an AI assistant.

Description:
{agent_data['description']}

Tone: {agent_data['tone']}

Your Role ({agent_type} bot):
{type_instructions}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content
