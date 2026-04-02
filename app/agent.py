from openai import OpenAI
client = OpenAI()

def run_agent(agent_data, user_message):
    system_prompt = f"""
You are {agent_data['name']}, a professional AI assistant.

Description:
{agent_data['description']}

Tone:
{agent_data['tone']}

Rules:
- Answer only based on this info
- If unsure, say you will follow up
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    return response.output_text
