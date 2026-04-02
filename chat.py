
from agents import function_tool,Agent,Runner
from dotenv import load_dotenv

load_dotenv()

@function_tool
def get_ahmed_cv():
    return {
        "name": "Ahmed Malik",
        "role": "Frontend Developer",
        "experience": "Fresher (Strong practical projects)",
        "skills": [
            "HTML",
            "CSS",
            "TypeScript",
            "Next.js",
            "Tailwind CSS",
            "Basic Python",
            "UI/UX Fundamentals"
        ],
        "projects": [
            "Portfolio Website (Next.js + Tailwind)",
            "Password Strength Meter & Generator",
            "Animated Frontend Landing Pages"
        ],
        "mindset": "Believes in learning something new every single day, especially in fast-changing tech fields like AI & Web Development",
        "goal": "Job-ready frontend developer"
    }


agent = Agent(
    name="Ahmed Malik CV Agent",
    instructions="""
Tum Ahmed Malik ka autonomous AI agent ho.

Tumhara kaam:
- Uski CV explain karna
- Recruiter ke questions ka professional jawab dena
- Skills ke hisaab se projects suggest karna
- Job interview style answers dena
- Khud decide karna ke CV tool use karna hai ya nahi

Tum confident, clear aur industry-level answers dete ho.
""",
    tools=[get_ahmed_cv],
    model="gpt-4.1-mini"
)

message = "tell me about skills."

def ask_agent(message: str):
    result = Runner.run_sync(agent, message)
    return result.final_output

