import os
from crewai import Agent, Task, Crew
from crewai.tools import tool
from crewai_tools import SerperDevTool

# ---------------------------------------------------------
# Set up your API keys (Replace with your actual keys)
# ---------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# ==========================================
# 1. The Environment (The Pheromone Board)
# ==========================================
class PheromoneBoard:
    def __init__(self):
        # Dictionary to hold dynamic trends discovered by the swarm
        self.trails = {}

    def drop_pheromone(self, trend_name: str, score: int, summary: str):
        # If the trend is new, add it to the board
        if trend_name not in self.trails:
            self.trails[trend_name] = {"score": 0, "trail_data": []}
        
        # Add the score and the summary string to the trail
        self.trails[trend_name]["score"] += score
        self.trails[trend_name]["trail_data"].append(summary)

    def read_strongest_trail(self) -> str:
        if not self.trails:
            return "No trails discovered."
        
        # Find the trend with the highest cumulative score
        best_trend = max(self.trails.items(), key=lambda x: x[1]['score'])
        trend_name = best_trend[0]
        data = best_trend[1]['trail_data']
        return f"Strongest trend is '{trend_name}' with aggregated findings: {data}"

# Instantiate the global environment for the swarm
environment = PheromoneBoard()

# ==========================================
# 2. Custom CrewAI Tools
# ==========================================
@tool("Drop Pheromone")
def drop_pheromone_tool(trend_name: str, score: int, finding: str) -> str:
    """
    Used by scouts to leave a score (1-10) and a summary of a finding for a specific trend.
    ALWAYS use this tool after researching a trend to record your findings to the swarm board.
    """
    environment.drop_pheromone(trend_name, score, finding)
    return f"Successfully dropped pheromone of {score} on {trend_name}."

@tool("Read Strongest Scent")
def read_scent_tool(dummy: str = "read") -> str:
    """
    Used by the Harvester to read the shared swarm environment and find which trend scored the highest.
    Provides the winning trend name and all summarized data from the scouts.
    """
    return environment.read_strongest_trail()

# Initialize standard search tool for the agents to browse the web
search_tool = SerperDevTool()

# ==========================================
# 3. Define the Swarm Agents
# ==========================================
tech_scout = Agent(
    role="Technology Scout",
    goal="Discover the latest technological trends related to the topic. Drop pheromones for the most viable tech trends.",
    backstory="You are a fast, analytical explorer. You do not write long reports. You search the web, identify specific tech trends, evaluate them, and leave scent trails using your Drop Pheromone tool.",
    tools=[search_tool, drop_pheromone_tool],
    allow_delegation=False,
    verbose=True
)

market_scout = Agent(
    role="Market & Consumer Scout",
    goal="Discover the latest market, consumer, or business trends related to the topic. Drop pheromones for the most viable market trends.",
    backstory="You are a sharp market analyst. You search the web, identify specific business or consumer behavior trends, evaluate their financial viability, and leave scent trails using your Drop Pheromone tool.",
    tools=[search_tool, drop_pheromone_tool],
    allow_delegation=False,
    verbose=True
)

harvester_agent = Agent(
    role="Swarm Harvester",
    goal="Read the strongest pheromone trail left by the scouts and write a comprehensive final report on the winning trend.",
    backstory="You rely entirely on the swarm's collective intelligence. You must use the 'Read Strongest Scent' tool to see what the scouts discovered. You then write a final briefing on the top-scored trend.",
    tools=[read_scent_tool],
    allow_delegation=False,
    verbose=True
)

# ==========================================
# 4. Orchestrate with a Function
# ==========================================
def run_swarm(topic: str):
    print(f"\n🚀 Deploying Swarm to research: {topic}\n")

    # The tasks dynamically accept the user's topic
    task_scout_tech = Task(
        description=f"Search the web for the latest technology trends related to '{topic}'. Identify at least 2 distinct tech trends. For each, use the 'Drop Pheromone' tool to assign a score (1-10) based on its current momentum and leave a 1-sentence summary.",
        expected_output="Confirmation that pheromones were successfully dropped for tech trends.",
        agent=tech_scout,
        async_execution=True # <--- Swarm Magic: Runs concurrently!
    )

    task_scout_market = Task(
        description=f"Search the web for the latest market/business trends related to '{topic}'. Identify at least 2 distinct market trends. For each, use the 'Drop Pheromone' tool to assign a score (1-10) based on financial potential and leave a 1-sentence summary.",
        expected_output="Confirmation that pheromones were successfully dropped for market trends.",
        agent=market_scout,
        async_execution=True # <--- Swarm Magic: Runs concurrently!
    )

    task_harvest = Task(
        description="First, use the 'Read Strongest Scent' tool to retrieve the winning trend from the environment board. Then, write a 3-paragraph final report explaining why this trend is dominating the topic and summarizing the scouts' findings.",
        expected_output="A well-formatted, 3-paragraph markdown report on the winning trend.",
        agent=harvester_agent,
        context=[task_scout_tech, task_scout_market] # <--- Swarm Magic: Harvester waits for both scouts to finish
    )

    # Assemble the Crew
    swarm_crew = Crew(
        agents=[tech_scout, market_scout, harvester_agent],
        tasks=[task_scout_tech, task_scout_market, task_harvest],
        verbose=True
    )

    # Kickoff the swarm
    result = swarm_crew.kickoff()
    return result

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    # Get the dynamic topic from the user
    user_topic = input("Enter a topic you want the swarm to research (e.g., 'Artificial Intelligence', 'Sustainable Agriculture', 'Wearable Tech'): ")
    
    final_report = run_swarm(user_topic)
    
    print("\n" + "="*50)
    print("🐝 SWARM FINAL REPORT 🐝")
    print("="*50)
    print(final_report)