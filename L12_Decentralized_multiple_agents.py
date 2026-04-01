import asyncio
from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Define the Specialized Agents
analyst_market = Agent(
    role='Market Trend Analyst',
    goal='Identify bullish/bearish trends for {topic}',
    backstory='Data-driven expert in market momentum.',
    verbose=True
)

analyst_risk = Agent(
    role='Risk Assessment Specialist',
    goal='Identify potential pitfalls for {topic}',
    backstory='Conservative expert focused on downside protection.',
    verbose=True
)

synthesizer = Agent(
    role='Consensus Coordinator',
    goal='Synthesize multiple viewpoints into a single balanced recommendation',
    backstory='Expert in resolving conflicting data points and finding truth.',
    verbose=True
)

# 2. Define the State
class MyFlowState(BaseModel):
    topic: str = ""
    market_data: str = ""
    risk_data: str = ""

# 3. The Decentralized Flow
class DecentralizedConsensusFlow(Flow[MyFlowState]):

    @start()
    def run_parallel_analysis(self):
        print(f"--- Analyzing: {self.state.topic} ---")
        
        # Convert StateProxy to a dict for the Crew
        # This is the fix for your TypeError
        flow_inputs = self.state.model_dump() 

        task_market = Task(
            description="Analyze market trends for {topic}", 
            agent=analyst_market,
            expected_output="Detailed market trend report."
        )
        task_risk = Task(
            description="Analyze risk factors for {topic}", 
            agent=analyst_risk,
            expected_output="A summary of risk factors."
        )

        # To run in parallel, we use a Crew with both tasks
        # By default, CrewAI executes tasks in the list order. 
        # For true decentralized parallel execution, we kickoff the crew.
        crew = Crew(
            agents=[analyst_market, analyst_risk], 
            tasks=[task_market, task_risk]
        )
        
        results = crew.kickoff(inputs=flow_inputs)
        
        # Extract individual results from the crew output
        self.state.market_data = task_market.output.raw
        self.state.risk_data = task_risk.output.raw
        
        return results.raw

    @listen(run_parallel_analysis)
    def aggregate_and_converge(self, analysis_results):
        print("--- Parallel Tasks Complete: Converging Results ---")
        
        convergence_task = Task(
            description=(
                f"Review these two independent reports:\n"
                f"1. Market Report: {self.state.market_data}\n"
                f"2. Risk Report: {self.state.risk_data}\n"
                "Find the common ground and provide one unified consensus strategy."
            ),
            agent=synthesizer,
            expected_output="A final unified consensus strategy document."
        )
        
        # Final convergence step
        crew = Crew(agents=[synthesizer], tasks=[convergence_task])
        return crew.kickoff().raw

# 4. Execute the System
if __name__ == "__main__":
    flow = DecentralizedConsensusFlow()
    final_strategy = flow.kickoff(inputs={"topic": "AI Hardware Startups"})

    print("\n\n########################")
    print("## FINAL CONSENSUS OUTPUT ##")
    print("########################\n")
    print(final_strategy)