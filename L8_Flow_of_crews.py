from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv
from litellm import completion
import os

# Load environment variables
load_dotenv()

# Set up API keys
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_key")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class BlogWritingFlow(Flow):
    model = "gpt-4o-mini"  # You can change this to your preferred model

    @start()
    def generate_topic(self):
        print("Starting flow - Generating random topic...")

        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Generate a random but interesting blog topic. Return only the topic name without any additional text.",
                },
            ],
        )

        random_topic = response["choices"][0]["message"]["content"]
        print(f"Generated Topic: {random_topic}")
        return random_topic

    @listen(generate_topic)
    def create_outline(self, random_topic):
        print("Creating blog outline...")
        
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Create a detailed outline for a blog post about {random_topic}.
                    Include:
                    - Main sections
                    - Key points for each section
                    - Potential subheadings
                    Return in a structured format.""",
                },
            ],
        )

        outline = response["choices"][0]["message"]["content"]
        print("Outline created!")
        return outline

    @listen(create_outline)
    def research_topic(self, outline):
        print("Researching the topic...")
        
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Based on this outline: {outline}
                    Provide key research points, statistics, and relevant information that should be included in the blog post.
                    Focus on factual, interesting, and engaging information.""",
                },
            ],
        )

        research = response["choices"][0]["message"]["content"]
        return research

    @listen(research_topic)
    def write_blog(self, research):
        print("Writing the blog post...")
        
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Using this research: {research}
                    Write a comprehensive blog post.
                    The blog post should be:
                    - Well-structured
                    - Engaging and informative
                    - Include an introduction and conclusion
                    - Written in a conversational tone
                    Format the post in markdown.""",
                },
            ],
        )

        blog_post = response["choices"][0]["message"]["content"]
        return blog_post

    @listen(write_blog)
    def edit_and_optimize(self, blog_post):
        print("Editing and optimizing the blog post...")
        
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Review and optimize this blog post:
                    {blog_post}
                    
                    Please:
                    1. Check for clarity and coherence
                    2. Improve sentence structure
                    3. Add SEO optimization
                    4. Ensure proper formatting
                    5. Add meta description and tags
                    
                    Return the complete optimized post with meta information.""",
                },
            ],
        )

        final_post = response["choices"][0]["message"]["content"]
        return final_post

def main():
    # Initialize and run the flow
    flow = BlogWritingFlow()
    result = flow.kickoff()
    
    # Save the blog post to a file
    with open("generated_blog_post.md", "w") as f:
        f.write(result)
    
    print("\nBlog post has been generated and saved to 'generated_blog_post.md'")
    print("\nFinal blog post:")
    print(result)

if __name__ == "__main__":
    main()