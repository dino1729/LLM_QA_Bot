from gpt_researcher import GPTResearcher
import dotenv
import asyncio
import logging

# Configure logging to show only errors
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

async def get_report(query: str, report_type: str):
    try:
        researcher = GPTResearcher(query, report_type)
        research_result = await researcher.conduct_research()
        report = await researcher.write_report()
        
        # Get additional information
        research_context = researcher.get_research_context()
        research_costs = researcher.get_costs()
        research_images = researcher.get_research_images()
        research_sources = researcher.get_research_sources()
        
        return report, research_context, research_costs, research_images, research_sources
    except Exception as e:
        logger.error(f"Error during research: {str(e)}")
        return None, None, None, None, None

if __name__ == "__main__":
    query = "Latest news headlines from Financial markets. Answer only with headlines and brief descriptions."
    report_type = "research_report"

    try:
        report, context, costs, images, sources = asyncio.run(get_report(query, report_type))
        
        if report:
            print("Report:")
            print(report)
        else:
            logger.error("Research failed to complete.")
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")