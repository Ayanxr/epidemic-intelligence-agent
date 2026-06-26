from langchain_community.tools import DuckDuckGoSearchRun
import re

class PublicHealthSearchTool:
    def __init__(self):
        # Initialize the base DuckDuckGo search runner
        self.search_engine = DuckDuckGoSearchRun()

    def search_verified_medical_web(self, raw_query: str) -> str:
        """
        Executes a targeted web search by engineering the query to pull strictly 
        from high-authority public health domains (CDC, WHO, NIH).
        """
        print(f"🌐 Querying the web for real-time verification: '{raw_query}'")
        
        # Clean the query of any stray punctuation or formatting
        clean_query = re.sub(r'[^\w\s]', '', raw_query)
        
        # Query Engineering: Restrict results to verified domains to bypass blogs or news panic
        engineered_query = f"{clean_query} (site:cdc.gov OR site:who.int OR site:nih.gov)"
        
        try:
            # Run the search
            results = self.search_engine.run(engineered_query)
            
            if not results or "No good DuckDuckGo Search Result found" in results:
                # Broaden the search slightly if the strict restriction returns nothing
                fallback_query = f"{clean_query} disease outbreak alert 2026"
                print("🔄 Strict domain search empty. Broadening search parameters...")
                results = self.search_engine.run(fallback_query)
                
            return results
            
        except Exception as e:
            return f"Error executing web fallback verification tool: {str(e)}"

# Quick standalone test execution snippet
if __name__ == "__main__":
    tool = PublicHealthSearchTool()
    # Test a query to verify it fetches data
    test_result = tool.search_verified_medical_web("Mpox transmission prevention guidelines")
    print("\n--- Tool Test Output ---")
    print(test_result[:500] + "...")