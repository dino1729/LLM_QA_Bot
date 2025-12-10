"""Autonomous research agent using LLM with Firecrawl tools."""

import logging
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .llm_client_base import LLMClient
from .firecrawl_client import FirecrawlClient

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Container for research findings."""
    summary: str
    key_facts: List[str]
    sources: List[Dict[str, str]]  # [{title, url}]
    raw_content: str  # Full research context for prompt
    success: bool
    error: Optional[str] = None


class ResearchAgent:
    """Autonomous research agent using LLM with Firecrawl tools.

    Works with any LLM backend (Ollama, LiteLLM, etc.) that implements
    the LLMClient interface and supports tool calling.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        firecrawl_client: FirecrawlClient,
        config: Dict[str, Any]
    ):
        """
        Initialize research agent.

        Args:
            llm_client: LLMClient instance for LLM calls (Ollama, LiteLLM, etc.)
            firecrawl_client: FirecrawlClient instance for web research
            config: Configuration dictionary
        """
        self.llm = llm_client
        self.firecrawl = firecrawl_client
        self.config = config

        research_config = config.get('research', {})
        self.max_iterations = research_config.get('max_iterations', 5)
        self.max_sources = research_config.get('max_sources', 8)
        self.search_limit = research_config.get('search_limit', 5)
        self.max_no_tool_attempts = research_config.get('max_no_tool_attempts', 2)

        # Track gathered information
        self.sources_gathered: List[Dict[str, str]] = []
        self.content_gathered: List[str] = []

    def research_topic(self, topic: str) -> ResearchResult:
        """
        Autonomously research a topic using tool calling.

        The LLM decides:
        - What queries to search
        - Which results to explore deeper
        - When enough information has been gathered

        Args:
            topic: Topic to research

        Returns:
            ResearchResult with summary, key facts, sources
        """
        logger.info(f"Starting autonomous research on topic: {topic}")

        try:
            # Reset state
            self.sources_gathered = []
            self.content_gathered = []
            
            # Store current topic for fallback query generation
            self._current_topic = topic

            # Run agent loop
            result = self._run_agent_loop(topic)

            logger.info(f"Research completed: {len(self.sources_gathered)} sources, {len(result.key_facts)} facts")
            return result

        except Exception as e:
            logger.error(f"Research failed with error: {e}", exc_info=True)
            return ResearchResult(
                summary="",
                key_facts=[],
                sources=[],
                raw_content="",
                success=False,
                error=str(e)
            )

    def _get_tools_definition(self) -> List[Dict]:
        """Define Firecrawl tools for Ollama tool calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information on a topic. Returns titles, URLs, and content previews.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant information"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (1-10)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_webpage",
                    "description": "Read the full content of a specific webpage. Use this to dive deeper into promising search results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL of the webpage to read"
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_research",
                    "description": "Signal that research is complete and provide a summary of findings.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Comprehensive summary of the research findings"
                            },
                            "key_facts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of the most important facts discovered"
                            }
                        },
                        "required": ["summary", "key_facts"]
                    }
                }
            }
        ]

    def _execute_tool_call(self, tool_name: str, arguments: Dict) -> str:
        """
        Execute a tool call and return result as string.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            String representation of tool result
        """
        try:
            if tool_name == "web_search":
                query = arguments.get('query', '')
                
                # Fallback: Generate query from topic if empty
                if not query or query.strip() == '':
                    # Get topic from the research context (stored in self during research_topic call)
                    query = getattr(self, '_current_topic', 'AI news December 2025')
                    logger.warning(f"Empty query detected, using fallback: {query}")
                
                limit = arguments.get('limit', self.search_limit)

                # Ensure limit is an integer
                if isinstance(limit, str):
                    try:
                        limit = int(limit)
                    except ValueError:
                        limit = self.search_limit

                logger.info(f"  Tool: web_search(query='{query}', limit={limit})")

                results = self.firecrawl.search(query=query, limit=limit, scrape=True)

                if not results:
                    return f"No results found for query: {query}"

                # Add to sources
                for result in results:
                    if len(self.sources_gathered) < self.max_sources:
                        self.sources_gathered.append({
                            'title': result.get('title', 'Untitled'),
                            'url': result.get('url', '')
                        })

                # Format results for LLM
                formatted_results = []
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'Untitled')
                    url = result.get('url', '')
                    description = result.get('description', 'No description')
                    markdown = result.get('markdown', '')

                    # Truncate markdown content if too long
                    if markdown and isinstance(markdown, str) and len(markdown) > 500:
                        markdown = markdown[:500] + "... [content truncated]"

                    formatted_results.append(
                        f"{i}. **{title}**\n   URL: {url}\n   {description}\n   Content: {markdown}\n"
                    )

                return "Search Results:\n" + "\n".join(formatted_results)

            elif tool_name == "read_webpage":
                url = arguments.get('url', '')

                logger.info(f"  Tool: read_webpage(url='{url}')")

                scraped = self.firecrawl.scrape(url=url)

                if not scraped:
                    return f"Failed to read webpage: {url}"

                markdown = scraped.get('markdown', '')
                metadata = scraped.get('metadata', {})

                # Store content
                if markdown:
                    self.content_gathered.append(markdown)

                # Truncate if too long
                if len(markdown) > 1500:
                    markdown = markdown[:1500] + "... [content truncated]"

                title = metadata.get('title', 'Untitled')
                return f"Content from {title} ({url}):\n\n{markdown}"

            elif tool_name == "finish_research":
                logger.info("  Tool: finish_research - Research complete!")
                return "Research finished."

            else:
                logger.warning(f"Unknown tool: {tool_name}")
                return f"Error: Unknown tool '{tool_name}'"

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def _run_agent_loop(self, topic: str) -> ResearchResult:
        """
        Run the agent loop:
        1. Send topic + tools to LLM
        2. LLM returns tool calls
        3. Execute tools, append results to conversation
        4. Repeat until finish_research called or max iterations

        Args:
            topic: Topic to research

        Returns:
            ResearchResult
        """
        system_prompt = f"""You are an autonomous research analyst gathering information on: {topic}

CRITICAL: Every tool call must have VALID, NON-EMPTY parameters!

Available tools:
1. web_search(query, limit) - Search the web for information
   - query: STRING of search terms (REQUIRED, must not be empty!)
   - limit: number of results (default: 5)
   
   Example valid calls:
   ✓ web_search(query="latest AI news December 2025", limit=5)
   ✓ web_search(query="Claude Haiku 4.5 features", limit=3)
   
   Example INVALID calls:
   ✗ web_search(query="", limit=5)  <- NEVER use empty query!
   ✗ web_search()  <- Missing required query parameter!

2. read_webpage(url) - Read full content from a specific URL
   - url: STRING of the webpage URL (REQUIRED, must be valid URL!)

3. finish_research(summary, key_facts) - Complete research with findings
   - summary: STRING with comprehensive multi-paragraph summary
   - key_facts: ARRAY of 5-10 key facts discovered

WORKFLOW:
1. First call: web_search with specific search terms about the topic
2. Read promising URLs with read_webpage
3. Gather 3-8 quality sources
4. Call finish_research with summary and key facts

STRICT RULES:
- EVERY tool call MUST have all required parameters filled with valid values
- Query strings must contain actual search terms, never empty strings
- You MUST call at least one tool in every response
- Do NOT provide natural language explanations until calling finish_research
"""

        tools = self._get_tools_definition()
        conversation_history: List[Dict[str, Any]] = []

        initial_prompt = f"Research this topic with the tools and gather current information: {topic}"
        iteration = 0
        finished = False
        forced_prompt: Optional[str] = None
        no_tool_call_streak = 0

        while iteration < self.max_iterations and not finished:
            iteration += 1
            logger.info(f"Research iteration {iteration}/{self.max_iterations}")

            if forced_prompt:
                user_prompt = forced_prompt
                forced_prompt = None
            elif iteration == 1:
                user_prompt = initial_prompt
            else:
                user_prompt = "Continue researching using the available tools. Use web_search first if you still need fresh sources."

            response = self.llm.generate_with_tools(
                prompt=user_prompt,
                tools=tools,
                system=system_prompt,
                context=conversation_history if conversation_history else None
            )

            content = response.get('content', '')
            tool_calls = response.get('tool_calls', [])

            # Record user prompt for future context
            conversation_history.append({"role": "user", "content": user_prompt})

            # Add assistant response to history
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
            formatted_tool_calls = []

            if tool_calls:
                for idx, tool_call in enumerate(tool_calls):
                    tool_call_id = tool_call.get('id') or f"tool_call_{iteration}_{idx}"
                    tool_call['id'] = tool_call_id  # ensure downstream consistency
                    
                    # Convert arguments to JSON string if it's a dict (required by OpenAI/Claude API)
                    arguments = tool_call.get('arguments', {})
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    
                    formatted_tool_calls.append({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.get('name', ''),
                            "arguments": arguments  # Now properly formatted as JSON string
                        }
                    })
                assistant_msg["tool_calls"] = formatted_tool_calls

            conversation_history.append(assistant_msg)

            if content:
                logger.debug(f"LLM reasoning: {content[:200]}...")

            # Handle missing tool calls with a corrective user instruction
            if not tool_calls:
                no_tool_call_streak += 1
                preview = (content[:200] + "...") if content and len(content) > 200 else (content or "<empty>")

                # Enhanced diagnostics
                logger.warning("="*70)
                logger.warning("❌ TOOL CALLING FAILURE")
                logger.warning("="*70)
                logger.warning(f"Model: {self.llm.model}")
                logger.warning(f"Attempt: {no_tool_call_streak}/{self.max_no_tool_attempts}")
                logger.warning(f"Response preview: {preview}")

                # Check for format issues
                if '<tool_call>' in content:
                    logger.warning("⚠️  Detected XML <tool_call> tags (qwen3 format)")
                    logger.warning("💡 This model is incompatible - use llama3.1:8b instead")
                elif 'function' in content.lower() or 'tool' in content.lower():
                    logger.warning("⚠️  Model mentions tools but none were parsed")

                logger.warning("="*70)

                if no_tool_call_streak >= self.max_no_tool_attempts:
                    error_msg = (
                        f"Research failed: Model '{self.llm.model}' doesn't support tool calling properly. "
                        f"Switch to llama3.1:8b in config.yaml (research_server: 'server_research')"
                    )
                    logger.error(error_msg)

                    # Return early with clear error
                    return ResearchResult(
                        summary=f"Research incomplete - tool calling not supported",
                        key_facts=[f"Model '{self.llm.model}' is incompatible with research agent"],
                        sources=self.sources_gathered,
                        raw_content=self._compile_raw_content("Error", [f"Tool calling failed - use llama3.1:8b"]),
                        success=False,
                        error=error_msg
                    )

                forced_prompt = (
                    "You must call web_search or read_webpage before providing any summary. "
                    f"Immediately perform a web_search about: {topic}"
                )
                continue

            no_tool_call_streak = 0

            for idx, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get('name', '')
                arguments = tool_call.get('arguments', {})
                tool_call_id = tool_call.get('id') or f"tool_call_{iteration}_{idx}"

                # Execute tool
                tool_result = self._execute_tool_call(tool_name, arguments)

                # Add tool result to history for follow-up reasoning
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result
                })

                # Check if research is finished
                if tool_name == "finish_research":
                    summary = arguments.get('summary', '')
                    key_facts = arguments.get('key_facts', [])

                    raw_content = self._compile_raw_content(summary, key_facts)

                    return ResearchResult(
                        summary=summary,
                        key_facts=key_facts,
                        sources=self.sources_gathered,
                        raw_content=raw_content,
                        success=True,
                        error=None
                    )

        # If we reach here, agent didn't call finish_research
        logger.warning("Research loop ended without finish_research being called")

        # Compile what we have
        summary = f"Research on {topic} - gathered {len(self.sources_gathered)} sources"
        key_facts = [f"Source {i+1}: {s['title']}" for i, s in enumerate(self.sources_gathered[:5])]

        raw_content = self._compile_raw_content(summary, key_facts)

        return ResearchResult(
            summary=summary,
            key_facts=key_facts,
            sources=self.sources_gathered,
            raw_content=raw_content,
            success=True,
            error="Research completed but agent did not explicitly finish"
        )

    def _compile_raw_content(self, summary: str, key_facts: List[str]) -> str:
        """
        Compile all research into a formatted context string.

        Args:
            summary: Research summary
            key_facts: List of key facts

        Returns:
            Formatted research content
        """
        parts = []

        parts.append("# Research Summary")
        parts.append(summary)
        parts.append("")

        parts.append("# Key Facts")
        for i, fact in enumerate(key_facts, 1):
            parts.append(f"{i}. {fact}")
        parts.append("")

        if self.sources_gathered:
            parts.append("# Sources")
            for i, source in enumerate(self.sources_gathered, 1):
                parts.append(f"{i}. {source['title']}")
                parts.append(f"   {source['url']}")
            parts.append("")

        return "\n".join(parts)
