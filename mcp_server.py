"""Tiny MCP server. agent.py launches it over stdio and loads its tools."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("study_tools")


@mcp.tool()
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


@mcp.tool()
def make_flashcard(question: str, answer: str) -> str:
    """Format a question/answer pair as a flashcard."""
    return f"Q: {question}\nA: {answer}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
