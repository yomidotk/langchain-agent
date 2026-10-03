"""Entry point for `langgraph dev` / Agent Chat UI (no MCP, no checkpointer: the server provides memory)."""
from agent import build_agent

agent = build_agent()
