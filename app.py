from typing import TypedDict, Annotated, Optional
from uuid import uuid4
import json

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode

load_dotenv()

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# --- Model and tools ---------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)  # renamed so it no longer clashes with the node function

search_tool = TavilySearch(max_results=4)
tools = [search_tool]

llm_with_tools = llm.bind_tools(tools=tools)

memory = InMemorySaver()


# --- Graph -------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]


async def call_model(state: State, config: RunnableConfig):
    # Pass config explicitly so streaming callbacks reach the model call
    result = await llm_with_tools.ainvoke(state["messages"], config)
    return {"messages": [result]}


def tools_router(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END


tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)
graph_builder.add_node("model", call_model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)


# --- FastAPI app -------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


def sse(payload: dict) -> str:
    """Format a dict as a Server-Sent Event with safely escaped JSON."""
    return f"data: {json.dumps(payload)}\n\n"


def extract_urls(output) -> list[str]:
    """Pull result URLs out of a Tavily tool output (ToolMessage, str, dict or list)."""
    if hasattr(output, "content"):  # ToolMessage
        output = output.content

    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return []

    results = output.get("results", []) if isinstance(output, dict) else output
    if not isinstance(results, list):
        return []

    return [r["url"] for r in results if isinstance(r, dict) and "url" in r]


async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None
    thread_id = str(uuid4()) if is_new_conversation else checkpoint_id
    config = {"configurable": {"thread_id": thread_id}}

    # First message of a conversation: tell the client which thread to reuse
    if is_new_conversation:
        yield sse({"type": "checkpoint", "checkpoint_id": thread_id})

    streamed_any = False

    try:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config,
        ):
            event_type = event["event"]
            name = event.get("name")

            # Token-by-token streaming (when the model events are available)
            if event_type == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                # Skip empty chunks (emitted while the model builds a tool call)
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    streamed_any = True
                    yield sse({"type": "content", "content": chunk.content})

            # Fallback: send the full text when the model node finishes
            elif event_type == "on_chain_end" and name == "model":
                output = event["data"].get("output") or {}
                messages = output.get("messages", []) if isinstance(output, dict) else []
                if messages and not streamed_any:
                    content = messages[-1].content
                    if isinstance(content, str) and content:
                        yield sse({"type": "content", "content": content})
                streamed_any = False  # reset for the next model turn

            # Search started: fires reliably from the tool itself
            elif event_type == "on_tool_start" and name == "tavily_search":
                tool_input = event["data"].get("input", {})
                yield sse({"type": "search_start", "query": tool_input.get("query", "")})

            elif event_type == "on_tool_end" and name == "tavily_search":
                yield sse({"type": "search_results", "urls": extract_urls(event["data"]["output"])})

    except Exception as e:
        # Surface the error to the client instead of silently ending the stream
        yield sse({"type": "error", "message": str(e)})
        raise

    yield sse({"type": "end"})


@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops proxies like nginx from buffering the stream
        },
    )