from langgraph.graph import StateGraph, START, END
from typing import Annotated, Any, Dict, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.messages.tool import ToolCall
#from langchain_openai import ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading
import uuid
import json
import os
import tempfile
from groq import AsyncGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
# this creates a new empty asyncio loop, separate from the main python loop 
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()
# this allows to offload async without blocking the main thread 


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)
# this takes a coroutine (async function), schedules it on the background async loop returns a concurrent.futures.future 

def run_async(coro):
    return _submit_async(coro).result()
# this allows to use async code from sync code 

def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)
# this schedule the async coroutine, immediately return a future and do not wait for result 

# -------------------
# 1. LLM + embeddings 
# -------------------
#llm = ChatOpenAI()
_GROQ_MODEL = "llama-3.1-8b-instant"

# Global variables for Groq LLM configuration
_groq_client = None # will store the single shared groq client instance so the code can reuse it instead of creating a new client each time 
_groq_tools = None  # will hold the list/dict of tools (functions) you want the model to be able to call 

def _init_groq_client():
    """Initialize Groq client with API key."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY must be set in env")
        _groq_client = AsyncGroq(api_key=api_key)
    return _groq_client

def bind_tools_to_groq(tools):
    """Bind tools for function calling."""
    global _groq_tools
    _groq_tools = tools
    return {"tools": _groq_tools}  # Return dict to maintain interface

def convert_tool_to_groq_function(tool):
    """Convert LangChain tool to Groq function format."""
    try:
        # Get tool schema with better error handling
        schema = {}
        if hasattr(tool, 'args_schema') and tool.args_schema:
            if hasattr(tool.args_schema, 'schema'):
                schema = tool.args_schema.schema()
            elif isinstance(tool.args_schema, dict):
                schema = tool.args_schema
        
        # Fix properties to ensure they have proper types
        properties = schema.get("properties", {})
        fixed_properties = {}
        for prop_name, prop_def in properties.items():
            if isinstance(prop_def, dict):
                # Ensure each property has a type, infer from name if missing
                if "type" not in prop_def:
                    # Infer type based on property name and content
                    if "amount" in prop_name.lower() or "price" in prop_name.lower():
                        prop_def["type"] = "number"
                    elif "date" in prop_name.lower():
                        prop_def["type"] = "string"
                        prop_def["format"] = "date"
                    else:
                        prop_def["type"] = "string"  # Default to string
                fixed_properties[prop_name] = prop_def
            else:
                # If property definition is not a dict, create a basic string property
                fixed_properties[prop_name] = {"type": "string", "description": str(prop_def)}
        
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": fixed_properties,
                    "required": schema.get("required", [])
                }
            }
        }
    except Exception as e:
        print(f"Schema conversion error for {tool.name}: {e}")
        # Fallback for tools without proper schema
        return {
            "type": "function", 
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }

async def groq_ainvoke(messages: list[BaseMessage], model: str = _GROQ_MODEL) -> BaseMessage:
    """Invoke Groq with proper tool calling support."""
    client = _init_groq_client()
    
    # Convert messages to Groq format
    groq_msgs = []
    for m in messages:
        if isinstance(m, ToolMessage):
            # Handle tool result messages
            groq_msgs.append({
                "role": "tool",
                "content": str(m.content),
                "tool_call_id": m.tool_call_id
            })
        else:
            # Handle regular messages
            role = getattr(m, "role", None)
            if role is None:
                cls_name = m.__class__.__name__.lower()
                if "human" in cls_name or "user" in cls_name:
                    role = "user"
                elif "system" in cls_name:
                    role = "system"
                elif "assistant" in cls_name or "ai" in cls_name:
                    role = "assistant"
                else:
                    role = "user"
            
            msg_dict = {"role": role, "content": str(m.content)}
            
            # Handle tool calls in assistant messages
            if hasattr(m, 'tool_calls') and m.tool_calls:
                msg_dict["tool_calls"] = []
                for tc in m.tool_calls:
                    msg_dict["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    })
            
            groq_msgs.append(msg_dict)

    # Prepare API call parameters
    api_params = {
        "messages": groq_msgs,
        "model": model,
    }
    
    # Add tools if available
    global _groq_tools
    if _groq_tools:
        functions = [convert_tool_to_groq_function(tool) for tool in _groq_tools]
        print(f"Sending {len(functions)} tools to Groq:")
        for func in functions:
            print(f"  - {func['function']['name']}: {func['function']['parameters']}")
        api_params["tools"] = functions
        api_params["tool_choice"] = "auto"

    # Call Groq API
    completion = await client.chat.completions.create(**api_params)
    
    # Process response
    choice = completion.choices[0]
    message = choice.message
    
    # Handle tool calls
    if message.tool_calls:
        tool_calls = []
        for tc in message.tool_calls:
            try:
                # Parse arguments safely
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_calls.append({
                    "name": tc.function.name,
                    "args": args,
                    "id": tc.id
                })
            except json.JSONDecodeError as e:
                print(f"Failed to parse tool call arguments: {e}")
                print(f"Raw arguments: {tc.function.arguments}")
                # Skip malformed tool calls
                continue
        
        return AIMessage(
            content=message.content or "",
            tool_calls=tool_calls
        )
    else:
        # Regular response
        return AIMessage(content=message.content or "")

# Create a simple wrapper object to maintain interface compatibility
class GroqLLMWrapper:
    """Simple wrapper to maintain interface compatibility."""
    def bind_tools(self, tools):
        bind_tools_to_groq(tools)
        return self
    
    async def ainvoke(self, messages: list[BaseMessage]) -> BaseMessage:
        return await groq_ainvoke(messages)

# Initialize LLM wrapper
llm = GroqLLMWrapper()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

client = MultiServerMCPClient(
    {
        #"arith": {
        #    "transport": "stdio",
        #    "command": "python3",
        #    "args": ["/Users/nitish/Desktop/mcp-math-server/main.py"],
        #},
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        tools = run_async(client.get_tools())
        print(f"Loaded {len(tools)} MCP tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            try:
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    if hasattr(tool.args_schema, 'schema'):
                        schema = tool.args_schema.schema()
                        print(f"  Schema: {schema}")
                    else:
                        print(f"  Schema: {tool.args_schema}")
            except Exception as schema_error:
                print(f"  Schema error: {schema_error}")
        return tools
    except Exception as e:
        print(f"Failed to load MCP tools: {e}")
        return []


# Re-enable MCP tools to investigate the schema issue
mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, rag_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price"
            "tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = state["messages"]
    # Add system message if this is the first message in the conversation
    if len(messages) == 1:
        messages = [system_message] + messages
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


# -------------------
# 8. Main Execution
# -------------------
async def main():
    """Main function to run the chatbot interactively."""
    print("🤖 LangGraph MCP Chatbot")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'threads' to see all conversation threads")
    print("-" * 50)
    
    # Get or create thread ID
    thread_id = input("Enter thread ID (or press Enter for new thread): ").strip()
    if not thread_id:
        thread_id = str(uuid.uuid4())[:8]
        print(f"Created new thread: {thread_id}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            user_input = input(f"\n[{thread_id}] You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'threads':
                threads = retrieve_all_threads()
                print(f"Available threads: {threads}")
                continue
            elif not user_input:
                continue
            
            # Send message to chatbot
            print("🤖 Assistant: ", end="", flush=True)
            
            async for event in chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config
            ):
                for value in event.values():
                    if "messages" in value and value["messages"]:
                        message = value["messages"][-1]
                        if hasattr(message, 'content'):
                            print(message.content)
                        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Failed to start chatbot: {e}")
        print("\nMake sure you have:")
        print("1. Created a .env file with your GROQ_API_KEY")
        print("2. Installed all dependencies: pip install -r requirements.txt")