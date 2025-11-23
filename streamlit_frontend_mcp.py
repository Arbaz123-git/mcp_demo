import queue
import uuid

import streamlit as st
from langgraph_mcp_chatbot import chatbot, retrieve_all_threads, submit_async_task, ingest_pdf, thread_has_document, thread_document_metadata
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ============================ PDF Upload ============================
st.sidebar.header("📄 Document Upload")

# Check if current thread has a document
current_thread_id = str(st.session_state["thread_id"])
has_doc = thread_has_document(current_thread_id)

if has_doc:
    doc_metadata = thread_document_metadata(current_thread_id)
    st.sidebar.success(f"✅ Document loaded: {doc_metadata.get('filename', 'Unknown')}")
    st.sidebar.info(f"📊 {doc_metadata.get('documents', 0)} pages, {doc_metadata.get('chunks', 0)} chunks")
else:
    st.sidebar.info("📝 No document uploaded for this chat")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF to ask questions about it",
    type=["pdf"],
    help="Upload a PDF document to enable document-based Q&A using the RAG tool"
)

if uploaded_file is not None:
    if not has_doc:  # Only process if no document is already loaded
        with st.spinner("Processing PDF..."):
            try:
                # Read file bytes
                file_bytes = uploaded_file.read()
                
                # Process the PDF
                result = ingest_pdf(file_bytes, current_thread_id, uploaded_file.name)
                
                st.sidebar.success(f"✅ Successfully processed: {result['filename']}")
                st.sidebar.info(f"📊 {result['documents']} pages, {result['chunks']} chunks")
                
                # Rerun to update the UI
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"❌ Error processing PDF: {str(e)}")
    else:
        st.sidebar.warning("⚠️ A document is already loaded for this chat. Start a new chat to upload a different document.")

# ============================ Main UI ============================

# Show helpful message if no document is uploaded and no conversation history
if not has_doc and len(st.session_state["message_history"]) == 0:
    st.info("💡 **Welcome to the MCP Chatbot!** You can:\n\n"
           "• 📄 **Upload a PDF** in the sidebar to ask questions about documents\n"
           "• 🔍 **Search the web** - try: 'Search for latest AI news'\n"
           "• 📈 **Get stock prices** - try: 'What's Apple's stock price?'\n"
           "• 💰 **Track expenses** - try: 'Add expense of $50 for groceries today'")

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )