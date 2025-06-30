
import streamlit as st
from planner import create_agent
from tools import init_vectorstore
import os
import glob

st.set_page_config(page_title="Multi-Agent QA System", layout="centered")
st.title("📄 Multi-Agent Document QA (Planner-based)")

if "agent" not in st.session_state:
    st.session_state.agent = None
if "log" not in st.session_state:
    st.session_state.log = ""

# Upload document
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a .txt file", type="txt")

if uploaded_file and st.session_state.agent is None:
    content = uploaded_file.read().decode("utf-8")
    init_vectorstore(content)
    st.session_state.agent = create_agent()
    st.success("✅ Document uploaded and agent initialized!")

# Ask question
if st.session_state.agent:
    question = st.text_input("Enter your question about the document:")
    if question:
        with st.spinner("Running multi-agent pipeline..."):
            import sys
            from io import StringIO

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()

            result = st.session_state.agent.invoke({"input": question})

            # Reset stdout
            sys.stdout = old_stdout
            log_output = mystdout.getvalue()
            st.session_state.log = log_output

        st.text_area("🔍 Agent Trace Log", value=st.session_state.log, height=300)

        st.success("✅ Final Answer:")
        st.write(result["output"])

        # Show most recent .md file
        md_files = sorted(glob.glob("reports/*.md"), key=os.path.getmtime, reverse=True)
        if md_files:
            with open(md_files[0], "r", encoding="utf-8") as f:
                md_content = f.read()
            st.markdown("### 📝 Latest Markdown Output")
            st.markdown(md_content)
            with open(md_files[0], "rb") as f:
                st.download_button("⬇️ Download Answer (.md)", f, file_name=os.path.basename(md_files[0]), mime="text/markdown")
else:
    st.info("Upload a document to begin.")
