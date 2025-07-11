import streamlit as st
from tools import init_vectorstore, interpreter_tool, retriever_tool, writer_tool
import os
import glob
import sys
from io import StringIO

st.set_page_config(page_title="Multi-Agent QA System (Manual)", layout="centered")
st.title("📄 Multi-Agent Document QA (Manual Chain)")

if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "log" not in st.session_state:
    st.session_state.log = ""

# Upload document
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a .txt file", type="txt")

if uploaded_file and not st.session_state.initialized:
    content = uploaded_file.read().decode("utf-8")
    init_vectorstore(content)
    st.session_state.initialized = True
    st.success("✅ Document uploaded and vectorstore initialized!")

# Ask question
if st.session_state.initialized:
    question = st.text_input("Enter your question about the document:")
    if question:
        with st.spinner("Running multi-agent chain..."):
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()

            refined = interpreter_tool.invoke(question)
            print("🔍 Refined Question:", refined)

            evidence = retriever_tool.invoke(refined)
            print("📚 Retrieved Evidence:\n", evidence)

            final_answer = writer_tool.invoke(evidence)
            print("✅ Final Answer:\n", final_answer)

            sys.stdout = old_stdout
            log_output = mystdout.getvalue()
            st.session_state.log = log_output

        st.text_area("🧠 Execution Log", value=st.session_state.log, height=300)

        st.success("✅ Final Answer:")
        st.write(final_answer)

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
