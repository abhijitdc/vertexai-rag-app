import streamlit as st
from datastorehelper import SearchHelper
from checkgrounding import GroundingChecker
from PIL import Image
import logging

logger = logging.getLogger(__name__)

st.set_page_config(layout="wide")


# st.image(image, caption='Sunrise by the mountains')
def context_cols(page_content, metadata, cnt):
    # st.divider()
    with st.expander(label=f"**:gray[Document-{cnt}]**"):
        st.caption(page_content)
        st.divider()
        st.write(f"Metadata: **:green[{metadata}]**")


@st.cache_resource
def getVertexAIDataStore(projectid, datastoreid):
    return SearchHelper(PROJECT_ID=projectid, DATA_STORE_ID=datastoreid)


# sidebar
with st.sidebar:
    st.header("🔎 Vertex AI Search")
    # "with LangChain 🦜"
    st.divider()

    # st.subheader("VertexAI Data Store Details")
    PROJECT_ID = st.text_input(
        value="[PROJECT_ID]", placeholder="", label="**:blue[Project]**"
    )
    DO_TRANSLATE = st.selectbox(
        "**:blue[Language]**", ("English", "Hindi", "Chineese", "French", "German")
    )

    DATA_STORE_ID = st.selectbox(
        "**:blue[Data Store]**",
        ("[DATASTORE_ID]", "[DATASTORE_ID]", "[DATASTORE_ID]", "[DATASTORE_ID]"),
    )
    # st.write(f"Selected DataStore: **:blue[{DATA_STORE_ID}]**")

    rank_cnt = st.slider(
        "**:blue[Number of documents for grounding]**",
        min_value=1,
        max_value=10,
        value=10,
    )
    # st.write(f"Maximum number of documents for grouding: **:blue[{rank_cnt}]**")

    st.divider()

    "**:orange[Try these questions.....]**"
    "What is vaping and how it affects the teens?"
    "What are the chemicals in vape?"
    "How to talk to a teen about vape?"
    "What are the annual leave policy?"
    "How is casual employment rates are calculated?"
    "How can I complain about the policy?"
    "What are the license types?"
    "How long I have to drive on probation?"


vxds = getVertexAIDataStore(PROJECT_ID, DATA_STORE_ID)


search_tab, chat_tab, design_tab = st.tabs(
    ["**:brown[Search]**", "**:brown[Chat]**", "**:brown[Design]**"]
)

with search_tab:
    st.session_state["messages"] = []
    with st.container():
        if prompt := st.chat_input("Search here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            ranked_docs = vxds.searchWithRanksDocs(
                search_query=prompt, rnk_cnt=rank_cnt, trans_lang=DO_TRANSLATE
            )
            if ranked_docs is not None and len(ranked_docs) > 0:
                cnt = 1
                for doc in ranked_docs:
                    context_cols(doc.page_content, doc.metadata, cnt)
                    # context_cols(doc['page_content'],doc['p_metadata'], cnt)
                    cnt += 1
            else:
                with st.expander(label=f"**:red[No results found]**"):
                    st.caption("No results found")
                    st.divider()
                    st.write(f"Metadata: **:red[No results found]**")

with chat_tab:
    st.session_state["messages"] = []
    with st.container():
        if prompt := st.chat_input(key="chat_box", placeholder="Search here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            llm_msg = vxds.searchWithRanks(
                search_query=prompt, rnk_cnt=rank_cnt, trans_lang=DO_TRANSLATE
            )
            # print(ranked_docs)
            if llm_msg is not None and len(llm_msg) > 0:
                st.session_state.messages.append(llm_msg)
                st.chat_message("assistant").write(llm_msg)
                gChecker = GroundingChecker()
                ranked_docs = vxds.searchWithRanksDocs(
                    search_query=prompt, rnk_cnt=rank_cnt, trans_lang=DO_TRANSLATE
                )
                gResult = gChecker.check_grounding(llm_msg, ranked_docs)
                st.subheader("Grounding Check")
                st.code(gResult)

            else:
                st.chat_message("assistant").write("No results found")

with design_tab:
    st.image("./assets/design.png")
