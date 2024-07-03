from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import VertexAI
from langchain_google_community import VertexAISearchRetriever
from langchain_google_community.vertex_rank import VertexAIRank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
import json
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
import logging
import vertexai
from langchain.retrievers import ContextualCompressionRetriever

logging.getLogger().setLevel(logging.DEBUG)


class SearchHelper:

    REGION = "us-central1"  # @param {type:"string"}
    MODEL = "gemini-1.5-flash-001"  # "text-bison@002" #"gemini-1.0-pro"

    def __init__(
        self,
        PROJECT_ID="[PROJECT_ID]",
        DATA_STORE_ID="[VERTEXAI_DATASET_ID]",
        DATA_STORE_LOCATION="global",
        rank_cnt=10,
    ):
        self.PROJECT_ID = PROJECT_ID  # @param {type:"string"}
        self.DATA_STORE_LOCATION = DATA_STORE_LOCATION  # @param {type:"string"}
        self.DATA_STORE_ID = DATA_STORE_ID  # @param {type:"string"}
        self.rank_cnt = rank_cnt  # @param {type:"integer"}
        self.initLlm()

    def initLlm(self):

        vertexai.init(project=self.PROJECT_ID, location=self.REGION)
        self.llm = VertexAI(model_name=self.MODEL)

        self.retriever = VertexAISearchRetriever(
            project_id=self.PROJECT_ID,
            location_id=self.DATA_STORE_LOCATION,
            data_store_id=self.DATA_STORE_ID,
            get_extractive_answers=True,
            max_documents=self.rank_cnt,
            max_extractive_segment_count=1,
            max_extractive_answer_count=1,
            engine_data_type=0,
        )

        self.reranker = VertexAIRank(
            project_id=self.PROJECT_ID,
            location_id=self.DATA_STORE_LOCATION,
            ranking_config="default_ranking_config",
            title_field="source",
            top_n=self.rank_cnt,
            engine_data_type=0,
        )

    def search(self, search_query, rnk_cnt=10, trans_lang="English"):

        self.retriever.max_documents = rnk_cnt
        self.reranker.top_n = rnk_cnt

        template = """
            <context>
            {context}
            </context>

            Question:
            {query}

            You must always use the context and context only to answer the question. If the context is empty or you do not know the answer, just say "I don't know". Don't give information outside the context or repeat your findings. Provide the answer in {language}
            Answer:
            """
        prompt = PromptTemplate.from_template(template)

        setup_and_retrieval = RunnableParallel(
            {
                "context": self.retriever,
                "query": RunnablePassthrough(),
                "language": RunnableLambda(lambda x: trans_lang),
            }
        )

        llm_chain = setup_and_retrieval | prompt | self.llm

        results = llm_chain.invoke(search_query)

        return results

    def searchWithRanks(self, search_query, rnk_cnt=10, trans_lang="English"):

        retriever_with_reranker = ContextualCompressionRetriever(
            base_retriever=self.retriever, base_compressor=self.reranker
        )

        self.retriever.max_documents = rnk_cnt
        self.reranker.top_n = rnk_cnt

        template = """
            <context>
            {context}
            </context>

            Question:
            {query}

            You are a friendly AI assistant. You must always use the context and context only to answer the question. Never try to make up an answer. If the context is empty or you do not know the answer, just say "I don't know".
            Don't give information outside the context or repeat your findings. Provide the answer in {language}
            Answer:
            """
        prompt = PromptTemplate.from_template(template)

        reranker_setup_and_retrieval = RunnableParallel(
            {
                "context": retriever_with_reranker,
                "query": RunnablePassthrough(),
                "language": RunnableLambda(lambda x: trans_lang),
            }
        )
        llm_chain = reranker_setup_and_retrieval | prompt | self.llm

        results = llm_chain.invoke(search_query)

        return results

    def searchWithRanksDocs(self, search_query, rnk_cnt=10, trans_lang="English"):

        retriever_with_reranker = ContextualCompressionRetriever(
            base_retriever=self.retriever, base_compressor=self.reranker
        )

        self.retriever.max_documents = rnk_cnt
        self.reranker.top_n = rnk_cnt

        ranked_docs = retriever_with_reranker.invoke(search_query)

        return ranked_docs


if __name__ == "__main__":
    sh = SearchHelper()
    show_results = sh.searchWithRanksDocs(
        "What is vaping and how it affects the teens?"
    )
    print(show_results)
