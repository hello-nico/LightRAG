import asyncio
from lightrag.integrations import DeerFlowRetriever

deerflow_retriever = DeerFlowRetriever()


async def main():
    resources = await deerflow_retriever.list_resources()
    print(resources)
    documents = await deerflow_retriever.query_relevant_documents("什么是RAG", resources)
    print(documents)

if __name__ == "__main__":
    asyncio.run(main())