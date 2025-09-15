"""
RAG PDF处理器使用示例
展示如何使用RAG PDF处理器处理学术论文
"""

import asyncio
import os
import logging
from pathlib import Path

from .rag_pdf_processor import RAGPDFProcessor, RAGPDFProcessorConfig
from .qwen_embedding_config import get_qwen_config, create_custom_config, env_llm_model_func
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


async def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 创建配置
    config = RAGPDFProcessorConfig(
        pdf_dir="rag_pdfs",
        output_dir="./rag_kg_output",
        rag_prompts_path="datasets/rag_domain_prompts.json",
        max_concurrent_pdfs=3
    )
    
    # 创建处理器
    processor = RAGPDFProcessor(config)
    
    try:
        # 初始化
        await processor.initialize()
        
        # 处理单个PDF文件
        pdf_path = "rag_pdfs/2104.01111_A-Comprehensive-Survey-of-Scene-Graphs-Generation-and-Application.pdf"
        if Path(pdf_path).exists():
            result = await processor.process_pdf(pdf_path)
            print(f"处理结果: {result}")
        else:
            print(f"PDF文件不存在: {pdf_path}")
        
    except Exception as e:
        print(f"处理失败: {e}")


async def initialize_rag():
    """初始化RAG"""
     # 创建处理器配置
    config = RAGPDFProcessorConfig(
        pdf_dir="rag_pdfs",
        llm_model_func=env_llm_model_func,
        max_concurrent_pdfs=3,
        language="English"
    )
    
    # 创建处理器
    processor = RAGPDFProcessor(config)
    
    try:
        # 初始化
        await processor.initialize()
        
        return processor
        
    except Exception as e:
        print(f"初始化失败: {e}")
        return None

async def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 创建处理器配置
    config = RAGPDFProcessorConfig(
        pdf_dir="rag_pdfs",
        llm_model_func=env_llm_model_func,
        max_concurrent_pdfs=3,
        language="English"
    )
    
    # 创建处理器
    processor = RAGPDFProcessor(config)
    
    try:
        # 初始化
        await processor.initialize()
        
        # 处理所有PDF文件
        results = await processor.process_all_pdfs()
        print(f"批量处理结果: {results}")
        
        # 显示统计信息
        if results["status"] == "completed":
            print(f"处理完成:")
            print(f"  - 总计PDF: {results['total_pdfs']}")
            print(f"  - 成功: {results['successful']}")
            print(f"  - 失败: {results['failed']}")
        
    except Exception as e:
        print(f"批量处理失败: {e}")


async def test_rag_query():
    """测试RAG查询示例"""
    print("\n=== 测试RAG查询示例 ===")
    
    # 初始化RAG
    processor = await initialize_rag()

    try:
        query_basic = [
            "什么是RAG？",
            "rag领域影响力最大的论文是哪篇？",
            "解释下什么是naiveRAG？"
        ]
        
        query_reasoning = [
            "为什么构建知识图谱对RAG很重要？",
            "rag技术的演进历程是怎样的？",
            "结合当前rag的发展，分析下未来的发展趋势是怎样的？"
        ]
        
        print(f"当前使用模型: {os.getenv('LLM_MODEL')}")
        
        for query in query_basic:
            print(f"\n查询: {query}")
            try:
                # 本地查询
                local_result = await processor.query(query, mode="local")
                print(f"本地查询结果: {local_result}")
                
                # # 全局查询
                # global_result = await processor.query(query, mode="global")
                # print(f"全局查询结果: {global_result[:200]}...")
                
                # 混合查询
                hybrid_result = await processor.query(query, mode="hybrid")
                print(f"混合查询结果: {hybrid_result}")
                
            except Exception as e:
                print(f"查询失败: {e}")
        
    except Exception as e:
        print(f"知识图谱查询示例失败: {e}")


async def main():
    """主函数 - 运行所有示例"""
    print("RAG PDF处理器使用示例")
    print("=" * 50)
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行示例
    #await example_basic_usage()
    #await example_batch_processing()
    #await example_custom_embedding()
    await test_rag_query()
    #await example_error_handling()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成!")


if __name__ == "__main__":
    asyncio.run(main())