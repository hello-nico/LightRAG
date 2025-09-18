"""
RAG PDF处理器 - 使用统一配置系统

使用RAG domain prompts和Qwen embedding模型处理PDF文件，
现在使用LightRAG核心配置系统和实例管理器。
"""

import os
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# 导入PDF提取器
from .pdf_reader import PDFExtractor
from .qwen_embedding_config import env_llm_model_func
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入LightRAG核心模块
from lightrag.core import (
    load_core_config,
    get_lightrag_instance,
    inject_prompts_from_config,
    initialize_lightrag_with_config,
)


@dataclass
class RAGPDFProcessorConfig:
    """RAG PDF处理器配置"""
    # 基础配置
    pdf_dir: str = "rag_pdfs"
    output_dir: str = "./rag_kg_output"
    instance_name: str = "rag_pdf_processor"

    # 实例管理配置
    auto_init: bool = True
    force_recreate: bool = False

    # 处理配置
    max_concurrent_pdfs: int = 3

    # 日志配置
    enable_logging: bool = True
    log_level: str = "INFO"


class RAGPDFProcessor:
    """RAG PDF处理器 - 使用统一配置系统"""

    def __init__(self, config: RAGPDFProcessorConfig):
        """
        初始化RAG PDF处理器

        Args:
            config: 处理器配置
        """
        self.config = config
        self.pdf_extractor = PDFExtractor()
        self.lightrag_instance = None

        # 设置日志
        if config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

    async def initialize(self):
        """初始化处理器"""
        self.logger.info("初始化RAG PDF处理器...")

        # 加载核心配置
        core_config = load_core_config()
        core_config.instance_name = self.config.instance_name

        # 可选：覆盖某些配置项
        core_config.working_dir = "./rag_lightrag_storage"
        core_config.input_dir = self.config.pdf_dir

        # 创建Qwen embedding构建器
        def create_qwen_embedding_rag(config):
            """创建使用Qwen embedding的LightRAG实例"""
            from lightrag.utils import wrap_embedding_func_with_attrs
            from lightrag.llm.qwen import qwen_embedding_func
            import numpy as np

            @wrap_embedding_func_with_attrs(embedding_dim=1024)
            async def embedding_func(texts: List[str]) -> np.ndarray:
                return await qwen_embedding_func(
                    texts=texts,
                    model=os.getenv("QWEN_EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"),
                    base_url=os.getenv("QWEN_EMBEDDING_HOST", "http://localhost:51200/v1/embeddings"),
                    api_key=os.getenv("QWEN_EMBEDDING_API_KEY", "your-api-key-here"),
                    dimensions=int(os.getenv("EMBEDDING_DIM", "1024"))
                )

            # 创建LightRAG实例
            from lightrag import LightRAG

            rag_config = {
                "working_dir": config.working_dir,
                "llm_model_func": env_llm_model_func,
                "embedding_func": embedding_func,
                "log_level": self.config.log_level,
            }

            # 添加存储配置（从环境变量读取）
            if config.kv_storage:
                rag_config["kv_storage"] = config.kv_storage
            if config.vector_storage:
                rag_config["vector_storage"] = config.vector_storage
            if config.graph_storage:
                rag_config["graph_storage"] = config.graph_storage
            if config.doc_status_storage:
                rag_config["doc_status_storage"] = config.doc_status_storage

            # 添加工作空间
            if config.workspace:
                rag_config["workspace"] = config.workspace

            # 添加配置参数
            addon_params = {
                "language": config.summary_language,
                "entity_types": config.entity_types,
            }

            # 添加自定义配置
            addon_params.update(config.custom_config)
            rag_config["addon_params"] = addon_params

            return LightRAG(**rag_config)

        # 获取或创建LightRAG实例
        self.lightrag_instance = await get_lightrag_instance(
            config=core_config,
            builder=create_qwen_embedding_rag,
            auto_init=self.config.auto_init,
            force_recreate=self.config.force_recreate
        )

        # 注入PROMPTS（如果配置了prompts路径）
        await inject_prompts_from_config(self.lightrag_instance, core_config)

        # 如果未自动初始化，手动初始化存储
        if not self.config.auto_init:
            await self.lightrag_instance.initialize_storages()

        self.logger.info("RAG PDF处理器初始化完成")

    async def process_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        处理单个PDF文件

        Args:
            pdf_path: PDF文件路径

        Returns:
            处理结果字典
        """
        try:
            self.logger.info(f"开始处理PDF: {pdf_path}")

            # 1. 提取PDF内容
            extracted_content = self.pdf_extractor.extract(pdf_path)
            if not extracted_content:
                self.logger.error(f"PDF内容提取失败: {pdf_path}")
                return None

            # 2. 使用LightRAG进行处理
            if self.lightrag_instance:
                # 将论文内容插入到LightRAG
                doc_id = Path(pdf_path).stem
                await self.lightrag_instance.ainsert(extracted_content.full_text, doc_id)

                # 提取实体和关系
                from lightrag import QueryParam
                entities = await self.lightrag_instance.aquery(
                    f"Extract entities from the paper about {extracted_content.paper_id}",
                    param=QueryParam(mode="local")
                )

                self.logger.info(f"成功处理PDF: {pdf_path}")

                return {
                    "doc_id": doc_id,
                    "metadata": extracted_content.metadata,
                    "entities": entities,
                    "status": "success"
                }
            else:
                self.logger.error("LightRAG实例未初始化")
                return None

        except Exception as e:
            self.logger.error(f"处理PDF失败 {pdf_path}: {e}")
            return None

    async def process_all_pdfs(self) -> Dict[str, Any]:
        """
        处理所有PDF文件

        Returns:
            处理结果汇总
        """
        try:
            self.logger.info(f"开始处理PDF目录: {self.config.pdf_dir}")

            # 获取所有PDF文件
            pdf_files = []
            for file in os.listdir(self.config.pdf_dir):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(self.config.pdf_dir, file))

            if not pdf_files:
                self.logger.warning(f"没有找到PDF文件: {self.config.pdf_dir}")
                return {"status": "no_files", "processed": 0, "failed": 0}

            self.logger.info(f"找到 {len(pdf_files)} 个PDF文件")

            # 批量处理
            results = []
            successful = 0
            failed = 0

            # 使用信号量控制并发
            semaphore = asyncio.Semaphore(self.config.max_concurrent_pdfs)

            async def process_with_semaphore(pdf_path: str):
                async with semaphore:
                    return await self.process_pdf(pdf_path)

            # 并发处理所有PDF
            tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_files]
            process_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for i, result in enumerate(process_results):
                if isinstance(result, Exception):
                    self.logger.error(f"处理PDF失败 {pdf_files[i]}: {result}")
                    failed += 1
                elif result and result.get("status") == "success":
                    results.append(result)
                    successful += 1
                else:
                    self.logger.error(f"处理PDF失败: {pdf_files[i]}")
                    failed += 1

            # 保存处理结果
            result_summary = {
                "total_pdfs": len(pdf_files),
                "successful": successful,
                "failed": failed,
                "results": results,
                "status": "completed"
            }

            # 保存到文件
            output_file = os.path.join(self.config.output_dir, "processing_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_summary, f, ensure_ascii=False, indent=2)

            self.logger.info(f"PDF处理完成: 成功 {successful}, 失败 {failed}")
            return result_summary

        except Exception as e:
            self.logger.error(f"批量处理PDF失败: {e}")
            return {"status": "error", "error": str(e), "processed": 0, "failed": 0}

    async def query(self, query: str, mode: str = "hybrid") -> str:
        """
        直接调用lightrag实例进行查询

        Args:
            query: 查询文本
            mode: 查询模式 (local, global, hybrid, naive, mix, bypass)

        Returns:
            查询结果
        """
        try:
            if not self.lightrag_instance:
                raise ValueError("LightRAG实例未初始化")

            from lightrag import QueryParam
            param = QueryParam(mode=mode)
            result = await self.lightrag_instance.aquery(query, param=param)
            return result

        except Exception as e:
            self.logger.error(f"查询知识图谱失败: {e}")
            raise

    async def close(self):
        """清理资源"""
        if self.lightrag_instance:
            try:
                # 注意：这里不销毁实例，只是清理当前处理器的引用
                # 实例会由全局管理器统一管理
                self.lightrag_instance = None
                self.logger.info("RAG PDF处理器已关闭")
            except Exception as e:
                self.logger.warning(f"关闭处理器时出错: {e}")


async def initialize_rag():
    """初始化RAG - 使用统一配置系统"""
    # 创建处理器配置
    config = RAGPDFProcessorConfig(
        pdf_dir="rag_pdfs",
        instance_name="rag_pdf_processor",
        max_concurrent_pdfs=3,
        auto_init=True,
        force_recreate=False
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


async def main():
    """主函数 - 演示新的统一配置系统"""
    # 创建配置
    config = RAGPDFProcessorConfig(
        pdf_dir="rag_pdfs",
        output_dir="./rag_kg_output",
        instance_name="rag_pdf_processor_demo",
        max_concurrent_pdfs=3,
        auto_init=True
    )

    # 创建处理器
    processor = RAGPDFProcessor(config)

    try:
        # 初始化
        await processor.initialize()

        # 处理所有PDF
        results = await processor.process_all_pdfs()

        print(f"处理结果: {results}")

        # 示例查询
        if results["successful"] > 0:
            query_result = await processor.query(
                "What are the main RAG architectures mentioned in the papers?",
                mode="hybrid"
            )
            print(f"查询结果: {query_result}")

    except Exception as e:
        print(f"处理失败: {e}")
        raise e
    finally:
        await processor.close()


def demo_direct_usage():
    """演示直接使用统一配置系统"""
    async def demo():
        # 1. 加载配置
        from lightrag.core import load_core_config, get_default_config

        # 使用预设配置
        config = get_default_config("rag")
        config.working_dir = "./rag_demo_storage"
        config.input_dir = "./rag_pdfs"

        # 2. 获取实例
        rag = await initialize_lightrag_with_config(config)

        # 3. 使用实例
        print(f"获取到实例: {rag}")

        # 4. 查询实例信息
        from lightrag.core import get_global_manager
        manager = get_global_manager()
        instance_names = await manager.get_instance_names()
        print(f"当前管理的实例: {instance_names}")

    asyncio.run(demo())


if __name__ == "__main__":
    # 选择运行模式
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # 运行演示模式
        demo_direct_usage()
    else:
        # 运行主程序
        start_time = time.time()
        asyncio.run(main())
        end_time = time.time()
        print(f"处理时间: {end_time - start_time} 秒")