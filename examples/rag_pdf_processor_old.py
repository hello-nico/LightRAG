"""
RAG PDF处理器
使用RAG domain prompts和Qwen embedding模型处理PDF文件
"""

import os
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

# 导入PDF提取器
from .pdf_reader import PDFExtractor
from .qwen_embedding_config import env_llm_model_func
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()



# 导入LightRAG相关模块
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.prompt import PROMPTS
    from lightrag.constants import DEFAULT_ENTITY_TYPES
    from lightrag.llm.qwen import qwen_embedding_func
    from lightrag.utils import wrap_embedding_func_with_attrs
    from lightrag.kg.shared_storage import initialize_pipeline_status
except ImportError as e:
    logging.error(f"导入LightRAG模块失败: {e}")
    raise


@dataclass
class RAGPDFProcessorConfig:
    """RAG PDF处理器配置"""
    # 基础配置
    pdf_dir: str = "rag_pdfs"
    output_dir: str = "./rag_kg_output"
    rag_prompts_path: str = "datasets/rag_domain_prompts.json"
    
    # Embedding配置
    embedding_base_url: str = "http://10.0.62.206:51200/v1/embeddings"
    embedding_api_key: str = "sk-uHj8K2mNpL5vR9xQ4tY7wB3cA6nE0iF1gD8sZ2yX4jM9kP3h"
    embedding_model: str = "Qwen3-Embedding-0.6B"
    embedding_dimensions: int = 1024
    
    # 从环境变量读取存储配置，保持与API服务器一致                                                                
    kv_storage: str = field(default=os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"))                       
    vector_storage: str = field(default=os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"))         
    graph_storage: str = field(default=os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"))              
    doc_status_storage: str = field(default=os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"))
    
    # LightRAG配置
    llm_model_func: Optional[callable] = env_llm_model_func
    working_dir: str = "./rag_lightrag_storage"
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # # 存储配置
    # vector_storage: str = "QdrantVectorDBStorage"
    # graph_storage: str = "Neo4JStorage"
    # kv_storage: str = "JsonKVStorage"
    
    # # Qdrant配置
    # qdrant_url: str = "http://localhost:6333"
    # qdrant_api_key: str = "abc123"
    
    # # Neo4j配置
    # neo4j_uri: str = "bolt://172.31.64.1:7687"
    # neo4j_username: str = "neo4j"
    # neo4j_password: str = "12345678q"
    
    # 处理配置
    max_concurrent_pdfs: int = 3
    # chunk_size: int = 8000
    # max_gleaning: int = 3
    
    # RAG专用配置
    language: str = "English"


class RAGPDFProcessor:
    """RAG PDF处理器"""
    
    def __init__(self, config: RAGPDFProcessorConfig):
        """
        初始化RAG PDF处理器
        
        Args:
            config: 处理器配置
        """
        self.config = config
        self.pdf_extractor = PDFExtractor()
        self.lightrag_instance = None
        self.rag_prompts = None
        
        # 设置日志
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
        
        # 1. 加载RAG domain prompts
        await self._load_rag_prompts()
        # 2. 初始化LightRAG
        await self._initialize_lightrag()
        
        self.logger.info("RAG PDF处理器初始化完成")
        
    async def _load_rag_prompts(self):
        """加载RAG domain prompts"""
        try:
            if os.path.exists(self.config.rag_prompts_path):
                with open(self.config.rag_prompts_path, 'r', encoding='utf-8') as f:
                    self.rag_prompts = json.load(f)
                self.logger.info(f"成功加载RAG prompts: {self.config.rag_prompts_path}")
            else:
                self.logger.warning(f"RAG prompts文件不存在: {self.config.rag_prompts_path}")
                self.rag_prompts = {}
        except Exception as e:
            self.logger.error(f"加载RAG prompts失败: {e}")
            self.rag_prompts = {}
            
                
    async def _initialize_lightrag(self):
        """初始化LightRAG"""
        try:
            @wrap_embedding_func_with_attrs(embedding_dim=self.config.embedding_dimensions)
            async def embedding_func(texts: List[str]) -> np.ndarray:
                return await qwen_embedding_func(
                    texts=texts,
                    model=self.config.embedding_model,
                    base_url=self.config.embedding_base_url,
                    api_key=self.config.embedding_api_key,
                    dimensions=self.config.embedding_dimensions
                )
            
            # 创建LightRAG配置
            lightrag_config = {
                "working_dir": self.config.working_dir,
                "llm_model_func": self.config.llm_model_func,
                "embedding_func": embedding_func,
                "log_level": self.config.log_level,
                "kv_storage": self.config.kv_storage,
                "vector_storage": self.config.vector_storage,
                "graph_storage": self.config.graph_storage,
                "doc_status_storage": self.config.doc_status_storage,
            }
            
            # # 添加RAG专用配置
            if self.rag_prompts:
                lightrag_config["addon_params"] = {
                    "entity_types": self.rag_prompts.get("entity_types", DEFAULT_ENTITY_TYPES),
                    "language": self.config.language,
                }
                # lightrag_config["entity_extract_max_gleaning"] = self.config.max_gleaning
                # lightrag_config["chunk_token_size"] = self.config.chunk_size
            
            # 创建LightRAG实例
            self.lightrag_instance = LightRAG(**lightrag_config)
            
            # 初始化存储
            await self.lightrag_instance.initialize_storages()
            
            # 初始化管道状态
            await initialize_pipeline_status()
            
            # 注入RAG prompts
            if self.rag_prompts:
                await self._inject_rag_prompts()
                
            self.logger.info("成功初始化LightRAG")
            
        except Exception as e:
            self.logger.error(f"初始化LightRAG失败: {e}")
            raise
            
    async def _inject_rag_prompts(self):
        """注入RAG prompts到LightRAG"""
        try:
            PROMPTS.update(self.rag_prompts)
            # # 更新全局PROMPTS字典
            # if "entity_extraction" in self.rag_prompts:
            #     PROMPTS["entity_extraction"] = self.rag_prompts["entity_extraction"]
            # if "entity_summarization" in self.rag_prompts:
            #     PROMPTS["entity_summarization"] = self.rag_prompts["entity_summarization"]
            # if "relationship_extraction" in self.rag_prompts:
            #     PROMPTS["relationship_extraction"] = self.rag_prompts["relationship_extraction"]
                
                
            self.logger.info("成功注入RAG prompts到LightRAG")
            
        except Exception as e:
            self.logger.error(f"注入RAG prompts失败: {e}")
            
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
                
            param = QueryParam(mode=mode)
            result = await self.lightrag_instance.aquery(query, param=param)
            return result
            
        except Exception as e:
            self.logger.error(f"查询知识图谱失败: {e}")
            raise
        

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


async def main():
    """主函数"""
    # 创建配置
    config = RAGPDFProcessorConfig(
        pdf_dir="rag_pdfs",
        output_dir="./rag_kg_output",
        rag_prompts_path="datasets/rag_domain_prompts.json"
    )
    
    # 创建处理器
    processor = RAGPDFProcessor(config)
    
    try:
        # 初始化
        await processor.initialize()
        
        # 处理所有PDF
        results = await processor.process_all_pdfs()
        
        print(f"处理结果: {results}")
        
        # # 示例查询
        # if results["successful"] > 0:
        #     query_result = await processor.query_knowledge_graph(
        #         "What are the main RAG architectures mentioned in the papers?",
        #         mode="hybrid"
        #     )
        #     print(f"查询结果: {query_result}")
            
    except Exception as e:
        print(f"处理失败: {e}")
        raise e


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"处理时间: {end_time - start_time} 秒")