#!/usr/bin/env python3
"""
演示如何导入转换后的domain_kg数据到LightRAG
"""

import os
import json
import asyncio
import sys
from typing import Dict, Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup logger
setup_logger("lightrag", level="INFO")

# 配置参数
WORKING_DIR = "./domain_kg_rag"
DOMAIN_KG_JSON_PATH = "./datasets/domain_kg_lightrag_format.json"

def load_domain_kg_data(json_path: str) -> Dict[str, Any]:
    """
    加载转换后的domain KG数据

    Args:
        json_path: JSON文件路径

    Returns:
        domain KG数据的字典
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查必要的字段
        if not all(k in data for k in ['entities', 'relationships', 'chunks']):
            raise ValueError("JSON文件必须包含 'entities', 'relationships' 和 'chunks' 字段")

        print("✅ 成功加载domain KG数据")
        print(f"   - 实体数量：{len(data['entities'])}")
        print(f"   - 关系数量：{len(data['relationships'])}")
        print(f"   - 文本块数量：{len(data['chunks'])}")

        # 显示元数据
        metadata = data.get('metadata', {})
        if metadata:
            print(f"   - 数据集名称：{metadata.get('dataset_name', 'N/A')}")
            print(f"   - 领域：{metadata.get('domain', 'N/A')}")
            print(f"   - 描述：{metadata.get('description', 'N/A')}")

        return data

    except FileNotFoundError:
        print(f"❌ 找不到文件：{json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误：{e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载JSON文件时出错：{e}")
        sys.exit(1)

async def initialize_rag():
    """
    初始化LightRAG实例

    Returns:
        配置好的LightRAG实例
    """
    try:
        # 注意：实际使用时请替换为您的API密钥
        if "OPENAI_API_KEY" not in os.environ:
            print("⚠️  警告：未设置 OPENAI_API_KEY 环境变量")
            print("   请设置：export OPENAI_API_KEY='your-openai-api-key-here'")
            # 使用一个示例密钥（实际使用时需要替换）
            os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=openai_embed,
        )

        # 初始化存储和流水线
        print("🔄 初始化LightRAG存储...")
        await rag.initialize_storages()
        await initialize_pipeline_status()

        print("✅ LightRAG初始化完成")
        return rag

    except Exception as e:
        print(f"❌ 初始化LightRAG时出错：{e}")
        return None

async def import_domain_kg_demo():
    """
    演示domain KG导入和查询
    """
    print("=" * 60)
    print("🚀 LightRAG Domain KG 导入演示")
    print("=" * 60)

    # 1. 创建工作目录
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
        print(f"📁 创建工作目录：{WORKING_DIR}")

    # 2. 加载domain KG数据
    print("\n📂 加载domain KG数据文件...")
    domain_kg_data = load_domain_kg_data(DOMAIN_KG_JSON_PATH)

    # 3. 初始化LightRAG
    print("\n🚀 初始化LightRAG...")
    rag = await initialize_rag()
    if not rag:
        return

    # 4. 导入domain KG
    print("\n📊 导入domain KG到LightRAG...")
    try:
        await rag.ainsert_custom_kg(domain_kg_data)
        print("✅ domain KG导入成功")

        # 显示导入统计
        print("\n📈 导入统计信息：")
        metadata = domain_kg_data.get('metadata', {})
        if metadata:
            print("   数据集名称：", metadata.get('dataset_name', 'N/A'))
            print("   领域：", metadata.get('domain', 'N/A'))
            print("   实体数量：", metadata.get('total_entities', 'N/A'))
            print("   关系数量：", metadata.get('total_relationships', 'N/A'))

        # 5. 执行查询测试
        print("\n🔍 执行RAG领域相关查询测试...")
        
        test_queries = [
            ("什么是RAG？", "local"),
            ("DPR在RAG中有什么作用？", "global"),
            ("BERT和GPT模型在RAG中的应用对比", "hybrid"),
            ("有哪些评估RAG系统性能的指标？", "local"),
            ("SELF-RAG和传统RAG有什么区别？", "global"),
            ("RAG在对话系统中的应用", "hybrid"),
        ]

        print("\n" + "=" * 40)
        print("测试查询结果：")
        print("=" * 40)

        for question, mode in test_queries:
            print(f"\n❓ 问题：{question}")
            print(f"🔧 查询模式：{mode}")

            try:
                result = await rag.aquery(
                    question,
                    param=QueryParam(mode=mode)
                )
                print(f"🤖 回答：{result[:300]}..." if len(result) > 300 else f"🤖 回答：{result}")

            except Exception as e:
                print(f"❌ 查询失败：{e}")

        # 6. 展示知识图谱查询
        print("\n📊 知识图谱查询示例...")
        
        # 查询特定实体信息
        entity_queries = [
            "DPR",
            "BERT", 
            "SELF-RAG",
            "Patrick Lewis"
        ]
        
        for entity_name in entity_queries:
            print(f"\n🔍 查询实体：{entity_name}")
            try:
                # 使用local模式查询实体相关信息
                result = await rag.aquery(
                    f"Tell me about {entity_name}",
                    param=QueryParam(mode="local")
                )
                print(f"📝 {result[:200]}..." if len(result) > 200 else f"📝 {result}")
            except Exception as e:
                print(f"❌ 查询失败：{e}")

        # 7. 清理和总结
        print("\n" + "=" * 60)
        print("🎉 Domain KG导入演示完成！")
        print("=" * 60)

        print("\n💡 使用提示：")
        print("   • 您的数据已保存在 ./domain_kg_rag 目录中")
        print("   • 可以尝试不同的查询模式：local, global, hybrid")
        print("   • 支持中文和英文查询")
        print("   • 可以查询RAG相关的论文、方法、模型等")

        print("\n📚 支持的查询类型：")
        print("   • 论文信息查询")
        print("   • 作者和机构查询")
        print("   • 方法和模型对比")
        print("   • 评估指标查询")
        print("   • 应用场景查询")

        print("=" * 60)

    except Exception as e:
        print(f"❌ 导入domain KG时出错：{e}")
        import traceback
        traceback.print_exc()

def print_usage_info():
    """打印使用说明"""
    print("""
📖 使用说明
============

1. 环境准备：
   export OPENAI_API_KEY='your-openai-api-key-here'
   pip install lightrag

2. 运行演示：
   python import_domain_kg_demo.py

3. 查询模式说明：
   • local: 基于本地知识图谱的检索
   • global: 基于全局理解的推理
   • hybrid: 混合本地检索和全局推理

4. 数据集信息：
   • 领域：RAG (Retrieval-Augmented Generation)
   • 实体数量：1500+
   • 关系数量：4000+
   • 覆盖内容：论文、作者、方法、模型、数据集、评估指标等
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print_usage_info()
    else:
        print("注意：运行此演示前，请确保设置正确的 OPENAI_API_KEY")
        print("您可以设置环境变量：export OPENAI_API_KEY='your-key-here'")
        print("\n开始演示...")
        asyncio.run(import_domain_kg_demo())