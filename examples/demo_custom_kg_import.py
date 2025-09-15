"""
演示如何使用自定义图谱数据导入LightRAG

此demo展示如何：
1. 从JSON文件加载自定义KG数据
2. 初始化LightRAG实例
3. 导入自定义图谱
4. 执行查询测试

使用方法：
python demo_custom_kg_import.py
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
WORKING_DIR = "./custom_kg_demo"
CUSTOM_KG_JSON_PATH = "./examples/example_custom_kg.json"

def load_custom_kg_from_json(json_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载自定义图谱数据

    Args:
        json_path: JSON文件路径

    Returns:
        自定义图谱数据的字典
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查必要的字段
        if not all(k in data for k in ['entities', 'relationships']):
            raise ValueError("JSON文件必须包含 'entities' 和 'relationships' 字段")

        print("✅ 成功加载自定义图谱数据"        print(f"   - 实体数量：{len(data['entities'])}")
        print(f"   - 关系数量：{len(data['relationships'])}")
        print(f"   - 文本块数量：{len(data.get('chunks', []))}")

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

async def import_custom_kg_demo():
    """
    演示自定义图谱导入和查询
    """
    print("=" * 60)
    print("🚀 LightRAG 自定义图谱导入演示")
    print("=" * 60)

    # 1. 创建工作目录
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
        print(f"📁 创建工作目录：{WORKING_DIR}")

    # 2. 加载自定义图谱数据
    print("\n📂 加载自定义图谱数据文件...")
    custom_kg_data = load_custom_kg_from_json(CUSTOM_KG_JSON_PATH)

    # 3. 初始化LightRAG
    print("\n🚀 初始化LightRAG...")
    rag = await initialize_rag()
    if not rag:
        return

    # 4. 导入自定义图谱
    print("\n📊 导入自定义图谱到LightRAG...")
    try:
        await rag.ainsert_custom_kg(custom_kg_data)
        print("✅ 自定义图谱导入成功")

        # 显示导入统计
        print("\n📈 导入统计信息：")
        metadata = custom_kg_data.get('metadata', {})
        if metadata:
            print("   数据集名称：", metadata.get('dataset_name', 'N/A'))
            print("   创建日期：", metadata.get('dataset_created_date', 'N/A'))
            print("   描述：", metadata.get('description', 'N/A'))

        print(".0f"        print(".0f"        print(".0f"
        # 5. 执行查询测试
        print("
🔍 执行查询测试..."        test_queries = [
            ("清华大学在哪里？", "local"),
            ("阿里巴巴总部在哪里？", "global"),
            ("北京有什么特色？", "hybrid"),
            ("哪些公司在发展人工智能？", "hybrid"),
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
                print(f"🤖 回答：{result[:200]}..." if len(result) > 200 else f"🤖 回答：{result}")

            except Exception as e:
                print(f"❌ 查询失败：{e}")

        # 6. 清理演示
        print("
🧹 演示完成，正在清理..."        print("✅ 演示运行完毕")

        print("\n" + "=" * 60)
        print("💡 提示：")
        print("   - 您可以修改 example_custom_kg.json 文件来添加更多数据")
        print("   - 尝试不同的查询模式（local/global/hybrid）来测试效果")
        print("   - 数据会保存在 WORK_DIR 中，您可以复用这些数据")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 导入自定义图谱时出错：{e}")

if __name__ == "__main__":
    print("注意：运行此演示前，请确保设置正确的 OPENAI_API_KEY")
    print("您可以设置环境变量：export OPENAI_API_KEY='your-key-here'")
    print("\n开始演示..."    asyncio.run(import_custom_kg_demo())