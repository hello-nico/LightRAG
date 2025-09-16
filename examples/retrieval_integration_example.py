"""
LightRAG 检索集成使用示例

此示例展示了如何使用 LightRAG 的检索集成功能。
"""

import asyncio
import logging
from pathlib import Path

# 导入 LightRAG 相关模块
from lightrag import LightRAG
from lightrag.integrations import (
    DeerFlowIntegration,
    IntegrationManager,
    RetrievalRequest,
    BatchRetrievalRequest,
    ListResourcesRequest,
    get_global_config_manager,
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """主函数"""
    # 1. 初始化 LightRAG 实例（这里使用模拟配置）
    # 在实际使用中，你需要配置真实的存储和模型参数

    # 2. 创建集成管理器
    manager = IntegrationManager()

    # 3. 配置 DeerFlow 集成
    deer_flow_config = {
        "type": "deer_flow",
        "config": {
            "rag_instance": None,  # 这里应该传入实际的 LightRAG 实例
            "similarity_threshold": 0.5,
            "default_mode": "mix",
            "max_results": 10,
        },
        "is_default": True,
        "is_enabled": True
    }

    manager.register_integration_from_dict("deer_flow", deer_flow_config)

    # 4. 初始化集成管理器
    await manager.initialize()

    # 5. 示例1：单次检索
    print("=== 示例1：单次检索 ===")
    try:
        retrieval_request = RetrievalRequest(
            query="什么是人工智能？",
            max_results=5,
            min_score=0.3,
            include_metadata=True
        )

        result = await manager.retrieve(retrieval_request)
        print(f"查询: {result.query}")
        print(f"结果数: {result.total_results}")
        print(f"检索时间: {result.retrieval_time:.2f}秒")
        print(f"实体数: {len(result.entities)}")
        print(f"关系数: {len(result.relationships)}")

        # 显示前几个文本块
        for i, chunk in enumerate(result.chunks[:2]):
            print(f"\n块 {i+1}:")
            print(f"ID: {chunk.id}")
            print(f"内容: {chunk.content[:100]}...")
            print(f"相似度: {chunk.similarity}")
            print(f"元数据: {chunk.metadata}")

    except Exception as e:
        print(f"检索失败: {e}")

    # 6. 示例2：批量检索
    print("\n=== 示例2：批量检索 ===")
    try:
        batch_request = BatchRetrievalRequest(
            queries=["机器学习", "深度学习", "神经网络"],
            max_results_per_query=3,
            include_metadata=True
        )

        batch_result = await manager.batch_retrieve(batch_request)
        print(f"总查询数: {len(batch_request.queries)}")
        print(f"成功查询数: {len(batch_result.results)}")

        for i, result in enumerate(batch_result.results):
            print(f"\n查询 {i+1} ({result.query}):")
            print(f"结果数: {result.total_results}")
            print(f"实体数: {len(result.entities)}")
            print(f"关系数: {len(result.relationships)}")

    except Exception as e:
        print(f"批量检索失败: {e}")

    # 7. 示例3：列出资源
    print("\n=== 示例3：列出资源 ===")
    try:
        list_request = ListResourcesRequest(
            limit=10,
            offset=0
        )

        resources = await manager.list_resources(list_request)
        print(f"总资源数: {resources.total_count}")
        print(f"当前页资源数: {len(resources.resources)}")
        print(f"是否还有更多: {resources.has_more}")

        for resource in resources.resources[:5]:
            print(f"\n资源: {resource.id}")
            print(f"类型: {resource.type}")
            print(f"标题: {resource.title}")

    except Exception as e:
        print(f"列出资源失败: {e}")

    # 8. 示例4：健康检查
    print("\n=== 示例4：健康检查 ===")
    try:
        health_status = await manager.health_check()
        print("健康检查结果:")
        for name, status in health_status.items():
            print(f"{name}: {status.get('status', 'unknown')}")

    except Exception as e:
        print(f"健康检查失败: {e}")

    # 9. 清理资源
    await manager.close()
    print("\n演示完成")


async def direct_integration_example():
    """直接使用 DeerFlow 集成的示例"""
    print("\n=== 直接使用 DeerFlow 集成示例 ===")

    try:
        # 创建 DeerFlow 集成实例
        config = {
            "rag_instance": None,  # 这里应该传入实际的 LightRAG 实例
            "similarity_threshold": 0.5,
            "default_mode": "mix",
            "max_results": 5,
        }

        integration = DeerFlowIntegration(config)

        # 使用异步上下文管理器
        async with integration:
            # 检查健康状态
            health = await integration.health_check()
            print(f"集成健康状态: {health.get('status', 'unknown')}")

            # 执行检索
            request = RetrievalRequest(
                query="Python 编程语言",
                max_results=3
            )

            result = await integration.retrieve(request)
            print(f"查询: {result.query}")
            print(f"结果数: {result.total_results}")

            # 显示实体
            if result.entities:
                print("\n相关实体:")
                for entity in result.entities[:3]:
                    print(f"- {entity.name} ({entity.type})")

            # 显示关系
            if result.relationships:
                print("\n相关关系:")
                for rel in result.relationships[:3]:
                    print(f"- {rel.source_entity_id} -> {rel.target_entity_id} ({rel.relation_type})")

    except Exception as e:
        print(f"直接集成示例失败: {e}")


async def configuration_example():
    """配置管理示例"""
    print("\n=== 配置管理示例 ===")

    try:
        # 获取全局配置管理器
        config_manager = get_global_config_manager()

        # 获取当前配置
        config = config_manager.get_config()
        print(f"默认提供商: {config.default_provider}")
        print(f"相似度阈值: {config.similarity_threshold}")
        print(f"默认模式: {config.default_mode}")
        print(f"最大结果数: {config.max_results}")

        # 获取 DeerFlow 配置
        deer_flow_config = config.get_deer_flow_config()
        print(f"\nDeerFlow 配置:")
        for key, value in deer_flow_config.items():
            print(f"  {key}: {value}")

        # 列出所有提供商
        providers = config_manager.list_providers()
        print(f"\n可用提供商: {providers}")

        # 验证配置
        errors = config_manager.validate_config()
        if errors:
            print(f"\n配置验证错误: {errors}")
        else:
            print("\n配置验证通过")

    except Exception as e:
        print(f"配置示例失败: {e}")


async def api_usage_example():
    """API 使用示例"""
    print("\n=== API 使用示例 ===")

    print("当 LightRAG 服务器运行时，你可以使用以下 API 端点：")
    print("\n1. 单次检索:")
    print("   POST /api/v1/retrieve")
    print("   {")
    print('     "query": "你的查询",')
    print('     "max_results": 10,')
    print('     "min_score": 0.3,')
    print('     "include_metadata": true')
    print("   }")

    print("\n2. 批量检索:")
    print("   POST /api/v1/batch")
    print("   {")
    print('     "queries": ["查询1", "查询2"],')
    print('     "max_results_per_query": 5')
    print("   }")

    print("\n3. 列出资源:")
    print("   POST /api/v1/resources")
    print("   {")
    print('     "limit": 50,')
    print('     "offset": 0')
    print("   }")

    print("\n4. 健康检查:")
    print("   GET /api/v1/health")

    print("\n5. 列出集成:")
    print("   GET /api/v1/integrations")

    print("\n使用 curl 的示例:")
    print('curl -X POST "http://localhost:8000/api/v1/retrieve" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"query": "什么是人工智能？", "max_results": 5}\'')


if __name__ == "__main__":
    async def run_examples():
        """运行所有示例"""
        await main()
        await direct_integration_example()
        await configuration_example()
        await api_usage_example()

    # 运行示例
    asyncio.run(run_examples())