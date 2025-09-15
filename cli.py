"""
LightRAG CLI 工具
基于统一实例化对象的6种query模式实现
"""
import asyncio
import typer
from pathlib import Path
from typing import Optional
from examples.rag_pdf_processor import initialize_rag
from lightrag import QueryParam
import os

app = typer.Typer(
    name="lightrag-cli",
    help="LightRAG CLI 工具 - 支持6种查询模式的RAG系统",
    add_completion=False
)


class Args:
    """模拟命令行参数对象，使用配置文件中的默认值"""
    def __init__(self, model: str = "grok-code-fast-1"):
        self.api_base = os.getenv("LLM_BINDING_HOST", "http://10.0.62.214:15000/k-llm")
        self.api_key = os.getenv("LLM_BINDING_API_KEY", "sk-4e76fcdf3f95467198edabdc0d6627f6")
        self.model = model


def print_help():
    """打印帮助信息"""
    help_text = """
LightRAG CLI 工具 - 使用说明

支持的查询模式：
- local: 本地查询，基于上下文相关信息
- global: 全局查询，利用全局知识
- hybrid: 混合查询，结合本地和全局检索方法
- naive: 朴素查询，执行基本搜索而不使用高级技术
- mix: 混合查询，集成知识图谱和向量检索
- bypass: 绕过查询，直接使用LLM回答

流式输出：
- 默认启用流式输出（--stream 或 -s）
- 使用 --no-stream 或 -ns 禁用流式输出

使用示例：
    # 基本查询（默认流式输出）
    python -m cli -q local "什么是RAG"

    # 指定模型（默认流式输出）
    python -m cli -m gpt-5 -q local "什么是RAG"

    # 禁用流式输出
    python -m cli -q local "什么是RAG" --no-stream

    # 强制启用流式输出
    python -m cli -q local "什么是RAG" --stream

    # 查看帮助
    python -m cli --help
    """
    print(help_text)


@app.command()
def query(
    query_text: str = typer.Argument(..., help="查询文本"),
    mode: str = typer.Option("local", "--mode", "-q", help="查询模式: local, global, hybrid, naive, mix, bypass"),
    model: str = typer.Option("grok-code-fast-1", "--model", "-m", help="使用的模型名称"),
    stream: bool = typer.Option(True, "--stream/--no-stream", "-s/-ns", help="是否使用流式输出，默认为True")
):
    """
    执行查询

    示例：
    python -m cli -q local "什么是RAG"
    python -m cli -m gpt-5 -q hybrid "RAG系统的主要组成部分"
    """
    # 验证查询模式
    valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
    if mode not in valid_modes:
        typer.echo(f"错误：无效的查询模式 '{mode}'，支持的模式: {', '.join(valid_modes)}", err=True)
        raise typer.Exit(1)

    # 创建参数对象
    args = Args(model)

    # 执行查询
    try:
        typer.echo(f"🚀 正在使用模型 {args.model} 以 {mode} 模式查询...")
        typer.echo(f"📝 查询内容: {query_text}")
        typer.echo("-" * 60)

        if stream:
            typer.echo("💬 查询结果 (流式输出):")
            asyncio.run(execute_streaming_query(query_text, mode, args))
        else:
            typer.echo("⏳ 正在查询...")
            result = asyncio.run(execute_query(query_text, mode, args))
            typer.echo("💬 查询结果:")
            typer.echo(result)

    except Exception as e:
        typer.echo(f"❌ 查询失败: {e}", err=True)
        typer.echo("💡 请检查网络连接和API配置", err=True)
        raise typer.Exit(1)


@app.command()
def batch_query(
    query_file: str = typer.Argument(..., help="包含查询文本的文件路径"),
    mode: str = typer.Option("local", "--mode", "-q", help="查询模式"),
    model: str = typer.Option("gpt-5", "--model", "-m", help="使用的模型名称"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件路径")
):
    """
    批量查询

    示例：
    python -m cli batch-query queries.txt -q local -o results.txt
    """
    # 验证查询模式
    valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
    if mode not in valid_modes:
        typer.echo(f"错误：无效的查询模式 '{mode}'，支持的模式: {', '.join(valid_modes)}", err=True)
        raise typer.Exit(1)

    # 验证查询文件
    query_path = Path(query_file)
    if not query_path.exists():
        typer.echo(f"错误：查询文件不存在: {query_file}", err=True)
        raise typer.Exit(1)

    # 读取查询文本
    try:
        with open(query_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        typer.echo(f"读取查询文件失败: {e}", err=True)
        raise typer.Exit(1)

    if not queries:
        typer.echo("错误：查询文件为空", err=True)
        raise typer.Exit(1)

    # 创建参数对象
    args = Args(model)

    # 执行批量查询
    try:
        typer.echo(f"📊 正在批量执行 {len(queries)} 个查询，使用 {mode} 模式...")
        typer.echo("-" * 60)

        results = []
        for i, query_text in enumerate(queries, 1):
            typer.echo(f"🔍 查询 {i}/{len(queries)}: {query_text}")
            result = asyncio.run(execute_query(query_text, mode, args))
            results.append({
                "query": query_text,
                "result": result,
                "mode": mode,
                "model": model
            })
            typer.echo(f"✅ 完成 {i}/{len(queries)}")
            typer.echo("-" * 30)

        # 输出结果
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"🔍 查询: {result['query']}\n")
                    f.write(f"🎯 模式: {result['mode']}\n")
                    f.write(f"🤖 模型: {result['model']}\n")
                    f.write(f"💬 结果: {result['result']}\n")
                    f.write("=" * 60 + "\n")
            typer.echo(f"✅ 结果已保存到: {output_file}")
        else:
            for result in results:
                typer.echo(f"🔍 查询: {result['query']}")
                typer.echo(f"🎯 模式: {result['mode']}")
                typer.echo(f"🤖 模型: {result['model']}")
                typer.echo(f"💬 结果: {result['result']}")
                typer.echo("=" * 60)

    except Exception as e:
        typer.echo(f"批量查询失败: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def modes():
    """显示所有支持的查询模式"""
    modes_info = {
        "local": "本地查询 - 基于上下文相关信息",
        "global": "全局查询 - 利用全局知识",
        "hybrid": "混合查询 - 结合本地和全局检索方法",
        "naive": "朴素查询 - 执行基本搜索而不使用高级技术",
        "mix": "混合查询 - 集成知识图谱和向量检索",
        "bypass": "绕过查询 - 直接使用LLM回答"
    }

    typer.echo("🎯 支持的查询模式:")
    typer.echo("=" * 50)
    for mode, description in modes_info.items():
        typer.echo(f"  {mode:<8} - {description}")
    typer.echo("=" * 50)
    typer.echo("💡 使用 -q 参数选择查询模式")


@app.command()
def models():
    """显示可用的模型信息"""
    typer.echo("🤖 可用的模型配置:")
    typer.echo("=" * 50)
    typer.echo(f"默认模型: grok-code-fast-1")
    typer.echo(f"💡 可通过 -m 参数指定其他模型")
    typer.echo("🔧 环境变量:")
    typer.echo(f"  LLM_BINDING_HOST: {os.getenv('LLM_BINDING_HOST', 'http://10.0.62.214:15000/k-llm')}")
    typer.echo(f"  LLM_BINDING_API_KEY: {os.getenv('LLM_BINDING_API_KEY', 'sk-4e76fcdf3f95467198edabdc0d6627f6')}")
    typer.echo("=" * 50)


@app.command()
def version():
    """显示版本信息"""
    typer.echo("🚀 LightRAG CLI v1.0.0")
    typer.echo("💡 基于 LightRAG 的6种查询模式实现")
    typer.echo("📝 支持流式输出和多种模型配置")


async def execute_query(query_text: str, mode: str, args: Args) -> str:
    """
    执行查询

    Args:
        query_text: 查询文本
        mode: 查询模式
        args: 参数对象

    Returns:
        查询结果
    """
    # 初始化RAG系统
    try:
        processor = await initialize_rag()
    except Exception as e:
        raise RuntimeError(f"RAG系统初始化失败: {e}")

    if not processor:
        raise ValueError("RAG系统初始化失败")

    if args.model:
        os.environ["LLM_MODEL"] = args.model
        print(f"设置模型: {os.environ['LLM_MODEL']}")

    # 创建查询参数，禁用流式输出
    param = QueryParam(mode=mode, stream=False)

    # 执行查询
    try:
        result = await processor.query(query_text, mode=mode)
        return result
    except Exception as e:
        raise RuntimeError(f"查询执行失败: {e}")


async def execute_streaming_query(query_text: str, mode: str, args: Args):
    """
    执行流式查询

    Args:
        query_text: 查询文本
        mode: 查询模式
        args: 参数对象
    """
    # 初始化RAG系统
    try:
        processor = await initialize_rag()
    except Exception as e:
        print(f"错误：RAG系统初始化失败: {e}")
        return

    if not processor:
        print("错误：RAG系统初始化失败")
        return

    if args.model:
        os.environ["LLM_MODEL"] = args.model
        print(f"设置模型: {os.environ['LLM_MODEL']}")

    # 创建查询参数，启用流式输出
    param = QueryParam(mode=mode, stream=True)

    # 执行流式查询
    try:
        # 直接调用LightRAG实例的aquery方法
        if hasattr(processor, 'lightrag_instance') and processor.lightrag_instance:
            response = await processor.lightrag_instance.aquery(query_text, param=param)

            # 检查返回类型，如果是AsyncIterator则进行流式处理
            if hasattr(response, '__aiter__'):
                async for chunk in response:
                    print(chunk, end='', flush=True)
                print()  # 换行
            else:
                # 如果返回的是字符串，直接输出
                print(response)
        else:
            # 如果无法直接访问lightrag_instance，使用非流式查询
            print("警告：无法访问LightRAG实例，使用非流式查询")
            result = await processor.query(query_text, mode=mode)
            print(result)
    except Exception as e:
        print(f"\n警告：流式查询失败，回退到非流式查询")
        print(f"错误详情: {e}")
        try:
            result = await processor.query(query_text, mode=mode)
            print(result)
        except Exception as fallback_error:
            print(f"错误：非流式查询也失败: {fallback_error}")
            print("请检查网络连接和API配置")


def main():
    """主函数"""
    app()


if __name__ == "__main__":
    main()