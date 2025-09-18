"""
LightRAG CLI 工具
基于统一实例化对象的6种query模式实现
"""
import asyncio
import typer
from lightrag import QueryParam
from lightrag.core import get_lightrag_instance, load_core_config
from lightrag.tools.pdf_reader import PDFExtractor
from lightrag.kg.shared_storage import initialize_pipeline_status
from pathlib import Path
import os
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass

class ProcessingMode(str, Enum):
    """处理模式枚举"""
    DRY_RUN = "dry-run"
    INSERT = "insert"


@dataclass
class ProcessingResult:
    """处理结果"""
    file_path: str
    success: bool
    content_length: int
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


app = typer.Typer(
    name="lightrag-cli",
    help="LightRAG CLI 工具 - 支持6种查询模式与批量处理PDF文件",
    add_completion=False
)


async def get_cli_instance(working_dir: Optional[str] = None):
    """
    获取CLI专用的LightRAG实例

    Args:
        working_dir: 工作目录，如果为None则使用默认配置

    Returns:
        LightRAG实例
    """
    # 加载核心配置
    if working_dir:
        config = load_core_config(custom_defaults={"working_dir": working_dir})
    else:
        config = load_core_config()

    # 获取LightRAG实例
    return await get_lightrag_instance(config=config, auto_init=True)


class PDFProcessor:
    """PDF handler"""

    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = working_dir or os.getenv("WORKING_DIR", "./rag_storage")
        self.extractor = PDFExtractor()
        self.lightrag = None
        self.init_pipeline_status = False
        self.processing_results: List[ProcessingResult] = []

    def _ensure_working_dir(self):
        """确保工作目录存在"""
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

    async def _initialize_lightrag(self):
        """初始化LightRAG实例"""
        if self.lightrag is None:
            self.lightrag = await get_cli_instance(self.working_dir)
            if not self.init_pipeline_status:
                await initialize_pipeline_status()
                self.init_pipeline_status = True

    async def process_single_pdf(self, pdf_path: Path, mode: ProcessingMode) -> ProcessingResult:
        """
        处理单个PDF文件

        Args:
            pdf_path: PDF文件路径
            mode: 处理模式

        Returns:
            处理结果
        """
        try:
            # 提取PDF内容
            extracted = self.extractor.extract(str(pdf_path))

            result = ProcessingResult(
                file_path=str(pdf_path),
                success=True,
                content_length=len(extracted.full_text),
                metadata=extracted.metadata
            )

            # 根据模式执行不同的处理
            if mode == ProcessingMode.DRY_RUN:
                # 仅提取和统计，不需要网络/API
                pass
            else:
                await self._initialize_lightrag()
                await self.lightrag.ainsert(
                    extracted.full_text,
                    ids=pdf_path.stem,
                    file_paths=str(pdf_path)
                )
            
            return result

        except Exception as e:
            return ProcessingResult(
                file_path=str(pdf_path),
                success=False,
                content_length=0,
                error_message=str(e)
            )

    async def process_pdfs(self,
                          input_dir: str = "",
                          output_dir: Optional[str] = None,
                          max_concurrent: int = 4,
                          mode: ProcessingMode = ProcessingMode.INSERT) -> List[ProcessingResult]:
        """
        处理多个PDF文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录（工作目录）
            max_concurrent: 最大并发数
            mode: 处理模式

        Returns:
            Processing result list
        """
        if output_dir:
            self.working_dir = output_dir

        self._ensure_working_dir()

        # 查找PDF文件
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            typer.echo(f"在 {input_dir} 中未找到PDF文件", err=True)
            return []

        typer.echo(f"找到 {len(pdf_files)} 个PDF文件，开始处理...")

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(pdf_path: Path) -> ProcessingResult:
            async with semaphore:
                result = await self.process_single_pdf(pdf_path, mode)
                return result

        # 并发处理
        tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    file_path=str(pdf_files[i]),
                    success=False,
                    content_length=0,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)

        self.processing_results = processed_results
        return processed_results


class Args:
    """模拟命令行参数对象，使用配置文件中的默认值"""
    def __init__(self, model: str = "grok-code-fast-1"):
        self.api_base = os.getenv("LLM_BINDING_HOST")
        self.api_key = os.getenv("LLM_BINDING_API_KEY")
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
    api_key = os.getenv('LLM_BINDING_API_KEY')
    if api_key:
        typer.echo(f"  LLM_BINDING_API_KEY: {'*' * (len(api_key) - 4) + api_key[-4:]}")
    else:
        typer.echo("  LLM_BINDING_API_KEY: 未设置")
    typer.echo("=" * 50)


@app.command()
def extract(
    input_dir: str = typer.Option("./rag_pdfs", "--input-dir", help="PDF文件输入目录"),
    output_dir: str = typer.Option(os.getenv("WORKING_DIR", "./rag_storage"), "--output-dir", help="输出目录（工作目录）"),
    max_concurrent: int = typer.Option(4, "--max-concurrent", help="最大并发处理数"),
    mode: ProcessingMode = typer.Option(ProcessingMode.INSERT, "--mode", help="处理模式: dry-run, insert")
):
    """
    提取和处理PDF文件

    示例：
    python -m cli extract --input-dir ./pdfs --output-dir ./output --max-concurrent 4 --mode dry-run
    """
    try:
        # 创建PDF处理器
        processor = PDFProcessor(working_dir=output_dir)

        # 处理PDF文件
        results = asyncio.run(processor.process_pdfs(
            input_dir=input_dir,
            output_dir=output_dir,
            max_concurrent=max_concurrent,
            mode=mode
        ))

        # 统计结果
        success_count = sum(1 for r in results if r.success)
        failed_count = len(results) - success_count

        # 显示结果
        typer.echo(f"\n📊 处理完成！")
        typer.echo(f"✅ 成功: {success_count} 个文件")
        typer.echo(f"❌ 失败: {failed_count} 个文件")
        typer.echo(f"📁 工作目录: {processor.working_dir}")
        typer.echo(f"🔧 处理模式: {mode.value}")

        # 显示成功文件的统计
        if success_count > 0:
            total_chars = sum(r.content_length for r in results if r.success)
            typer.echo(f"📝 总字符数: {total_chars:,}")

        # 显示失败的文件
        if failed_count > 0:
            typer.echo(f"\n❌ 失败文件列表:")
            for result in results:
                if not result.success:
                    typer.echo(f"  - {Path(result.file_path).name}: {result.error_message}")

        # 如果有失败，返回非零退出码
        if failed_count > 0:
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ 处理失败: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def version():
    """显示版本信息"""
    typer.echo("🚀 LightRAG CLI v1.0.0")
    typer.echo("💡 基于 LightRAG 的6种查询模式实现")
    typer.echo("📝 支持流式输出和多种模型配置")
    typer.echo("🔧 新增PDF处理功能")


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
    # 获取LightRAG实例
    try:
        lightrag = await get_cli_instance()
    except Exception as e:
        raise RuntimeError(f"LightRAG实例获取失败: {e}")

    if args.model:
        os.environ["LLM_MODEL"] = args.model
        print(f"设置模型: {os.environ['LLM_MODEL']}")

    # 创建查询参数，禁用流式输出
    param = QueryParam(mode=mode, stream=False)

    # 执行查询
    try:
        result = await lightrag.aquery(query_text, param=param)
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
    # 获取LightRAG实例
    try:
        lightrag = await get_cli_instance()
    except Exception as e:
        print(f"错误：LightRAG实例获取失败: {e}")
        return

    if args.model:
        os.environ["LLM_MODEL"] = args.model
        print(f"设置模型: {os.environ['LLM_MODEL']}")

    # 创建查询参数，启用流式输出
    param = QueryParam(mode=mode, stream=True)

    # 执行流式查询
    try:
        response = await lightrag.aquery(query_text, param=param)

        # 检查返回类型，如果是AsyncIterator则进行流式处理
        if hasattr(response, '__aiter__'):
            async for chunk in response:
                print(chunk, end='', flush=True)
            print()  # 换行
        else:
            # 如果返回的是字符串，直接输出
            print(response)
    except Exception as e:
        print(f"\n警告：流式查询失败，回退到非流式查询")
        print(f"错误详情: {e}")
        try:
            # 回退到非流式查询
            param.stream = False
            result = await lightrag.aquery(query_text, param=param)
            print(result)
        except Exception as fallback_error:
            print(f"错误：非流式查询也失败: {fallback_error}")
            print("请检查网络连接和API配置")


def main():
    """主函数"""
    app()


if __name__ == "__main__":
    main()