"""
论文知识图谱提取 CLI 工具
Paper Knowledge Graph Extraction CLI

使用 Typer 框架封装 kg_generator 功能，提供命令行界面。
"""
import asyncio
import typer
from pathlib import Path
from typing import List
from core.paper.kg_generator import (
    process_single_file, process_directory, merge_existing_kg_files)
from config import settings

# 创建 Typer 应用
app = typer.Typer(
    name="paper-cli",
    help="论文知识图谱提取工具 - 从综述以及综合得分较高的论文中提取实体和关系，为构建RAG领域核心知识本体提供数据支持",
    add_completion=False
)


class Args:
    """模拟命令行参数对象，使用配置文件中的默认值"""
    def __init__(self):
        self.api_base = settings.api_base
        self.api_key = settings.api_key
        self.model = settings.model
        self.language = settings.language


@app.command()
def process(
    input_file: str = typer.Argument(..., help="输入论文文件路径（支持 .txt, .md, .pdf）"),
    output_dir: str = typer.Argument(..., help="输出目录路径"),
    model: str = typer.Option(settings.model, help="模型名称")
):
    """
    处理单个论文文件，提取知识图谱
    
    示例：
    paper-cli process input.pdf results/ --model gpt-5-chat
    """
    # 验证输入文件
    input_path = Path(input_file)
    if not input_path.exists():
        typer.echo(f"错误：输入文件不存在: {input_file}", err=True)
        raise typer.Exit(1)
    
    # 验证支持的文件格式
    supported_extensions = ['.txt', '.md', '.pdf']
    if input_path.suffix.lower() not in supported_extensions:
        typer.echo(f"错误：不支持的文件格式 {input_path.suffix}，支持的格式: {', '.join(supported_extensions)}", err=True)
        raise typer.Exit(1)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建参数对象
    args = Args()
    
    if model:
        args.model = model
        print(f"使用模型: {args.model}")
    
    # 处理文件
    try:
        typer.echo(f"开始处理文件: {input_file}")
        asyncio.run(process_single_file(input_file, output_dir, args))
        typer.echo(f"处理完成！结果保存在: {output_dir}")
    except Exception as e:
        typer.echo(f"处理失败: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="输入目录路径"),
    output_dir: str = typer.Argument(..., help="输出目录路径"),
    model: str = typer.Option(settings.model, help="模型名称")
):
    """
    批量处理目录中的所有论文文件
    
    示例：
    paper-cli batch papers/ results/ --model gpt-5-chat
    """
    # 验证输入目录
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        typer.echo(f"错误：输入目录不存在: {input_dir}", err=True)
        raise typer.Exit(1)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建参数对象
    args = Args()
    
    if model:
        args.model = model
        print(f"使用模型: {args.model}")
    
    # 处理目录
    try:
        typer.echo(f"开始批量处理目录: {input_dir}")
        asyncio.run(process_directory(input_dir, output_dir, args))
        typer.echo(f"批量处理完成！结果保存在: {output_dir}")
    except Exception as e:
        typer.echo(f"处理失败: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def merge(
    input_files: List[str] = typer.Argument(..., help="要合并的知识图谱文件路径"),
    output_dir: str = typer.Argument(..., help="输出目录路径")
):
    """
    合并多个现有的知识图谱文件
    
    示例：
    paper-cli merge file1.json file2.json file3.json results/
    """
    # 验证输入文件
    valid_files = []
    for file_path in input_files:
        path = Path(file_path)
        if not path.exists():
            typer.echo(f"警告：文件不存在，跳过: {file_path}", err=True)
            continue
        if path.suffix.lower() != '.json':
            typer.echo(f"警告：文件不是 JSON 格式，跳过: {file_path}", err=True)
            continue
        valid_files.append(file_path)
    
    if not valid_files:
        typer.echo("错误：没有有效的 JSON 文件可以合并", err=True)
        raise typer.Exit(1)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 合并文件
    try:
        typer.echo(f"开始合并 {len(valid_files)} 个文件")
        asyncio.run(merge_existing_kg_files(valid_files, output_dir))
        typer.echo(f"合并完成！结果保存在: {output_dir}/merged.json")
    except Exception as e:
        typer.echo(f"合并失败: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def version():
    """显示版本信息"""
    typer.echo("paper-cli v1.0.0")


if __name__ == "__main__":
    app()