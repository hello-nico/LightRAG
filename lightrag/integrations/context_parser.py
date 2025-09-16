"""
LightRAG 上下文解析器

此模块负责解析 LightRAG.aquery(..., only_need_context=True) 返回的 Markdown 格式上下文，
将其转换为结构化的实体、关系和文本块数据。
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .models import Entity, Relationship, Chunk

logger = logging.getLogger(__name__)


@dataclass
class ParsedContext:
    """解析后的上下文数据"""
    chunks: List[Chunk] = None
    entities: List[Entity] = None
    relationships: List[Relationship] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
        if self.entities is None:
            self.entities = []
        if self.relationships is None:
            self.relationships = []
        if self.metadata is None:
            self.metadata = {}


class ContextParser:
    """上下文解析器"""

    def __init__(self):
        self.entity_pattern = re.compile(r'^(.+?)\s*\[(.+?)\](?:\s*:\s*(.+))?$')
        self.relationship_pattern = re.compile(r'^(.+?)\s*-\[.+?\]->\s*(.+?)(?:\s*:\s*(.+))?$')
        self.chunk_pattern = re.compile(r'^#{1,6}\s*(.+)$')

    def parse(self, markdown_context: str) -> ParsedContext:
        """
        解析 Markdown 格式的上下文

        Args:
            markdown_context: LightRAG 返回的 Markdown 格式上下文

        Returns:
            ParsedContext: 解析后的结构化数据
        """
        if not markdown_context:
            logger.warning("收到空的上下文内容")
            return ParsedContext()

        try:
            return self._parse_markdown(markdown_context)
        except Exception as e:
            logger.error(f"解析上下文时发生错误: {e}")
            return ParsedContext(metadata={"error": str(e)})

    def _parse_markdown(self, markdown: str) -> ParsedContext:
        """解析 Markdown 内容"""
        lines = markdown.strip().split('\n')
        parsed = ParsedContext()

        current_chunk = None
        chunk_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否是标题（新的 chunk 开始）
            if self._is_chunk_header(line):
                # 保存前一个 chunk
                if current_chunk and chunk_content:
                    chunk_text = '\n'.join(chunk_content).strip()
                    current_chunk.content = chunk_text
                    parsed.chunks.append(current_chunk)
                    chunk_content = []

                # 创建新的 chunk
                chunk_title = self._extract_chunk_title(line)
                current_chunk = Chunk(
                    id=self._generate_chunk_id(chunk_title),
                    doc_id="",  # 将由转换器填充
                    content="",
                    chunk_index=len(parsed.chunks),
                    metadata={"title": chunk_title}
                )

            elif current_chunk is not None:
                # 解析实体
                if self._is_entity_line(line):
                    entity = self._parse_entity(line)
                    if entity:
                        parsed.entities.append(entity)
                        if current_chunk:
                            current_chunk.metadata.setdefault("entities", []).append(entity.id)

                # 解析关系
                elif self._is_relationship_line(line):
                    relationship = self._parse_relationship(line)
                    if relationship:
                        parsed.relationships.append(relationship)
                        if current_chunk:
                            current_chunk.metadata.setdefault("relationships", []).append(relationship.id)

                # 普通文本内容
                else:
                    chunk_content.append(line)

        # 保存最后一个 chunk
        if current_chunk and chunk_content:
            chunk_text = '\n'.join(chunk_content).strip()
            current_chunk.content = chunk_text
            parsed.chunks.append(current_chunk)

        # 如果没有找到 chunk 标题，将整个内容作为一个 chunk
        if not parsed.chunks and markdown.strip():
            parsed.chunks.append(Chunk(
                id="default_chunk",
                doc_id="",
                content=markdown.strip(),
                chunk_index=0,
                metadata={"source": "full_content"}
            ))

        parsed.metadata.update({
            "total_chunks": len(parsed.chunks),
            "total_entities": len(parsed.entities),
            "total_relationships": len(parsed.relationships)
        })

        return parsed

    def _is_chunk_header(self, line: str) -> bool:
        """检查是否是 chunk 标题行"""
        return bool(self.chunk_pattern.match(line))

    def _extract_chunk_title(self, line: str) -> str:
        """提取 chunk 标题"""
        match = self.chunk_pattern.match(line)
        return match.group(1) if match else line

    def _generate_chunk_id(self, title: str) -> str:
        """生成 chunk ID"""
        # 简单的 ID 生成策略，可以根据需要改进
        return f"chunk_{hash(title) & 0xffffffff}"

    def _is_entity_line(self, line: str) -> bool:
        """检查是否是实体行"""
        return bool(self.entity_pattern.match(line))

    def _parse_entity(self, line: str) -> Optional[Entity]:
        """解析实体"""
        match = self.entity_pattern.match(line)
        if not match:
            return None

        name = match.group(1).strip()
        entity_type = match.group(2).strip()
        description = match.group(3).strip() if match.group(3) else None

        return Entity(
            id=f"entity_{hash(name) & 0xffffffff}",
            name=name,
            type=entity_type,
            description=description,
            metadata={"source_line": line}
        )

    def _is_relationship_line(self, line: str) -> bool:
        """检查是否是关系行"""
        return bool(self.relationship_pattern.match(line))

    def _parse_relationship(self, line: str) -> Optional[Relationship]:
        """解析关系"""
        match = self.relationship_pattern.match(line)
        if not match:
            return None

        source = match.group(1).strip()
        target = match.group(2).strip()
        description = match.group(3).strip() if match.group(3) else None

        # 简单的关系类型推断
        relation_type = "related"
        if "属于" in line or "is_a" in line.lower():
            relation_type = "is_a"
        elif "包含" in line or "contains" in line.lower():
            relation_type = "contains"
        elif "位于" in line or "located_in" in line.lower():
            relation_type = "located_in"

        return Relationship(
            id=f"rel_{hash(source + target) & 0xffffffff}",
            source_entity_id=f"entity_{hash(source) & 0xffffffff}",
            target_entity_id=f"entity_{hash(target) & 0xffffffff}",
            relation_type=relation_type,
            description=description,
            metadata={"source_line": line}
        )

    def validate_parsed_data(self, parsed: ParsedContext) -> List[str]:
        """
        验证解析后的数据

        Args:
            parsed: 解析后的上下文数据

        Returns:
            List[str]: 验证错误列表
        """
        errors = []

        # 验证实体 ID 引用
        entity_ids = {entity.id for entity in parsed.entities}

        for relationship in parsed.relationships:
            if relationship.source_entity_id not in entity_ids:
                errors.append(f"关系引用了不存在的源实体: {relationship.source_entity_id}")
            if relationship.target_entity_id not in entity_ids:
                errors.append(f"关系引用了不存在的目标实体: {relationship.target_entity_id}")

        # 验证 chunk 中的实体引用
        for chunk in parsed.chunks:
            chunk_entities = chunk.metadata.get("entities", [])
            for entity_id in chunk_entities:
                if entity_id not in entity_ids:
                    errors.append(f"Chunk {chunk.id} 引用了不存在的实体: {entity_id}")

        return errors