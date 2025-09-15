#!/usr/bin/env python3
"""
将domain_kg_enhanced.json转换为LightRAG格式的转换脚本
"""

import json
import uuid
from typing import Dict, List, Any
from collections import defaultdict

def convert_domain_kg_to_lightrag_format(input_file: str, output_file: str):
    """
    将domain_kg_enhanced.json转换为LightRAG所需的格式
    
    Args:
        input_file: 输入的KG文件路径
        output_file: 输出的LightRAG格式文件路径
    """
    
    # 加载原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    domain_name = original_data.get('domain_name', 'RAG')
    entities = original_data.get('entities', [])
    relationships = original_data.get('relationships', [])
    
    print(f"🔄 开始转换 {len(entities)} 个实体和 {len(relationships)} 个关系...")
    
    # 创建LightRAG格式的数据结构
    lightrag_data = {
        "chunks": [],
        "entities": [],
        "relationships": [],
        "metadata": {
            "dataset_name": f"{domain_name} Knowledge Graph",
            "created_date": "2025-01-20",
            "description": f"Converted from domain_kg_enhanced.json with {len(entities)} entities and {len(relationships)} relationships",
            "source_format": "domain_kg_enhanced",
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "domain": domain_name
        }
    }
    
    # 按论文分组的实体和关系
    paper_groups = defaultdict(lambda: {
        'entities': [],
        'relationships': [],
        'papers': []
    })
    
    # 首先按论文分组
    for entity in entities:
        if entity.get('type') == 'Paper':
            paper_name = entity.get('name')
            paper_groups[paper_name]['papers'].append(entity)
    
    # 将实体分配到对应的论文组
    for entity in entities:
        if entity.get('type') != 'Paper':
            # 根据描述中的论文名称分配到对应组
            description = entity.get('description', '')
            assigned = False
            
            for paper_name in paper_groups.keys():
                if paper_name in description:
                    paper_groups[paper_name]['entities'].append(entity)
                    assigned = True
                    break
            
            # 如果没有找到对应论文，创建一个通用组
            if not assigned:
                paper_groups['general']['entities'].append(entity)
    
    # 将关系分配到对应的论文组
    for relationship in relationships:
        src_id = relationship.get('src_id', '')
        tgt_id = relationship.get('tgt_id', '')
        description = relationship.get('description', '')
        assigned = False
        
        for paper_name in paper_groups.keys():
            if paper_name in description or paper_name == src_id or paper_name == tgt_id:
                paper_groups[paper_name]['relationships'].append(relationship)
                assigned = True
                break
        
        if not assigned:
            paper_groups['general']['relationships'].append(relationship)
    
    # 为每个组创建chunks和对应的实体、关系
    chunk_index = 0
    for group_name, group_data in paper_groups.items():
        if not group_data['entities'] and not group_data['relationships']:
            continue
            
        # 创建chunk内容
        if group_name == 'general':
            chunk_content = f"General {domain_name} domain knowledge containing various entities and relationships."
        else:
            # 找到对应的论文实体
            paper_entity = next((p for p in group_data['papers'] if p.get('name') == group_name), None)
            if paper_entity:
                chunk_content = f"Paper: {group_name}\n\n"
                chunk_content += f"Description: {paper_entity.get('description', '')}\n\n"
                
                # 添加相关作者
                authors = [e for e in group_data['entities'] if e.get('type') == 'Author']
                if authors:
                    chunk_content += "Authors:\n"
                    for author in authors[:5]:  # 限制显示数量
                        chunk_content += f"- {author.get('name', '')}\n"
                    chunk_content += "\n"
                
                # 添加相关机构
                institutions = [e for e in group_data['entities'] if e.get('type') == 'Institution']
                if institutions:
                    chunk_content += "Institutions:\n"
                    for inst in institutions[:3]:
                        chunk_content += f"- {inst.get('name', '')}\n"
                    chunk_content += "\n"
                
                # 添加主要方法
                methods = [e for e in group_data['entities'] if e.get('type') == 'Method']
                if methods:
                    chunk_content += "Key Methods:\n"
                    for method in methods[:3]:
                        chunk_content += f"- {method.get('name', '')}\n"
            else:
                chunk_content = f"Knowledge about {group_name} and related entities."
        
        # 创建chunk
        chunk_id = f"chunk_{chunk_index:04d}"
        chunk = {
            "content": chunk_content,
            "source_id": group_name.replace(' ', '_').lower(),
            "file_path": f"domain_kg/{domain_name.lower()}/",
            "chunk_order_index": chunk_index
        }
        lightrag_data["chunks"].append(chunk)
        
        # 添加实体到LightRAG格式
        for entity in group_data['entities']:
            lightrag_entity = {
                "entity_name": entity.get('name'),
                "entity_type": entity.get('type'),
                "description": entity.get('description', ''),
                "source_id": chunk_id,
                "file_path": f"domain_kg/{domain_name.lower()}/"
            }
            lightrag_data["entities"].append(lightrag_entity)
        
        # 添加关系到LightRAG格式
        for relationship in group_data['relationships']:
            # 从描述中提取关键词
            description = relationship.get('description', '')
            keywords = []
            
            # 基于关系类型添加关键词
            desc_lower = description.lower()
            if 'author' in desc_lower or '作者' in description:
                keywords.extend(['authorship', 'paper', 'research'])
            elif 'institution' in desc_lower or '机构' in description:
                keywords.extend(['affiliation', 'organization', 'institution'])
            elif 'component' in desc_lower or '组成部分' in description:
                keywords.extend(['component', 'part', 'architecture'])
            elif 'used' in desc_lower or '使用' in description:
                keywords.extend(['usage', 'implementation', 'tool'])
            elif 'evaluation' in desc_lower or '评估' in description:
                keywords.extend(['evaluation', 'metric', 'assessment'])
            elif 'application' in desc_lower or '应用' in description:
                keywords.extend(['application', 'use-case', 'scenario'])
            elif 'based' in desc_lower or '基础' in description:
                keywords.extend(['foundation', 'basis', 'underlying'])
            else:
                keywords.extend(['related', 'connection', 'association'])
            
            lightrag_relationship = {
                "src_id": relationship.get('src_id'),
                "tgt_id": relationship.get('tgt_id'),
                "description": description,
                "keywords": ", ".join(keywords),
                "weight": 0.8,  # 默认权重
                "source_id": chunk_id,
                "file_path": f"domain_kg/{domain_name.lower()}/"
            }
            lightrag_data["relationships"].append(lightrag_relationship)
        
        chunk_index += 1
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(lightrag_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 转换完成!")
    print(f"   • 原始实体: {len(entities)}")
    print(f"   • 原始关系: {len(relationships)}")
    print(f"   • 生成chunks: {len(lightrag_data['chunks'])}")
    print(f"   • 转换后实体: {len(lightrag_data['entities'])}")
    print(f"   • 转换后关系: {len(lightrag_data['relationships'])}")
    print(f"   • 输出文件: {output_file}")
    
    return lightrag_data

def validate_converted_data(data: Dict[str, Any]):
    """验证转换后的数据格式"""
    required_fields = {
        'chunks': ['content', 'source_id', 'file_path', 'chunk_order_index'],
        'entities': ['entity_name', 'entity_type', 'description', 'source_id', 'file_path'],
        'relationships': ['src_id', 'tgt_id', 'description', 'keywords', 'weight', 'source_id', 'file_path']
    }
    
    for section, fields in required_fields.items():
        if section not in data:
            raise ValueError(f"Missing section: {section}")
        
        for item in data[section]:
            for field in fields:
                if field not in item:
                    raise ValueError(f"Missing field '{field}' in {section}")
    
    print("✅ 数据格式验证通过!")

if __name__ == "__main__":
    input_file = "/home/chencheng/py/src/LightRAG/datasets/domain_kg_enhanced.json"
    output_file = "/home/chencheng/py/src/LightRAG/datasets/domain_kg_lightrag_format.json"
    
    try:
        converted_data = convert_domain_kg_to_lightrag_format(input_file, output_file)
        validate_converted_data(converted_data)
        print("\n🎉 数据转换成功完成! 现在可以在LightRAG中使用了。")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()