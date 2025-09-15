#!/usr/bin/env python3
"""
分析domain_kg_enhanced.json数据结构并统计信息
"""

import json
from collections import Counter
from typing import Dict, List, Any

def analyze_kg_structure(file_path: str) -> Dict[str, Any]:
    """分析KG数据结构并返回统计信息"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 基本统计
    domain_name = data.get('domain_name', 'Unknown')
    entities = data.get('entities', [])
    relationships = data.get('relationships', [])
    
    # 实体统计
    entity_types = Counter()
    entity_names = []
    
    for entity in entities:
        entity_type = entity.get('type', 'Unknown')
        entity_name = entity.get('name', 'Unknown')
        entity_types[entity_type] += 1
        entity_names.append(entity_name)
    
    # 关系统计
    relationship_descriptions = []
    relationship_weights = []
    
    for rel in relationships:
        description = rel.get('description', 'Unknown')
        weight = rel.get('weight', 0.0)
        relationship_descriptions.append(description)
        relationship_weights.append(weight)
    
    # 分析关系类型（基于描述和实体类型组合）
    relationship_types = Counter()
    for rel in relationships:
        src_id = rel.get('src_id', '')
        tgt_id = rel.get('tgt_id', '')
        description = rel.get('description', '')
        
        # 根据描述推断关系类型
        desc_lower = description.lower()
        if 'author' in desc_lower or '作者' in description:
            rel_type = 'authorship'
        elif 'affiliated' in desc_lower or 'institution' in desc_lower or '机构' in description:
            rel_type = 'affiliation'
        elif 'component' in desc_lower or '组成部分' in description:
            rel_type = 'component_of'
        elif 'used' in desc_lower or '使用' in description:
            rel_type = 'uses'
        elif 'evaluated' in desc_lower or 'evaluation' in desc_lower or '评估' in description:
            rel_type = 'evaluation'
        elif 'application' in desc_lower or '应用' in description:
            rel_type = 'application'
        elif 'based' in desc_lower or '基础' in description:
            rel_type = 'based_on'
        else:
            rel_type = 'related_to'
        
        relationship_types[rel_type] += 1
    
    return {
        'domain_name': domain_name,
        'total_entities': len(entities),
        'total_relationships': len(relationships),
        'entity_types': dict(entity_types),
        'relationship_types': dict(relationship_types),
        'sample_entities': entity_names[:10],  # 显示前10个实体
        'sample_relationships': relationship_descriptions[:5],  # 显示前5个关系
        'weight_stats': {
            'min': min(relationship_weights) if relationship_weights else 0,
            'max': max(relationship_weights) if relationship_weights else 0,
            'avg': sum(relationship_weights) / len(relationship_weights) if relationship_weights else 0
        }
    }

def print_analysis_report(stats: Dict[str, Any]):
    """打印分析报告"""
    print("=" * 60)
    print(f"📊 KG数据分析报告: {stats['domain_name']}")
    print("=" * 60)
    
    print(f"\n📈 基本统计:")
    print(f"   • 实体总数: {stats['total_entities']}")
    print(f"   • 关系总数: {stats['total_relationships']}")
    print(f"   • 实体类型数: {len(stats['entity_types'])}")
    print(f"   • 关系类型数: {len(stats['relationship_types'])}")
    
    print(f"\n🏷️  实体类型分布:")
    for entity_type, count in sorted(stats['entity_types'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_entities']) * 100
        print(f"   • {entity_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n🔗 关系类型分布:")
    for rel_type, count in sorted(stats['relationship_types'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_relationships']) * 100
        print(f"   • {rel_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n⚖️  关系权重统计:")
    weight_stats = stats['weight_stats']
    print(f"   • 最小值: {weight_stats['min']:.2f}")
    print(f"   • 最大值: {weight_stats['max']:.2f}")
    print(f"   • 平均值: {weight_stats['avg']:.2f}")
    
    print(f"\n📝 实体样本 (前10个):")
    for i, entity in enumerate(stats['sample_entities'], 1):
        print(f"   {i:2d}. {entity}")
    
    print(f"\n🔗 关系样本 (前5个):")
    for i, rel in enumerate(stats['sample_relationships'], 1):
        print(f"   {i}. {rel}")

if __name__ == "__main__":
    file_path = "/home/chencheng/py/src/LightRAG/datasets/domain_kg_enhanced.json"
    stats = analyze_kg_structure(file_path)
    print_analysis_report(stats)