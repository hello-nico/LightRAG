import re
import hashlib
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
import fitz  # PyMuPDF


@dataclass
class ExtractedContent:
    """提取的论文内容"""
    paper_id: str
    full_text: str
    metadata: Dict[str, any]


class PDFExtractor:
    """PDF论文内容提取器 - 仅提取去除引用的全文"""
    
    def extract(self, pdf_path: str) -> ExtractedContent:
        """
        从PDF文件中提取内容
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            ExtractedContent对象，包含去除引用的全文
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 生成paper_id
        paper_id = hashlib.md5(str(pdf_path.absolute()).encode()).hexdigest()[:16]
        
        # 打开PDF文档
        doc = fitz.open(str(pdf_path))
        
        try:
            # 提取元数据
            metadata = {
                'page_count': doc.page_count,
                'file_name': pdf_path.name,
                'file_size': pdf_path.stat().st_size
            }
            
            # 提取全文
            full_text = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text.append(text)
            
            # 合并全文
            combined_text = "\n\n".join(full_text)
            
            # 去除引用部分
            clean_text = self._remove_references(combined_text)
            
            return ExtractedContent(
                paper_id=paper_id,
                full_text=clean_text,
                metadata=metadata
            )
            
        finally:
            doc.close()
    
    def _remove_references(self, text: str) -> str:
        """去除引用部分"""
        
        # 查找References部分的开始位置
        ref_markers = [
            'References', 'REFERENCES', 
            'Bibliography', 'BIBLIOGRAPHY',
            'Works Cited', 'WORKS CITED'
        ]
        
        # 找到最早出现的引用标记位置
        ref_start = len(text)
        for marker in ref_markers:
            # 查找独立成行的标记
            pattern = f'\\n\\s*{re.escape(marker)}\\s*\\n'
            match = re.search(pattern, text)
            if match:
                ref_start = min(ref_start, match.start())
        
        # 截断到引用部分之前
        if ref_start < len(text):
            text = text[:ref_start]
        
        # 去除文内引用标记
        text = re.sub(r'\[\d+(?:[-,]\d+)*\]', '', text)  # 去除 [1], [2,3], [1-5]
        text = re.sub(r'\(\d{4}\)', '', text)  # 去除 (2023)
        text = re.sub(r'\([A-Za-z]+(?:\s+et\s+al\.)?,?\s*\d{4}\)', '', text)  # 去除 (Author et al., 2023)
        
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()

    
# 使用示例
def main():
    """测试提取器"""
    extractor = PDFExtractor()
    
    # 替换为您的PDF文件路径
    pdf_file = "rag_pdfs/2104.01111_A-Comprehensive-Survey-of-Scene-Graphs-Generation-and-Application.pdf"
    
    result = extractor.extract(pdf_file)
    
    if result:
        print(f"文件ID: {result.paper_id}")
        print(f"正文长度: {len(result.full_text)} 字符")
        print(f"总页数: {result.metadata['page_count']}")
    else:
        print("PDF提取失败")


if __name__ == "__main__":
    main()