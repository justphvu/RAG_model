import re
from src.config import Constants
from typing import Dict, List, Tuple
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkBuilder:
    """
    Инициализация двух текстовых сплиттеров: для родительских и дочерних чанков.
    Родительские — крупные логические части, дочерние — более мелкие.
    """
    def __init__(
        self,
        parent_chunk_size: int = Constants.parent_chunk_size,
        parent_overlap: int = Constants.parent_overlap,
        child_chunk_size: int = Constants.child_chunk_size,
        child_overlap: int = Constants.child_overlap,
        separators: List[str] = Constants.MARKDOWN_SEPARATORS
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            length_function=len, # Character length with len()
            add_start_index=True, # If `True`, includes chunk's start index in metadata
            strip_whitespace=True, # If `True`, strips whitespace from the start and end of every document
            separators=separators
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            length_function=len, # Character length with len()
            add_start_index=True, # If `True`, includes chunk's start index in metadata
            strip_whitespace=True, # If `True`, strips whitespace from the start and end of every document
            separators=separators
        )

    def build_chunks_with_context(self, file_data: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Принимает данные одного файла и разбивает его текст и таблицы на чанки.
        
        :param file_data: {
            'file_name': имя файла,
            'title': заголовок (например, название документа),
            'content': {
                'text': обычный текст,
                'table': список таблиц со структурой {'text': ..., 'metadata': {...}}
            }
        }
        :return: два списка чанков — родительские и дочерние.
        """
        file_name = file_data['file_name']
        context_header = file_data['title']
        full_text = file_data['content']['text']
        tables = file_data['content']['table']

        # Split the full text into blocks - it is possible to use two hyphens (logical paragraphs)
        all_text_blocks = [block.strip() for block in full_text.split('\n\n') if block.strip()]
        parent_docs = []
        child_docs = []

        # -------- TEXT BLOCKS --------
        for block in all_text_blocks:
            parent_chunks = self.parent_splitter.split_text(block)
            for parent_chunk in parent_chunks:
                parent_id = str(uuid4())
                parent_docs.append({
                    "id": parent_id,
                    "text": f"{context_header}\n\n{parent_chunk}",
                    "metadata": {
                        "source": file_name,
                        "title": context_header,
                        "type": "text"
                    }
                })

                child_chunks = self.child_splitter.split_text(parent_chunk)
                for child_chunk in child_chunks:
                    child_docs.append({
                        "id": str(uuid4()),
                        "text": f"{context_header}\n\n{child_chunk}",
                        "metadata": {
                            "source": file_name,
                            "title": context_header,
                            "type": "text",
                            "parent_id": parent_id
                        }
                    })

        # -------- TABLE BLOCKS --------
        for table in tables:
            table_text = table['text']
            table_html = table['metadata'].get('text_as_html', '')
            table_desc = table['metadata'].get('table_description', '')
            table_parent_id = str(uuid4())
            # Tabular parents (by tables)
            parent_docs.append({
                "id": table_parent_id,
                "text": f"{table_desc}\n\n{table_text}",
                "metadata": {
                    "source": file_name,
                    "title": context_header,
                    "type": "table",
                    "table_description": table_desc,
                    "html": table_html
                }
            })
            # Tabular child (by lines)
            for row in table_text.split('\n'):
                if not row.strip():
                    continue
                child_docs.append({
                    "id": str(uuid4()),
                    "text": f"{table_desc}\n\n{row}",
                    "metadata": {
                        "source": file_name,
                        "title": context_header,
                        "type": "table",
                        "parent_id": table_parent_id,
                        "table_description": table_desc
                    }
                })

        return parent_docs, child_docs
