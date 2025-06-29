import os
import re
import unicodedata
from bs4 import BeautifulSoup
from typing import List, Dict
from uuid import uuid4

class HTMLPreprocess:
    """
    Принимает путь к директории, содержащей HTML-файлы.
    Преобразует HTML-файлы в текст
    Возвращает список словарей с данными о каждом файле тч:
    file_data: {
        'file_name': имя файла,
        'title': заголовок (например, название документа),
        'content': {
            'text': обычный текст,
            'table': список таблиц со структурой {'text': ..., 'metadata': {...}}
        }
    }

    Args:
        html_dir (str): Путь к директории с HTML-файлами.
    """
    def __init__(self, html_dir: str):
        self.html_dir = html_dir
        self.html_file_paths = self._gather_html_files()

    def _gather_html_files(self) -> List[str]:
        return [
            os.path.join(self.html_dir, f)
            for f in os.listdir(self.html_dir)
            if f.endswith('.html') or f.endswith('.htm')
        ]

    # Function to clean and normalize and clean text
    def _clean_text(self, text: str) -> str:
        return unicodedata.normalize('NFKC', ' '.join(text.split()))

    # Handle links inside tags (e.g. <a>)
    def _handle_links(self, element) -> None:
        for a in element.find_all('a'):
            if a.string and a.get('href'):
                a.replace_with(f"{a.string} ({a['href']})")

    def preprocess(self) -> List[Dict]:
        all_data = []

        for file_path in self.html_file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.find('title').get_text() if soup.find('title') else "Untitled"
            file_name = os.path.basename(file_path)

            # Remove unwanted tags
            for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
                tag.decompose()

            # Extract tabular content
            table_blocks = []
            for table in soup.find_all('table'):
                # Try to get a description before the table
                table_desc = ""
                prev_p = table.find_previous('p')
                while prev_p:
                    if prev_p.get_text(strip=True):
                        table_desc = prev_p.get_text(strip=True)
                        break
                    prev_p = prev_p.find_previous('p')

                # Clone and clean the table
                table_clone = table.__copy__()
                allowed_table_tags = {"table", "tr", "td"}
                for tag in table_clone.find_all(True):
                    if tag.name not in allowed_table_tags:
                        tag.unwrap() # Remove unwanted tags but keep content
                    else:
                        tag.attrs = {} # Clear all attributes
                    
                # Remove whitespace-only elements
                cleaned_table_clone = str(table_clone).replace("\n", "").replace("\r", "")
                rows = []
                for row in table.find_all('tr'):
                    cells = [self._clean_text(cell.get_text()) for cell in row.find_all(['td', 'th'])]
                    if cells:
                        rows.append(" | ".join(cells))

                table_blocks.append({
                    'text': "\n".join(rows),
                    'metadata': {
                        'text_as_html': cleaned_table_clone,
                        'table_description': f"{title}\n\n{table_desc}"
                    }
                })

                table.decompose()

            # Extract main content
            text_blocks = []
            main_content = soup.find('div', class_='content-page') or soup.body
            allowed_text_tags = {'p', 'li', 'div', 'span', 'h1', 'h2', 'h3', 'h4'}

            for element in main_content.descendants:
                if isinstance(element, str) or not element.name:
                    continue
                if element.name not in allowed_text_tags:
                    continue

                self._handle_links(element)
                # Special cases
                text = self._clean_text(element.get_text())

                if element.name.startswith('h') and len(element.name) == 2:
                    level = int(element.name[1])
                    text_blocks.append(f"{'#' * level} {text}")
                elif element.name == 'li':
                    text_blocks.append(f"- {text}")
                else:
                    text_blocks.append(text)

            # Collapse excessive newlines and whitespace
            text_output = "\n".join(text_blocks)
            text_output = re.sub(r'\n{3,}', '\n\n', text_output) # Remove excessive newlines
            text_output = re.sub(r'(?<!\n)\n(?!\n)', ' ', text_output).strip() # Join broken lines

            all_data.append({
                'file_name': file_name,
                'title': title,
                'content': {
                    'text': text_output,
                    'table': table_blocks
                }
            })

        return all_data
