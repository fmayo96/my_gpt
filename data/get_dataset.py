import sys
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

def epub_to_txt(epub_path, output_path):
    try:
        book = epub.read_epub(epub_path)
        text_content = []

        print(f"Reading EPUB: {epub_path}")

        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()
                text_content.append(text.strip())

        full_text = '\n\n'.join(text_content)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        print(f"Conversion complete: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python epub_to_txt.py input.epub output.txt")
    else:
        epub_path = sys.argv[1]
        output_path = sys.argv[2]
        epub_to_txt(epub_path, output_path)