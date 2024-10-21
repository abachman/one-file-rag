"""
Read in a .pdf file and dump it to stdout as raw text.

Usage: python -m one_file_rag.extract_pdf ~/Documents/SomeFile.pdf
"""

import sys
import os
import pymupdf

if __name__ == "__main__":
    # Extract text from PDF given as argv[0]
    file_name = sys.argv[1]

    # Check if file is a PDF and actually exists
    if file_name.endswith(".pdf") and os.path.exists(file_name):
        pdf = pymupdf.open(file_name)
        for page in pdf:
            text = page.get_text()
            print(text)
            print("\n\n--- PAGE ---\n")
