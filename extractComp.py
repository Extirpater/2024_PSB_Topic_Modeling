import os
import pdfplumber
import re
from textprocessor import TextProcessor

# Constants
BASE_PDF_FOLDER = '/Users/DevZhang/data-retrieval-project/PSB_Papers'
OUTPUT_FOLDER = '/Users/DevZhang/data-retrieval-project/PSB_Papers/text_outputs'
MAIN_BODY_FOLDER = '/Users/DevZhang/data-retrieval-project/PSB_Papers/main_body'
MIN_ABSTRACT_LENGTH = 100  # Minimum length for abstract text


def extract_text_pdfplumber(pdf_path):
    text = ''
    text_processor = TextProcessor()
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text_simple()
            page_text = page_text.lower()
            
            #page_text = page.extract_text()
            if page_text:
                lines = page_text.split('\n')
                filtered_lines = [text_processor.process_text(line, sc=False, lemma=False) for line in lines if not (line.strip().isdigit() or page.extract_text().startswith("1."))]
                #filtered_lines = [line for line in lines if not (line.strip().isdigit() or page.extract_text().startswith("1."))]
                text += '\n'.join(filtered_lines) + '\n'
    return text

def find_section(text, section_title):
    index = text.find(section_title)
    if index != -1:
        return index
    return None

def extract_abstract(text, intro_index):
    abstract_end_index = text.rfind('\n', 0, intro_index)
    if find_section(text, "abstract"):
        start = find_section(text, "abstract")
    else: start = 0
    abstract_text = text[start+8:abstract_end_index].strip()
    if len(abstract_text) >= MIN_ABSTRACT_LENGTH:
        return abstract_text
    return None

def extract_main_body(text, intro_index, ref_index):
    intro_start_index = text.rfind('\n', 0, intro_index) + 1
    if intro_start_index != -1:
        main_body_text = text[intro_start_index+12:ref_index].strip()
        return main_body_text
    return None

def extract_references(text, index):
    intro_start_index = text.rfind('\n', 0, index) + 1
    if intro_start_index != -1:
        references_text = text[intro_start_index:].strip()
        return references_text
    return -1

def save_text_to_file(text, output_path):
    with open(output_path, 'w') as f:
        f.write(text)
    print(f"Text has been saved to {output_path}")

def save_main_body_to_file(main_body_text, output_path):
    with open(output_path, 'w') as f:
        f.write(main_body_text)
    print(f"Main body has been saved to {output_path}")

def process_pdf(pdf_path, output_path, main_body_folder):
    """
    Process a single PDF file and save extracted sections to text files.
    """
    text = extract_text_pdfplumber(pdf_path)
    
    intro_index = find_section(text, "introduction")
    reference_index = find_section(text, "references")
    title = os.path.splitext(os.path.basename(output_path))[0]
    if intro_index is not None:
        abstract = extract_abstract(text, intro_index)
        main_body = extract_main_body(text, intro_index, reference_index)
        references = extract_references(text, reference_index)
        

        if abstract and main_body and references:
            save_text_to_file(f"Abstract:\n{abstract}\n\nMain Body:\n{main_body}\n\nReferences:\n{references}", output_path)
            main_body_output_path = os.path.join(main_body_folder, title + '_main_body.txt')
            save_main_body_to_file(main_body, main_body_output_path)
        elif main_body:
            main_body_output_path = os.path.join(main_body_folder, title + '_main_body.txt')
            save_main_body_to_file(main_body, main_body_output_path)
        else:
            print(f"Error extracting sections from {pdf_path}. Sections might be missing or too short.")
    else:
        # print(f"Saving whole document as body because Introduction was not found in {pdf_path}")
        # main_body_output_path = os.path.join(main_body_folder, title + '_main_body.txt')
        # save_main_body_to_file(text, main_body_output_path)
        print(f"Introduction not found in {pdf_path}. Skipping.")

def process_all_pdfs_in_folder(folder_path, output_folder, main_body_folder):
    """
    Process all PDF files in a folder and save extracted sections to text files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(main_body_folder):
        os.makedirs(main_body_folder)

    for filename in os.listdir(folder_path):
        pdf_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt')
        process_pdf(pdf_path, output_path, main_body_folder)
        # if filename.endswith('.pdf'):
            

# Loop through PDF folders by year
for year in range(1996, 2025):
    pdf_folder = os.path.join("./PSB_Papers/PDFS/", f'psb{year}_pdfs')

    # Process PDFs in the current year's folder
    process_all_pdfs_in_folder(pdf_folder, f"./PSB_Papers/text_outputs/{year}", f"./PSB_Papers/main_body/{year}")
