import os
import pdfplumber

# Constants
PDF_FOLDER = '/Users/DevZhang/data-retrieval-project/PSB_Papers/psb2016_pdfs'
OUTPUT_FOLDER = '/Users/DevZhang/data-retrieval-project/PSB_Papers/text_outputs'
MAIN_BODY_FOLDER = '/Users/DevZhang/data-retrieval-project/PSB_Papers/main_body'
MIN_ABSTRACT_LENGTH = 100  # Minimum length for abstract text

def extract_text_pdfplumber(pdf_path):
    """
    Extract text from a PDF using pdfplumber.
    """
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                lines = page_text.split('\n')
                filtered_lines = [line for line in lines if not (line.strip().isdigit() or page.extract_text().startswith("1."))]
                text += '\n'.join(filtered_lines) + '\n'
    return text

def find_section(text, section_title):
    """
    Find the index of a section title in the text.
    """
    index = text.find(section_title)
    if index != -1:
        return index
    return None

def extract_abstract(text, intro_index):
    """
    Extract the abstract section from the text.
    """
    abstract_end_index = text.rfind('\n', 0, intro_index)
    abstract_text = text[:abstract_end_index].strip()
    if len(abstract_text) >= MIN_ABSTRACT_LENGTH:
        return abstract_text
    return None

def extract_main_body(text, intro_index, ref_index):
    """
    Extract the main body section from the text.
    """
    intro_start_index = text.rfind('\n', 0, intro_index) + 1
    if intro_start_index != -1:
        main_body_text = text[intro_start_index:ref_index].strip()
        return main_body_text
    return None

def extract_references(text, intro_index):
    """
    Extract the references section from the text.
    """
    intro_start_index = text.rfind('\n', 0, intro_index) + 1
    if intro_start_index != -1:
        references_text = text[intro_start_index:].strip()
        return references_text
    return None

def save_text_to_file(text, output_path):
    """
    Save the text to a file.
    """
    with open(output_path, 'w') as f:
        f.write(text)
    print(f"Text has been saved to {output_path}")

def save_main_body_to_file(main_body_text, output_path):
    """
    Save the main body text to a file.
    """
    with open(output_path, 'w') as f:
        f.write(main_body_text)
    print(f"Main body has been saved to {output_path}")

def process_pdf(pdf_path, output_path, main_body_folder):
    """
    Process a single PDF file and save extracted sections to text files.
    """
    text = extract_text_pdfplumber(pdf_path)
    intro_index = find_section(text, "Introduction")
    reference_index = find_section(text, "References")

    if intro_index is not None:
        abstract = extract_abstract(text, intro_index)
        main_body = extract_main_body(text, intro_index, reference_index)
        references = extract_references(text, reference_index)

        if abstract and main_body and references:
            save_text_to_file(f"Abstract:\n{abstract}\n\nMain Body:\n{main_body}\n\nReferences:\n{references}", output_path)
            main_body_output_path = os.path.join(main_body_folder, os.path.splitext(os.path.basename(output_path))[0] + '_main_body.txt')
            save_main_body_to_file(main_body, main_body_output_path)
        else:
            print(f"Error extracting sections from {pdf_path}. Sections might be missing or too short.")
    else:
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
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt')
            process_pdf(pdf_path, output_path, main_body_folder)

# Example usage
process_all_pdfs_in_folder(PDF_FOLDER, OUTPUT_FOLDER, MAIN_BODY_FOLDER)
