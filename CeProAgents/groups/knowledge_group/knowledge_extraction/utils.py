import csv
import io
import json
import os
import yaml
import base64
import re
import time
from pathlib import Path

from easydict import EasyDict

# --- Core minerU backend components for "from scratch" implementation ---
os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.utils.enum_class import MakeMode

from PIL import Image
from chonky import ParagraphSplitter


# === Robust JSON parsing helpers (Kept from original file) ===
# ... (All your robust_parse_json helper functions remain unchanged here) ...
CONTROL_CHARS = re.compile(r'[\x00-\x1F\x7F]')
FENCES = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.MULTILINE)
_INVALID_UNICODE = re.compile(r'\\[uU]([0-9a-fA-F]{0,3})(?![0-9a-fA-F])')

def _fix_invalid_unicode_escapes(s: str) -> str:
    return _INVALID_UNICODE.sub(r'\\u\1', s)

def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = _fix_invalid_unicode_escapes(s)
        if s2 != s:
            return json.loads(s2)
        raise

def _strip_code_fences(s: str) -> str:
    return FENCES.sub('', s)

def _strip_control_chars(s: str) -> str:
    return CONTROL_CHARS.sub('', s)

def _extract_top_level_json(s: str) -> str:
    first_brace = s.find('{'); first_bracket = s.find('[')
    i = min([x for x in [first_brace, first_bracket] if x != -1], default=-1)
    if i == -1: return s
    open_sym = s[i]
    close_sym = '}' if open_sym == '{' else ']'
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == open_sym: depth += 1
        elif c == close_sym:
            depth -= 1
            if depth == 0:
                return s[i:j+1]
    return s

def robust_parse_json(raw: str):
    s = _strip_code_fences(raw or "")
    s = _strip_control_chars(s)
    s = _extract_top_level_json(s)
    try: return _safe_json_loads(s)
    except Exception: pass
    items = []
    for line in s.splitlines():
        t = line.strip()
        if not t: continue
        if t.startswith('{') and t.endswith('}'):
            try: items.append(_safe_json_loads(t)); continue
            except Exception: pass
        if t.startswith('"') and t.endswith('"'):
            try: items.append(_safe_json_loads(t)); continue
            except Exception: pass
    if items: return items
    m = re.findall(r'"((?:[^"\\]|\\.)*)"', s)
    if m:
        out = []
        for x in m:
            try: sx = _fix_invalid_unicode_escapes(x); out.append(json.loads(f'"{sx}"'))
            except Exception: pass
        if out: return out
    return None

# ==============================================================================
# The SequentialImageWriter class is no longer needed.
# ==============================================================================


def get_config_easydict(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def covert_json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path, 'r') as f:
        json_data_list = json.load(f)
    csv_data = []
    csv_title = json_data_list[0].keys()
    for json_data in json_data_list:
        csv_data.append(json_data.values())
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_title)
        writer.writerow(csv_title)
        writer.writerows(csv_data)


# ==============================================================================
# "FROM SCRATCH" REIMPLEMENTATION USING minerU's BACKEND COMPONENTS
# ==============================================================================
def parse_pdf_mineru(pdf_input_path: str, pdf_output_dir: str):
    """
    Replicates minerU's parsing pipeline from scratch using its backend components,
    then post-processes the output to match the desired folder structure and file naming.
    """
    if not os.path.exists(pdf_input_path):
        print(f"Error: PDF file not found - {pdf_input_path}")
        return

    pdf_name_no_ext = Path(pdf_input_path).stem
    
    # --- 1. Setup Directories ---
    # The final structure will be: <output_base_dir>/<pdf_name>/...
    image_output_dir = os.path.join(pdf_output_dir, "images")
    text_output_dir = os.path.join(pdf_output_dir, "text")

    if os.path.exists(pdf_output_dir) and os.listdir(pdf_output_dir):
        print(f"Warning: Output directory {pdf_output_dir} already exists and is not empty.")
        return

    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)
    
    try:
        # --- 2. Read PDF file ---
        print(f"--- [Step 1/5] Reading PDF file: {pdf_input_path} ---")
        with open(pdf_input_path, 'rb') as f:
            pdf_bytes = f.read()

        # --- 3. Core Document Analysis (like `pipeline_doc_analyze`) ---
        print("--- [Step 2/5] Starting document analysis with minerU backend... ---")
        # This is the main analysis call. It takes a list of bytes and returns results.
        # We process one file, so we pass a list with one item.
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
            pdf_bytes_list=[pdf_bytes],
            lang_list=['ch', 'en'] # Assuming Chinese, can be changed
        )
        
        # Unpack results for the single file we processed
        model_output = infer_results[0]
        images_list = all_image_lists[0]
        pdf_doc = all_pdf_docs[0]
        lang = lang_list[0]
        ocr_enabled = ocr_enabled_list[0]
        print("  - Document analysis complete.")

        # --- 4. Convert Model Output to Structured JSON (like `pipeline_result_to_middle_json`) ---
        print("--- [Step 3/5] Converting raw model output to structured middle JSON... ---")
        # This step organizes raw data and writes images to the specified directory.
        image_writer = FileBasedDataWriter(image_output_dir)
        middle_json = pipeline_result_to_middle_json(
            model_output, images_list, pdf_doc, image_writer, lang, ocr_enabled
        )
        print("  - Conversion complete. Images written to disk.")

        # --- 5. Generate Markdown from Structured JSON (like `pipeline_union_make`) ---
        print("--- [Step 4/5] Generating Markdown content... ---")
        pdf_info = middle_json["pdf_info"]
        image_dir_relative = os.path.basename(image_output_dir) # Should be "images"
        
        md_content_str = pipeline_union_make(
            pdf_info, 
            MakeMode.MM_MD, # Specify that we want Markdown output
            image_dir_relative
        )
        
        # Save the generated Markdown
        md_final_path = os.path.join(text_output_dir, f"{pdf_name_no_ext}.md")
        with open(md_final_path, 'w', encoding='utf-8') as f:
            f.write(md_content_str)
        print(f"  - Markdown file saved to: {md_final_path}")

        # --- 6. Post-process: Rename images and update links ---
        print("--- [Step 5/5] Post-processing images and links... ---")
        original_image_files = sorted(os.listdir(image_output_dir))
        filename_mapping = {}
        image_counter = 1
        
        for old_filename in original_image_files:
            file_extension = os.path.splitext(old_filename)[1]
            new_filename = f"{image_counter}{file_extension}"
            
            os.rename(
                os.path.join(image_output_dir, old_filename),
                os.path.join(image_output_dir, new_filename)
            )
            filename_mapping[old_filename] = new_filename
            image_counter += 1
            
        print(f"  - Renamed {len(filename_mapping)} images sequentially.")

        # Update links in the Markdown file
        if filename_mapping:
            with open(md_final_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            for old_name, new_name in filename_mapping.items():
                md_content = md_content.replace(f"({image_dir_relative}/{old_name})", f"({image_dir_relative}/{new_name})")
            with open(md_final_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            print(f"  - Updated image links in Markdown.")

    except Exception as e:
        print(f"\n[ERROR] An exception occurred during the from-scratch pipeline: {e}")
        import traceback
        traceback.print_exc() # Print full error for debugging
        return

    print(f"\n--- Successfully processed {pdf_name_no_ext} ---")


# ... (All other utility functions like safe_format, parse_json_output, etc. remain here) ...
def safe_format(template: str, **kwargs) -> str:
    sentinels = {k: f"__PLACEHOLDER__{k.upper()}__" for k in kwargs}
    tmp = template
    for k, tok in sentinels.items(): tmp = tmp.replace("{" + k + "}", tok)
    tmp = tmp.replace("{", "{{").replace("}", "}}")
    for k, tok in sentinels.items(): tmp = tmp.replace(tok, "{" + k + "}")
    return tmp.format(**kwargs)

def parse_json_output(raw_output: str, context: str = "parsing"):
    if not raw_output: print(f"Error: Received empty output during {context}."); return None
    data = robust_parse_json(raw_output)
    if data is None: print(f"Error decoding JSON during {context}: robust_parse_json failed."); print(f"Raw output was:\n---\n{raw_output}\n---"); return None
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list): return data["items"]
    return data

def encode_image(image_path: str):
    byte_io = io.BytesIO()
    with Image.open(image_path) as img:
        img.save(byte_io, format='png')
        bytes_image = byte_io.getvalue()
        image = base64.b64encode(bytes_image).decode('utf-8')
    return image

def chunk_text(text: str, max_len: int=5000, model: str='mirth/chonky_distilbert_uncased_1', device: str='cpu'):
    splitter = ParagraphSplitter(device=device, model_id=model)
    try: chunks = list(splitter(text))
    except Exception as e: print(f"[chunk_text] ParagraphSplitter failed, fallback to whole text: {e}"); return [text] if text else []
    chunks_new = []; chunk_new = ''
    for chunk in chunks:
        if len(chunk_new) + len(chunk) > max_len:
            if chunk_new: chunks_new.append(chunk_new)
            chunk_new = chunk
        else: chunk_new += chunk
    if chunk_new: chunks_new.append(chunk_new)
    if not chunks_new and text: chunks_new = [text]
    return chunks_new


if __name__ == '__main__':
    print("Start PDF parsing test...")

    pdf_to_test = r"CeProBench\knowledge\knowledge_extract\1_C5_Dehydrogenation_CN.pdf" 
    base_output_dir = r"./test_output"

    if not os.path.exists(pdf_to_test):
        print(f"Error: The specified PDF file '{pdf_to_test}' does not exist.")
    else:
        try:
            # Call the new "from scratch" function
            parse_pdf_mineru(pdf_to_test, base_output_dir)
            
            pdf_filename_without_ext = os.path.splitext(os.path.basename(pdf_to_test))[0]
            final_output_path = os.path.join(base_output_dir, pdf_filename_without_ext)
            
            print(f"\nPlease check the final output folder: {os.path.abspath(final_output_path)}")

        except Exception as main_e:
            print(f"An uncaught error occurred in the main test flow: {main_e}")

    print("\nPDF parsing test finished.")
    