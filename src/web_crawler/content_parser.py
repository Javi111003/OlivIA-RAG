# SCRAPPER 
#   Una vez que el crawler_manager le entrega el contenido 
#   (ej. HTML), este módulo debe ser capaz de extraer el texto relevante y
#   , crucialmente, las fórmulas o estructuras matemáticas de ese HTML. 
#   Esto implica entender la estructura HTML de los sitios web de donde sacarán la información

import re
from headers import HEADERS
import requests
from bs4 import BeautifulSoup

def extract_main_content(soup: BeautifulSoup):
    main_content_div = soup.find('main') or \
                        soup.find('article') or \
                        soup.find('div', id='content') or \
                        soup.find('div', id='main-content') or \
                        soup.find('div', class_=re.compile(r'^(article|post|main|body)-content$'))
    if main_content_div:
        for unwanted_tag_name in ['nav', 'aside', 'footer', 'header', 'form', 'script', 'style', 'iframe', 'img']:
            for tag in main_content_div.find_all(unwanted_tag_name):
                tag.decompose()
        text = main_content_div.get_text(separator='\n', strip=True)
        text = re.sub(r'\n\s*\n', '\n\n', text) 
        text = re.sub(r'[ \t]+', ' ', text)
    paragraphs = soup.find_all('p')
    if paragraphs:
        text = '\n'.join([p.get_text(strip=True) for p in paragraphs])
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text
    return ""

def extract_headers(soup: BeautifulSoup) -> list[str]:
    headers = []
    for i in range(1, 7):
        for h_tag in soup.find_all(f"h{i}"):
            text = h_tag.get_text(strip= True)
            if text:
                headers.append(text)
    return headers

def extract_historical_dates(text: str) -> list[str]:
    historical_dates = []
    historical_dates.extend(re.findall(r'\b(0?[1-9]|[12]\d|3[01])[/.-](0?[1-9]|1[0-2])[/.-](1[89]\d{2}|2000)\b', text))

    meses_es_regex = "(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)"
    historical_dates.extend(re.findall(rf'\b(0?[1-9]|[12]\d|3[01]) de {meses_es_regex} de (1[89]\d{{2}}|2000)\b', text, re.IGNORECASE))
    return historical_dates

def extract_math_forms_with_regex(soup: BeautifulSoup) -> list[str]:
    math_forms = []
    math_forms_regex_1 = re.findall(r'\b[a-zA-Z]?\s*\d*\.?\d*\s*[a-zA-Z]\^2\s*([+\-]\s*\d*\.?\d*\s*[a-zA-Z])?\s*([+\-]\s*\d*\.?\d*)?\s*=\s*(\d*\.?\d*|[a-zA-Z])\b', soup.text, re.IGNORECASE)
    math_forms.extend(math_forms_regex_1)
    math_forms_regex_2 = re.findall(r'\b(sen|cos|tan|cot|sec|csc|sin|tg)\s*\((\s*\d*\.?\d*[a-zA-Z]*([+\-*/]\d*\.?\d*[a-zA-Z]*)*\s*)\)\s*=\s*.*?', soup.text, re.IGNORECASE)
    math_forms.extend(math_forms_regex_2)
    math_forms_regex_3 = re.findall(r'\b(log|ln)(_?\d*\.?\d*)?\s*(\d*\.?\d*|[a-zA-Z]|\([^)]+\))\s*=\s*.*?', soup.text, re.IGNORECASE)
    math_forms.extend(math_forms_regex_3)
    math_forms_regex_4 = re.findall(r'\b√\s*(\(([^)]*?)\)|[a-zA-Z\d]+)\s*=\s*.*?', soup.text, re.IGNORECASE)
    math_forms.extend(math_forms_regex_4)
    return math_forms

def _extract_math_from_soup(soup: BeautifulSoup) -> list[str]:
    found_formulas = []

    math_containers_by_class = soup.find_all(class_=[
        "MathJax_CHTML", "MathJax", "MJX_Assistive_MathML", "MathJax_SVG",
        "katex", "katex-display", "katex-mathml",
        "equation", "formula", "math", "math-display", "math-inline",
        "mwe-math-element", "mwe-math-mathml-a11y"
    ])

    for container in math_containers_by_class:
        if container.name == 'math' or 'MathML' in container.get('class', ''):
            found_formulas.append(str(container))
        else:
            text = container.get_text(separator=' ', strip=True)
            if text:
                found_formulas.append(text)

    for math_tag in soup.find_all('math'):
        found_formulas.append(str(math_tag)) 

    for code_block in soup.find_all(['pre', 'code', 'div']):
        text_content = code_block.get_text()

        inline_latex_matches = re.findall(r'\$(.*?)\$', text_content)
        display_latex_matches = re.findall(r'\$\$(.*?)\$\$', text_content)

        paren_latex_matches = re.findall(r'\\\((.*?)\\\)', text_content)
        square_latex_matches = re.findall(r'\\\[(.*?)\\]', text_content)

        found_formulas.extend(inline_latex_matches)
        found_formulas.extend(display_latex_matches)
        found_formulas.extend(paren_latex_matches)
        found_formulas.extend(square_latex_matches)
    return list(set(f.strip() for f in found_formulas if f.strip()))

def extract_content_from_url(url: str, subject: str):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except:
        return None
    title = soup.title.string.strip() if soup.title else "Not Title"
    main_text = extract_main_content(soup)
    headings_list = extract_headers(soup)
    historical_dates = extract_historical_dates(soup.text) if subject == "hist" else None
    math_forms = (extract_math_forms_with_regex(soup) + (_extract_math_from_soup(soup))) if subject == "math" else None
    return {
            "url": url,
            "title": title,
            "main_content_text": main_text,
            "headings": headings_list,
            "historical_dates": historical_dates,
            "math_forms": math_forms
    }