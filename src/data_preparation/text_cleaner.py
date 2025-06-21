# src/data_preparation/text_cleaner.py

import re
import unicodedata
from typing import List, Dict, Any, Optional

import spacy

try:
    nlp_model = spacy.load("es_core_news_md")
    # añadir stopwords personalizadas aquí si es necesario
    # nlp_model.Defaults.stop_words.add("ej.")
except Exception as e:
    nlp_model = None
    print(f"No se pudo cargar el modelo de SpaCy 'es_core_news_md': {e}")
    print("La lematización y el filtrado de stopwords no se aplicarán en TextCleaner.")


class TextCleaner:
    """
    Clase para limpiar y normalizar texto extraído de documentos académicos.
    Aplica una serie de operaciones de pre-procesamiento para preparar el texto
    para su posterior segmentación, embedding y construcción del grafo de conocimiento.
    Ahora trabaja con la salida estructurada de Unstructured.io (lista de diccionarios).
    """
    def __init__(self):
        """
        Inicializa el limpiador de texto. El modelo de SpaCy se carga una vez a nivel de módulo.
        """
        self.nlp = nlp_model # Usar la instancia cargada globalmente

    def extract_math_formulas(self, text: str):

        pattern = r"""
            (                                       # Grupo principal
            \$.*?\$                                 # Fórmulas entre $...$
            |\[.*?\]                                 # Fórmulas entre \[...\] (Latex display math)
            |\\begin\{equation\}.*?\\end\{equation\} # Ambientes equation
            |[\w\s\^\_\+\-\*/\(\)\[\]=\.]+(?:=|<=|>=|<|>)[A-Za-z0-9\s\^\_\+\-\*/\(\)\[\]=\.]+ # Ecuaciones tipo texto plano
            )
        """
        formulas = re.findall(pattern, text, re.VERBOSE | re.DOTALL)
        
        text_sin_formulas = text
        for i, formula in enumerate(formulas):
            text_sin_formulas = text_sin_formulas.replace(formula, f"__FORMULA_{i}__")
        return text_sin_formulas, formulas
    
    def reincorporate_formulas(self, text: str, formulas: List[str]):
        for i, formula in enumerate(formulas):
            text = re.sub(re.escape(f"__FORMULA_{i}__"), formula, text, 1)
        return text

    def remove_extra_whitespace(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    def remove_headers_footers(self, text: str, common_patterns: Optional[List[str]] = None) -> str:
        """
        Intenta eliminar patrones comunes de encabezados y pies de página.
        Ajustado para no eliminar títulos de documentos completos.
        """
        if common_patterns is None:
            common_patterns = [
                r'(?i)página\s+\d+\s+de\s+\d+',      # "Página 1 de 10" (case-insensitive)
                r'^\s*\d+\s*$',                      # Líneas que contienen solo un número (posible número de página)
                r'^\s*\[\s*\d+\s*\]\s*$',            # Números de página entre corchetes "[ 5 ]"
                # Patrones más específicos para temas, solo si se repiten de forma identificable como header/footer
                r'(?i)matemática\s*(?:[a-z]{1,2}\.)?\s*(?:-\s*\d+)?', # "Matemática", "Matemática V.", "Matemática - 10"
                r'(?i)español-literatura\s*(?:[a-z]{1,2}\.)?\s*(?:-\s*\d+)?',
                r'(?i)historia\s*(?:[a-z]{1,2}\.)?\s*(?:-\s*\d+)?',
                # REMOVIDO: r'(?i)(?:libro|guía|examen|programa)\s+de\s+\S+\s*(?:[a-z]{1,2}\.)?',
                r'Ministerio de Educación\s*-\s*Cuba',
                r'Centro Preuniversitario\s+\S+'
            ]

        cleaned_text = text
        for pattern in common_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
        return cleaned_text

    def remove_urls(self, text: str) -> str:
        return re.sub(r'(https?://[^\s]+)|(www\.[^\s]+)', '', text)

    def remove_emails(self, text: str) -> str:
        return re.sub(r'\S*@\S*\s?', '', text)

    def remove_non_alphanumeric(self, text: str, keep_punctuation: str = '.,!?;:()[]{}%+-ⁿ₋₊–*/=^<>∅∈√∛⊆⊄℃⁰¹²³⁴⁵⁶⁷⁸⁹¼⅓⅖⅙⅛⅞↉⅟⅒⅝⅐⅘⅕¾∉∩∪≈≤≥⊈⊂∜∞∝') -> str:
        pattern = r'[^\w\s' + re.escape(keep_punctuation) + r']'
        text = re.sub(pattern, '', text)
        return text

    def convert_to_lowercase(self, text: str) -> str:
        return text.lower()

    def remove_stopwords_and_lemmatize(self, text: str, include_stopwords: bool = False) -> str:
        if not self.nlp:
            return text

        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if not token.is_space:
                if include_stopwords or (not token.is_stop):
                    tokens.append(token.lemma_)
        return " ".join(tokens)

    def clean_element_content(self, content: str, element_type: str, apply_lemmatization: bool = False) -> str:
        """
        Aplica una secuencia predefinida de operaciones de limpieza al contenido de un solo elemento.
        Permite adaptar la limpieza según el tipo de elemento.

        :param content: El texto en bruto del elemento a limpiar.
        :param element_type: El tipo de elemento (ej., 'Title', 'NarrativeText', 'Table').
        :param apply_lemmatization: Si es True, aplica lematización y elimina stopwords.
        """
        formulas = []
        cleaned_content = self.remove_urls(content)
        cleaned_content = self.remove_emails(cleaned_content)      
                
        cleaned_content, formulas = self.extract_math_formulas(cleaned_content)
                
        # Consideraciones especiales según el tipo de elemento
        if element_type == 'Table':
            # Las tablas ya vienen en Markdown
            cleaned_content = self.remove_extra_whitespace(cleaned_content)
            if formulas: 
                cleaned_content = self.reincorporate_formulas(cleaned_content, formulas)
                
            return cleaned_content # Terminar aquí para tablas

        # Limpieza general para otros tipos de elementos (NarrativeText, Title, CompositeElement, etc.)
        cleaned_content = self.remove_headers_footers(cleaned_content) 
        cleaned_content = self.remove_extra_whitespace(cleaned_content)    
        cleaned_content = self.normalize_unicode(cleaned_content) 
        cleaned_content = self.remove_non_alphanumeric(cleaned_content, keep_punctuation='.,!?;:()[]{}%+-ⁿ₋₊–*/=^<>∅∈√∛⊆⊄℃⁰¹²³⁴⁵⁶⁷⁸⁹¼⅓⅖⅙⅛⅞↉⅟⅒⅝⅐⅘⅕¾∉∩∪≈≤≥⊈⊂∜∞∝●')
        if formulas:
            cleaned_content = self.reincorporate_formulas(cleaned_content, formulas)
        if apply_lemmatization:
            cleaned_content = self.remove_stopwords_and_lemmatize(cleaned_content)
        cleaned_content = self.convert_to_lowercase(cleaned_content) 
        cleaned_content = self.remove_extra_whitespace(cleaned_content) 
        return cleaned_content

    def clean_documents(self, elements: List[Dict[str, Any]], apply_lemmatization: bool = False) -> List[Dict[str, Any]]:
        """
        Limpia una lista de elementos de documento estructurados (salida de DocumentLoader).
        Itera sobre cada elemento, limpia su contenido y devuelve la lista actualizada.

        :param elements: Una lista de diccionarios, donde cada diccionario es un elemento
                         extraído por Unstructured.io (content + metadata).
        :param apply_lemmatization: Si es True, aplica lematización y elimina stopwords.
        :return: La lista de elementos con el contenido limpio.
        """
        cleaned_elements: List[Dict[str, Any]] = []
        for element in elements:
            original_content = element.get("content", "")
            element_type = element["metadata"].get("type", "NarrativeText") # Obtener el tipo de elemento
            
            cleaned_content = self.clean_element_content(original_content, element_type, apply_lemmatization)
            
            cleaned_element = element.copy()
            cleaned_element["content"] = cleaned_content
            cleaned_element["cleaned_content"] = cleaned_content  
            cleaned_elements.append(cleaned_element)
            
        return cleaned_elements

# --- Código de prueba (para ejecutar este módulo directamente) ---
if __name__ == "__main__":
    import os
    import sys
    import uuid

    sample_elements_from_loader = [
        {
            "content": "EXAMEN DE INGRESO A LA EDUCACIÓN SUPERIOR\n\nCurso 2017-2018 1ra convocatoria\n\n1. Lee detenidamente y responde.\n\n1.1. Clasifica las siguientes proposiciones en verdaderas (V) o falsas (F). Escribe V o F en la línea dada. De las que consideres falsas, justifica por qué lo son.\n\na) ___ Sean A y B dos conjuntos no vacíos tales que, xA y xA\\\\B, entonces xB.\n\nb) ___ La función f definida en {x: x > 3} por la ecuación\n\nf(x) = log(x – 3) es inyectiva.",
            "metadata": {
                "source": "dummy_doc.pdf", "page_number": 1, "coordinates": None,
                "type": "CompositeElement", "chunk_id": str(uuid.uuid4())
            }
        },
        {
            "content": "1.2.2. En la figura se muestra el arco que\n\ny (en metros)\n\ndescribe un puente elevado que tiene forma\n\nde parábola, cuya ecuación es\n\ny =\n\n441 ye, 2 160\n\n. Si la altura máxima del\n\nA ● 0 80 x (en metros)\n\npuente la alcanza en el punto A, entonces su altura es igual a:\n\na) ___ 5 m b) ___ 10 m c) ___ 40 m d) ___ 80 m\n\n1.3. Completa los espacios en blanco de forma que se obtenga una proposición verdadera para cada caso:\n\nSean A(– 4 ; 1) ; B(1 ; – 1) ; C(3 ; 1) y D(– 2 ; 3), los vértices",
            "metadata": {
                "source": "dummy_doc.pdf", "page_number": 2, "coordinates": None,
                "type": "CompositeElement", "chunk_id": str(uuid.uuid4())
            }
        },
        {
            "content": "consecutivos de un paralelogramo ABCD.\n\n1.3.1. La diagonal tiene una longitud igual a _____ unidades. 1.3.2. La recta que contiene a la diagonal tiene como ecuación\n\n________.\n\n2. En la figura se muestra una circunferencia de centro O y diámetro\n\nAB\n\n, además se conoce que:\n\nradiosOB, OC\n\nM, N y P son puntos de los radios\n\ny\n\nrespectivamente,\n\nOMNP rombo,\n\n=",
            "metadata": {
                "source": "dummy_doc.pdf", "page_number": 2, "coordinates": None,
                "type": "CompositeElement", "chunk_id": str(uuid.uuid4())
            }
        },
        {
            "content": "| Header de prueba |\n|---|---|\n| Dato 1 | Valor A |\n| Dato 2 | Valor B |\n\nTexto adicional después de la tabla.",
            "metadata": {
                "source": "dummy_doc.pdf", "page_number": 3, "coordinates": None,
                "type": "Table", "chunk_id": str(uuid.uuid4())
            }
        },
        {
            "content": "Capítulo 5: Teoremas Fundamentales\nPara más información, visite www.ejemplo.com o escriba a contacto@email.org.\n[ 10 ]",
            "metadata": {
                "source": "dummy_doc.pdf", "page_number": 10, "coordinates": None,
                "type": "Title", "chunk_id": str(uuid.uuid4())
            }
        }
    ]

    cleaner = TextCleaner()

    print("--- Elementos Originales ---")
    for i, elem in enumerate(sample_elements_from_loader):
        print(f"\nElemento {i+1} (Tipo: {elem['metadata']['type']}):")
        print(elem['content'][:200]) # Mostrar solo el inicio 

    print("\n--- Elementos Limpios (sin lematización) ---")
    cleaned_elements_no_lemma = cleaner.clean_documents(sample_elements_from_loader, apply_lemmatization=False)
    for i, elem in enumerate(cleaned_elements_no_lemma):
        print(f"\nElemento {i+1} (Tipo: {elem['metadata']['type']}):")
        print(elem['content'][:200])

    print("\n--- Elementos Limpios y Lematizados ---")
    cleaned_elements_with_lemma = cleaner.clean_documents(sample_elements_from_loader, apply_lemmatization=True)
    for i, elem in enumerate(cleaned_elements_with_lemma):
        print(f"\nElemento {i+1} (Tipo: {elem['metadata']['type']}):")
        print(elem['content'][:200])