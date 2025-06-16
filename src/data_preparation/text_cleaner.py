# src/data_preparation/text_cleaner.py

import re
import unicodedata
from typing import List, Optional


import spacy
_ = spacy.load("es_core_news_sm")

class TextCleaner:
    """
    Clase para limpiar y normalizar texto extraído de documentos académicos.
    Aplica una serie de operaciones de pre-procesamiento para preparar el texto
    para su posterior segmentación, embedding y construcción del grafo de conocimiento.
    """
    def __init__(self, lang_model: str = "es_core_news_sm"):
        """
        Inicializa el limpiador de texto con un modelo de SpaCy para lematización y stopwords.
        :param lang_model: Nombre del modelo de SpaCy para español (ej., 'es_core_news_sm').
        """
        self.nlp = None
        if spacy:
            try:
                self.nlp = spacy.load(lang_model)
                # Añadir patrones de detención específicos si es necesario (ej. palabras cubanas)
                # self.nlp.Defaults.stop_words.add("ej.")
            except Exception as e:
                print(f"No se pudo cargar el modelo de SpaCy '{lang_model}': {e}")
                print("La lematización y el filtrado de stopwords no se aplicarán.")

    def extract_math_formulas(self, text: str):
    # Extrae expresiones matemáticas tipo $...$ o ecuaciones simples
        pattern = r"""
            (                           # Grupo principal
            \$.*?\$                 # Fórmulas entre $...$
            |\[.*?\]                # Fórmulas entre \[...\]
            |\\begin\{equation\}.*?\\end\{equation\}   # Ambientes equation
            |[A-Za-z0-9\s\^\_\+\-\*/\(\)\[\]=\.]+(?:=|<=|>=|<|>)[A-Za-z0-9\s\^\_\+\-\*/\(\)\[\]=\.]+  # Ecuaciones tipo texto plano
            )
        """
        formulas = re.findall(pattern, text)
        text_sin_formulas = text
        for i, formula in enumerate(formulas):
            text_sin_formulas = text_sin_formulas.replace(formula, f"__FORMULA_{i}__")
        return text_sin_formulas, formulas
    
    def reincorporate_formulas(self, text: str, formulas: List[str]):
        for i, formula in enumerate(formulas):
            text = text.replace(f"__FORMULA_{i}__", formula)
        return text


    def remove_extra_whitespace(self, text: str) -> str:
        """
        Elimina múltiples espacios en blanco, saltos de línea y tabs,
        reemplazándolos por un solo espacio. También elimina espacios al inicio y al final.
        """
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def normalize_unicode(self, text: str) -> str:
        """
        Normaliza caracteres Unicode a su forma NFKD (ej., convierte 'á' a 'a', 'ñ' a 'n').
        Esto es útil para estandarizar el texto y evitar problemas de codificación o búsqueda.
        """
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    def remove_headers_footers(self, text: str, common_patterns: Optional[List[str]] = None) -> str:
        """
        Intenta eliminar patrones comunes de encabezados y pies de página que suelen
        aparecer en documentos escaneados o digitalizados.

        :param text: El texto a limpiar.
        :param common_patterns: Una lista de expresiones regulares para identificar
                                encabezados y pies de página. Si es None, usa un conjunto por defecto.
        """
        if common_patterns is None:
            # Estos patrones son ejemplos. Deberás ajustarlos para los documentos
            # específicos de los exámenes de ingreso y libros de texto cubanos.
            common_patterns = [
                r'(?i)página\s+\d+\s+de\s+\d+',      # "Página 1 de 10" (case-insensitive)
                r'^\s*\d+\s*$',                      # Líneas que contienen solo un número (posible número de página)
                r'^\s*\[\s*\d+\s*\]\s*$',            # Números de página entre corchetes "[ 5 ]"
                r'(?i)matemática\s*(?:[a-z]{1,2}\.)?\s*(?:-\s*\d+)?', # "Matemática", "Matemática V.", "Matemática - 10"
                r'(?i)español-literatura\s*(?:[a-z]{1,2}\.)?\s*(?:-\s*\d+)?',
                r'(?i)historia\s*(?:[a-z]{1,2}\.)?\s*(?:-\s*\d+)?',
                r'(?i)(?:libro|guía|examen|programa)\s+de\s+\S+\s*(?:[a-z]{1,2}\.)?', # "Libro de Matemática"
                # Añade aquí patrones específicos que encuentres en tus documentos
                # Ejemplo: r'Ministerio de Educación\s*-\s*Cuba'
                # Ejemplo: r'Centro Preuniversitario\s+\S+'
            ]

        cleaned_text = text
        for pattern in common_patterns:
            # flags=re.MULTILINE permite que '^' y '$' coincidan con el inicio/fin de cada línea
            # flags=re.IGNORECASE hace que la búsqueda no distinga mayúsculas de minúsculas
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
        return cleaned_text

    def remove_urls(self, text: str) -> str:
        """Elimina URLs (direcciones web) del texto."""
        # Un patrón de regex común para URLs
        return re.sub(r'(https?://[^\s]+)|(www\.[^\s]+)', '', text)

    def remove_emails(self, text: str) -> str:
        """Elimina direcciones de correo electrónico del texto."""
        return re.sub(r'\S*@\S*\s?', '', text)

    def remove_non_alphanumeric(self, text: str, keep_punctuation: str = '.,!?;:^') -> str:
        """
        Elimina caracteres no alfanuméricos (que no son letras, números o espacios),
        excepto la puntuación especificada. Esto ayuda a limpiar caracteres extraños.

        :param text: El texto a limpiar.
        :param keep_punctuation: Una cadena de caracteres de puntuación que se deben conservar.
                                 Por defecto, conserva los signos de puntuación comunes.
        """
        # Escapa los caracteres de puntuación para que se interpreten literalmente en la regex
        pattern = r'[^\w\s' + re.escape(keep_punctuation) + r']'
        text = re.sub(pattern, '', text)
        return text

    def convert_to_lowercase(self, text: str) -> str:
        """Convierte todo el texto a minúsculas para normalización."""
        return text.lower()

    def remove_stopwords_and_lemmatize(self, text: str, include_stopwords: bool = False) -> str:
        """
        Lematiza el texto (reduce las palabras a su forma base) y opcionalmente
        elimina las palabras vacías (stopwords) usando SpaCy.
        Esto es útil para reducir la dimensionalidad y mejorar la relevancia en la búsqueda.

        :param text: El texto a procesar.
        :param include_stopwords: Si es True, no elimina las stopwords.
        """
        if not self.nlp:
            print("Advertencia: SpaCy no está cargado. No se aplicará lematización ni filtrado de stopwords.")
            return text # Retorna el texto original si SpaCy no está disponible

        doc = self.nlp(text)
        tokens = []
        for token in doc:
            # Solo añadir si no es un espacio en blanco y no es una stopword (si include_stopwords es False)
            if not token.is_space:
                if include_stopwords or (not token.is_stop):
                    tokens.append(token.lemma_) # Añade el lema (forma base de la palabra)
        return " ".join(tokens)

    def clean_text(self, text: str, apply_lemmatization: bool = False) -> str:
        """
        Aplica una secuencia predefinida de operaciones de limpieza al texto.

        :param text: El texto en bruto a limpiar.
        :param apply_lemmatization: Si es True, aplica lematización y elimina stopwords.
        """
        cleaned_text , formulas = self.extract_math_formulas(text)
        cleaned_text = self.remove_extra_whitespace(text)
        cleaned_text = self.remove_headers_footers(cleaned_text) # Ejecutar ANTES de normalizar a minúsculas
                                                                # por si los patrones tienen mayúsculas exactas
        cleaned_text = self.remove_urls(cleaned_text)
        cleaned_text = self.remove_emails(cleaned_text)
        cleaned_text = self.normalize_unicode(cleaned_text) # Después de quitar URLs/Emails por si tienen acentos
        cleaned_text = self.remove_non_alphanumeric(cleaned_text, keep_punctuation='.,!?;:()[]{}%+-ⁿ₋₊–*/=^<>∅∈√∛⊆⊄℃⁰¹²³⁴⁵⁶⁷⁸⁹¼⅓⅖⅙⅛⅞↉⅟⅒⅝⅐⅘⅕¾∉∩∪≈≤≥⊈⊂∜∞∝') # Añade símbolos matemáticos
        cleaned_text = self.convert_to_lowercase(cleaned_text) # Convertir a minúsculas al final, después de procesar patrones sensibles a mayúsculas

        if apply_lemmatization:
            cleaned_text = self.remove_stopwords_and_lemmatize(cleaned_text)

        if(len(formulas)>0):
            print(formulas)
            cleaned_text = self.reincorporate_formulas(cleaned_text, formulas) # Reincorporar las fórmulas matemáticas

        cleaned_text = self.remove_extra_whitespace(cleaned_text) # Una pasada final para limpiar espacios sobrantes

        return cleaned_text

# Ejemplo de uso (esto es solo para demostración, la lógica principal iría en pipeline.py)
if __name__ == "__main__":
    cleaner = TextCleaner()

    sample_text_math = """
    Página 3 de 20
    MINISTERIO DE EDUCACIÓN - CUBA
    Guía de Matemática Superior [Capítulo 2.1]
    Ejercicio 5: Resuelva la ecuación cuadrática $x^2 + 2x - 3 = 0$.
    Para más información, visite www.matematica.edu.cu o escriba a info@edu.cu.
    ¡Importante! Consulte la sección 4.2 para teoremas relacionados.
    Este es un texto con    espacios   extra y
    saltos de línea.
    La solución implica el Teorema de Pitágoras (a^2+b^2=c^2).
    """

    print("--- Texto Original de Matemáticas ---")
    print(sample_text_math)

    print("\n--- Texto Limpio (sin lematización) ---")
    cleaned_math_text = cleaner.clean_text(sample_text_math, apply_lemmatization=False)
    print(cleaned_math_text)

    print("\n--- Texto Limpio y Lematizado (sin stopwords ni puntuación, excepto símbolos) ---")
    lemmatized_math_text = cleaner.clean_text(sample_text_math, apply_lemmatization=True)
    print(lemmatized_math_text)

    sample_text_history = """
    Historia de Cuba - Período Colonial
    Capítulo 1: La Conquista y Colonización.
    [Página 15]
    En 1492, Cristóbal Colón llegó a la isla de Cuba, un evento trascendental.
    Los taínos habitaban la región. La fecha clave es 1492.
    Para dudas, contacte a historiadores@cuba.gov.cu.
    """
    print("\n--- Texto Original de Historia ---")
    print(sample_text_history)

    print("\n--- Texto Limpio de Historia (sin lematización) ---")
    cleaned_history_text = cleaner.clean_text(sample_text_history, apply_lemmatization=False)
    print(cleaned_history_text)

    print("\n--- Texto Limpio y Lematizado de Historia ---")
    lemmatized_history_text = cleaner.clean_text(sample_text_history, apply_lemmatization=True)
    print(lemmatized_history_text)