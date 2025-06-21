import os
from typing import List, Dict, Any, Union
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
import json
import uuid 
import spacy

class DocumentLoader:
    """
    Carga documentos desde un directorio especificado y sus subdirectorios,
    procesándolos con UnstructuredFileLoader para extraer elementos estructurados.
    Añade el 'document_type' a los metadatos de cada elemento basado en la ruta del archivo.
    """
    def __init__(self, base_data_dir: str):
        """
        Inicializa el cargador de documentos.
        :param base_data_dir: Directorio base (ej., './data/raw') que contiene subcarpetas como 'exams' y 'books'.
        """
        self.nlp = spacy.load("es_core_news_md")  # Cargar modelo de lenguaje en español
        self.base_data_dir = base_data_dir
        # Extensiones que UnstructuredFileLoader puede manejar y que deseas procesar
        self.supported_extensions = ['.pdf', '.docx', '.html', '.txt', '.md']
        self.vocabulary = []
        
    def _save_vocabulary(self, elements: List[Dict[str, Any]], lemmatize: bool):
        """
        Guarda el vocabulario en un archivo JSON.
        :param vocabulary: Lista de términos del vocabulario.
        :param output_path: Ruta donde se guardará el archivo JSON.
        """
        all_text = [item['content'] for item in elements]
        doc = self.nlp(all_text)
        vocabulary = []
        if lemmatize:
            vocabulary = list(set(token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop and not token.is_punct))
        else : 
            vocabulary = list(set(token.text.lower() for token in doc if token.is_alpha and not token.is_stop and not token.is_punct))
        self.vocabulary = vocabulary
        output_path = os.path.join(self.base_data_dir, 'vocabulary.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocabulary, f, ensure_ascii=False, indent=4)
        print(f"Vocabulario guardado en: {output_path}")
                
    def _infer_document_type_from_path(self, filepath: str) -> str:
        """
        Infiere el tipo de documento ('exam', 'book', 'unknown') basado en la ruta del archivo.
        Esto asume que los archivos están en subdirectorios nombrados 'exams' o 'books'
        directamente bajo el base_data_dir o sus subniveles.
        """
        normalized_path = os.path.normcase(filepath)
        
        try:
            relative_path = os.path.relpath(normalized_path, self.base_data_dir)
        except ValueError:
            relative_path = normalized_path

        path_parts = relative_path.split(os.sep)

        if 'exams' in path_parts:
            return 'exam'
        elif 'books' in path_parts:
            return 'book'
        else:
            return 'unknown'

    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Carga un documento específico utilizando UnstructuredFileLoader
        y enriquece sus metadatos, incluyendo el 'document_type'.

        :param file_path: Ruta completa al archivo.
        :return: Una lista de diccionarios con contenido y metadatos enriquecidos.
        """
        if not os.path.exists(file_path):
            print(f"Advertencia: Archivo no encontrado en: {file_path}")
            return []

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_extensions:
            print(f"Advertencia: Tipo de archivo '{file_ext}' no compatible para procesamiento con UnstructuredFileLoader. Saltando: {file_path}")
            return [] # Retornar lista vacía silenciosamente si no es compatible

        print(f"Cargando y procesando '{file_path}' con Unstructured...")
        try:
            loader = UnstructuredFileLoader(
                file_path,
                mode="elements",
                strategy="hi_res",
                chunking_strategy="basic"
            )
            elements: List[Document] = loader.load()

            processed_elements: List[Dict[str, Any]] = []
            
            doc_type = self._infer_document_type_from_path(file_path)
            
            for element in elements:
                metadata = element.metadata
                
                element_type = metadata.get('category') or metadata.get('type', 'NarrativeText')
                
                processed_metadata = {
                    "source": metadata.get('source', file_path),
                    "page_number": metadata.get('page_number'),
                    "coordinates": metadata.get('coordinates'),
                    "coordinate_system": metadata.get('coordinate_system'),
                    "type": element_type, # Tipo de elemento (Title, NarrativeText, Table, etc.)
                    "filetype": metadata.get('filetype'),
                    "filename": os.path.basename(metadata.get('filename', file_path)), # Solo el nombre del archivo
                    "last_modified": metadata.get('last_modified'),
                    "chunk_id": str(uuid.uuid4()), 
                    "document_type": doc_type 
                }
                
                content = element.page_content

                processed_elements.append({
                    "content": content, 
                    "metadata": processed_metadata
                })
            
            print(f"Documento '{os.path.basename(file_path)}' procesado. Tipo inferido: '{doc_type}'. Se extrajeron {len(processed_elements)} elementos.")
            return processed_elements

        except Exception as e:
            print(f"Error al cargar o procesar '{file_path}': {e}")
            return []

    def load_all_documents_from_base_dir(self) -> List[Dict[str, Any]]:
        """
        Carga recursivamente todos los documentos soportados dentro del directorio base
        (self.base_data_dir) y sus subdirectorios.
        Este es el método principal para cargar todos los documentos en el pipeline.
        """
        all_elements: List[Dict[str, Any]] = []
        print(f"Iniciando carga de documentos desde: {self.base_data_dir}")
        for root, dirs, files in os.walk(self.base_data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                elements_from_file = self.load_document(file_path)
                if elements_from_file:
                    all_elements.extend(elements_from_file)
        print(f"Carga completa. Se encontraron y procesaron {len(all_elements)} elementos.")
        self._save_vocabulary(self, all_elements, lemmatize=False)
        return all_elements

# --- Código de prueba (para ejecutar este módulo directamente) ---
if __name__ == "__main__":
    
    current_script_dir = os.path.dirname(__file__)

    project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..')) 
    
    RAW_DATA_ROOT = os.path.join(project_root, '.data', 'raw')
    print(f"DEBUG: RAW_DATA_ROOT is set to: {RAW_DATA_ROOT}")
    loader = DocumentLoader(base_data_dir=RAW_DATA_ROOT)
    
    print(f"Cargando todos los documentos REALES desde: {RAW_DATA_ROOT}")
    all_extracted_elements = loader.load_all_documents_from_base_dir()

    if not all_extracted_elements:
        print("No se extrajeron elementos. Revisa las rutas y los archivos en tu directorio RAW.")
    else:
        print(f"\n--- Se extrajeron {len(all_extracted_elements)} elementos de todos los documentos REALES ---")
        
        exam_elements = [e for e in all_extracted_elements if e['metadata'].get('document_type') == 'exam']
        book_elements = [e for e in all_extracted_elements if e['metadata'].get('document_type') == 'book']
        unknown_elements = [e for e in all_extracted_elements if e['metadata'].get('document_type') == 'unknown']

        print(f"\nTotal de elementos de Exámenes: {len(exam_elements)}")
        if exam_elements:
            print("\nEjemplo de elemento de Examen (primero encontrado):")
            print(f"  Contenido (inicio): {exam_elements[0]['content'][:150]}...")
            print(f"  Metadata: {json.dumps(exam_elements[0]['metadata'], indent=2)}")


        print(f"\nTotal de elementos de Libros: {len(book_elements)}")
        if book_elements:
            print("\nEjemplo de elemento de Libro (primero encontrado):")
            print(f"  Contenido (inicio): {book_elements[0]['content'][:150]}...")
            print(f"  Metadata: {json.dumps(book_elements[0]['metadata'], indent=2)}")
            
        print(f"\nTotal de elementos de tipo 'unknown': {len(unknown_elements)}")
        if unknown_elements:
            print("\nEjemplo de elemento 'unknown' (primero encontrado):")
            print(f"  Contenido (inicio): {unknown_elements[0]['content'][:150]}...")
            print(f"  Metadata: {json.dumps(unknown_elements[0]['metadata'], indent=2)}")

        output_dir = os.path.join(project_root, '.data', 'processed_loader_output')
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, 'all_raw_extracted_elements.json')
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_extracted_elements, f, ensure_ascii=False, indent=4)
        print(f"\nTodos los elementos extraídos guardados en: {output_filepath}")