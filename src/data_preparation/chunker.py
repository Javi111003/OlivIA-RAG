# src/data_preparation/semantic_chunker.py

from typing import List, Dict, Any, Tuple
import re
import uuid

class SemanticChunker:
    def __init__(self):

        self.EXAM_START_PATTERN = re.compile(r'^\s*EXAMEN DE INGRESO A LA EDUCACIÓN SUPERIOR\s*|^Curso \d{4}-\d{4} \d{1,2}ra convocatoria', re.IGNORECASE)
        self.EXERCISE_START_PATTERN = re.compile(r'^((\d+(\.\d+)*)\.?)\s*(.*)', re.IGNORECASE)
        self.SOLUTION_BLOCK_START_PATTERN = re.compile(r'^\s*Solución\s*:\s*', re.IGNORECASE)
        self.SPECIFIC_SOLUTION_PATTERN = re.compile(r'^\s*Solucion\s+Ejercicio\s+((\d+(\.\d+)*)\.?)\s*(.*)', re.IGNORECASE)
        
        # Book/Generic Patterns
        # Este patrón es flexible para 'Capítulo X', 'Tema X', 'Unidad X', 'I.', 'II.', '1.', '1.1.' seguido de un título
        self.BOOK_SECTION_TITLE_PATTERN = re.compile(r'^(Capítulo|Tema|Unidad|I{1,4}|V|X|\d+(\.\d+)*)\s*[:\.]?\s+(.*)', re.IGNORECASE)

    def chunk_documents(self, cleaned_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main method to chunk documents based on their type.
        It determines the type of document and calls the appropriate chunking method.

        :param cleaned_elements: List of cleaned elements from TextCleaner.
        :return: List of enriched chunks.
        """
        if not cleaned_elements:
            print("Warning: No cleaned elements provided for chunking.")
            return []

        first_element = cleaned_elements[0]
        document_type = first_element['metadata'].get('document_type', 'exam')
        if not document_type:
            # Fallback to a heuristic based on content or filename
            if 'exam' in first_element['metadata'].get('filename', '').lower():
                document_type = 'exam'
            elif 'book' in first_element['metadata'].get('filename', '').lower():
                document_type = 'book'
            else:
                document_type = 'generic'
        print(f"Detected document type: {document_type}")
        return self.chunk_document(cleaned_elements, document_type)
    
    def chunk_document(self, cleaned_elements: List[Dict[str, Any]], document_type: str = "exam") -> List[Dict[str, Any]]:
        """
        Main method to chunk documents based on their type.

        :param cleaned_elements: List of cleaned elements from TextCleaner.
        :param document_type: "exam" or "book" (can be extended).
        :return: List of enriched chunks.
        """
        if document_type.lower() == "exam":
            return self._chunk_exam_document(cleaned_elements)
        elif document_type.lower() == "book":
            return self._chunk_book_document(cleaned_elements)
        else:
            # Fallback for unknown types, perhaps a simple fixed-size chunker
            print(f"Warning: Unknown document type '{document_type}'. Using basic chunking.")
            return self._chunk_generic_document(cleaned_elements) # To be implemented

    def _create_empty_block(self) -> Dict[str, Any]:
        """Initializes an empty logical block structure."""
        return {
            "exercise_id": None,
            "section_title": None,
            "content_elements": [],
            "solution_elements": [],
            "is_general_solution_block": False,
            "metadata_accumulator": {
                "page_numbers": set(),
                "element_types": set(),
                "chunk_type": "generic_content",
                "is_solution": False
            }
        }

    def _finalize_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the block as is for further processing."""
        return block

    def _create_chunk_from_block(self, content: str, block: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """Creates a final chunk dictionary from combined content and accumulated block metadata."""
        chunk_metadata = {
            "chunk_id": str(uuid.uuid4()),
            "source_file": source_file,
            "page_numbers": sorted(list(block["metadata_accumulator"].get("page_numbers", []))),
            "element_types_in_chunk": sorted(list(block["metadata_accumulator"].get("element_types", []))),
            "exercise_id": block["exercise_id"],
            "section_title": block["section_title"],
            "chunk_type": block["metadata_accumulator"].get("chunk_type"),
            "is_solution": block["metadata_accumulator"].get("is_solution")
        }
        chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None and v != [] and (not isinstance(v, list) or v)}
        return {"content": content, "metadata": chunk_metadata}

    # --- Exam Specific Chunker Method ---
    def _chunk_exam_document(self, exam_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_exam_chunks = []
        
        current_exam_elements = []
        current_exam_filename = None
        
        for i, element in enumerate(exam_elements):
            filename = element['metadata'].get('filename', 'unknown_exam')
            content = element['cleaned_content']
            
            # Condition for a new exam:
            # 1. First element
            # 2. Filename changes
            # 3. EXAM_START_PATTERN matches and it's not the very first element of the whole document
            is_new_exam_start = (i == 0) or \
                                 (current_exam_filename is not None and current_exam_filename != filename) or \
                                 (self.EXAM_START_PATTERN.match(content) and i > 0)
            
            if is_new_exam_start:
                if current_exam_elements: # Process the elements of the previous exam block
                    all_exam_chunks.extend(self._process_exam_block(current_exam_elements, current_exam_filename))
                
                current_exam_elements = [element]
                current_exam_filename = filename
                
                # If it's an explicit exam header, create a dedicated chunk for it
                if self.EXAM_START_PATTERN.match(content):
                    exam_header_chunk = self._create_chunk_from_block(
                        content=content.strip(),
                        block=self._create_empty_block(),
                        source_file=current_exam_filename
                    )
                    exam_header_chunk['metadata']['chunk_type'] = 'exam_header'
                    exam_header_chunk['metadata']['is_solution'] = False
                    all_exam_chunks.append(exam_header_chunk)
                    current_exam_elements = [] # Clear elements as this header is now its own chunk
            else:
                current_exam_elements.append(element)
        
        # Process the last collected exam block
        if current_exam_elements:
            all_exam_chunks.extend(self._process_exam_block(current_exam_elements, current_exam_filename))
            
        return all_exam_chunks

    def _process_exam_block(self, exam_elements: List[Dict[str, Any]], exam_filename: str) -> List[Dict[str, Any]]:
        """
        Internal method to chunk elements within a single exam, based on exercise and solution patterns.
        """
        logical_blocks: List[Dict[str, Any]] = []
        current_block = self._create_empty_block()

        for element in exam_elements:
            content = element['cleaned_content']
            element_type = element['metadata'].get('type', 'NarrativeText') 
            metadata = element['metadata']

            page_num = metadata.get('page_number')
            if page_num:
                current_block["metadata_accumulator"]["page_numbers"].add(page_num)
            current_block["metadata_accumulator"]["element_types"].add(element_type)

            exercise_match = self.EXERCISE_START_PATTERN.match(content)
            section_match = self.BOOK_SECTION_TITLE_PATTERN.match(content) 
            solution_block_match = self.SOLUTION_BLOCK_START_PATTERN.match(content)
            specific_solution_match = self.SPECIFIC_SOLUTION_PATTERN.match(content)
            
            # Logic for detecting new blocks within an exam (ordered by priority)
            
            # 1. New Section (e.g., "Capítulo X")
            if section_match and element_type in ['Title', 'CompositeElement']:
                if current_block["content_elements"] or current_block["solution_elements"]:
                    logical_blocks.append(self._finalize_block(current_block))
                
                current_block = self._create_empty_block()
                current_block["section_title"] = section_match.group(0).strip()
                current_block["content_elements"].append(element)
                current_block["metadata_accumulator"]["chunk_type"] = "exam_section_title"
            
            # 2. New Exercise
            elif exercise_match:
                if current_block["content_elements"] or current_block["solution_elements"]:
                    logical_blocks.append(self._finalize_block(current_block))
                
                current_block = self._create_empty_block()
                current_block["exercise_id"] = exercise_match.group(1).strip('.')
                current_block["content_elements"].append(element)
                current_block["metadata_accumulator"]["chunk_type"] = "exercise"
                
            # 3. General Solution Block ("Solución :")
            elif solution_block_match:
                if current_block["content_elements"] or current_block["solution_elements"]:
                    logical_blocks.append(self._finalize_block(current_block))
                
                current_block = self._create_empty_block()
                current_block["is_general_solution_block"] = True
                current_block["solution_elements"].append(element) # The "Solución :" title itself is part of solutions
                current_block["metadata_accumulator"]["chunk_type"] = "general_solution"
                current_block["metadata_accumulator"]["is_solution"] = True

            # 4. Specific Solution ("Solucion Ejercicio X.Y.Z")
            elif specific_solution_match:
                solution_target_id = specific_solution_match.group(1).strip('.')
                
                # Attempt to append to the current exercise block if it matches, otherwise start a new one
                if current_block["exercise_id"] == solution_target_id and not current_block.get("is_general_solution_block"):
                    current_block["solution_elements"].append(element)
                    current_block["metadata_accumulator"]["is_solution"] = True
                else: # Mismatch or currently in a general solution block, so start a new specific solution block
                    if current_block["content_elements"] or current_block["solution_elements"]:
                        logical_blocks.append(self._finalize_block(current_block))
                    
                    current_block = self._create_empty_block()
                    current_block["exercise_id"] = solution_target_id # Associate this block with the target exercise
                    current_block["solution_elements"].append(element)
                    current_block["metadata_accumulator"]["chunk_type"] = "specific_solution"
                    current_block["metadata_accumulator"]["is_solution"] = True
            
            # 5. Regular content, append to current block
            else:
                if current_block.get("is_general_solution_block"):
                    current_block["solution_elements"].append(element)
                else:
                    current_block["content_elements"].append(element)
                    
        if current_block["content_elements"] or current_block["solution_elements"]:
            logical_blocks.append(self._finalize_block(current_block))

        # --- Create final chunks for this exam block ---
        final_chunks_in_exam = []
        for block in logical_blocks:
            chunk_content = ""
            
            if block["metadata_accumulator"].get("chunk_type") == "exam_section_title":
                if block["content_elements"]:
                    chunk_content = block["content_elements"][0]['cleaned_content']
            else: 
                for elem in block["content_elements"]:
                    chunk_content += elem['cleaned_content'] + "\n"
                
                if block["solution_elements"]:
                    if block["metadata_accumulator"].get("chunk_type") == "exercise":
                           chunk_content += "\nSolución:\n"
                    elif block["metadata_accumulator"].get("chunk_type") == "general_solution":
                        if not chunk_content.strip().startswith("Solución General:"): 
                            chunk_content = "Solución General:\n" + chunk_content
                    
                    for elem in block["solution_elements"]:
                        chunk_content += elem['cleaned_content'] + "\n"
            
            if chunk_content.strip(): 
                final_chunks_in_exam.append(self._create_chunk_from_block(chunk_content.strip(), block, exam_filename))
                
        return final_chunks_in_exam


    # --- Book Specific Chunker Method ---
    def _chunk_book_document(self, book_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_chunks = []
        current_section_elements = [] # Elements for the main narrative of the current section
        current_section_title = None
        current_section_page_numbers = set()
        
        section_exercises_map = {} # { "1.1": {"question_elements": [], "solution_elements": []} }
        general_section_solutions = []

        for i, element in enumerate(book_elements):
            content = element['cleaned_content']
            element_type = element['metadata'].get('type', 'NarrativeText')
            metadata = element['metadata']
            
            page_num = metadata.get('page_number')
            if page_num:
                current_section_page_numbers.add(page_num)

            section_match = self.BOOK_SECTION_TITLE_PATTERN.match(content)
            exercise_match = self.EXERCISE_START_PATTERN.match(content)
            solution_block_match = self.SOLUTION_BLOCK_START_PATTERN.match(content)
            specific_solution_match = self.SPECIFIC_SOLUTION_PATTERN.match(content)

            # 1. Check for new section/chapter start
            if section_match and element_type in ['Title', 'CompositeElement']:
                if current_section_elements or section_exercises_map or general_section_solutions:
                    final_chunks.extend(self._finalize_book_section(
                        current_section_elements, 
                        current_section_title, 
                        section_exercises_map, 
                        general_section_solutions,
                        sorted(list(current_section_page_numbers)),
                        metadata.get('filename') 
                    ))
                
                # Start a new section block
                current_section_elements = [element] 
                current_section_title = section_match.group(0).strip()
                current_section_page_numbers = {page_num} if page_num else set()
                section_exercises_map = {} # Reset exercises map for the new section
                general_section_solutions = [] # Reset general solutions for the new section

            # 2. Check for exercise within current section
            elif exercise_match:
                exercise_id = exercise_match.group(1).strip('.')
                if exercise_id not in section_exercises_map:
                    section_exercises_map[exercise_id] = {"question_elements": [], "solution_elements": []}
                section_exercises_map[exercise_id]["question_elements"].append(element)
                # Clear content_elements if a new exercise is detected to ensure it's not mixed with prior narrative
                current_section_elements = [] 

            # 3. Check for general solution block within current section (less common in books, but possible)
            elif solution_block_match:
                # If a general solution block appears within a section, add it to general solutions for that section
                general_section_solutions.append(element)
                current_section_elements = [] # Clear content_elements

            # 4. Check for specific solution within current section
            elif specific_solution_match:
                solution_target_id = specific_solution_match.group(1).strip('.')
                if solution_target_id not in section_exercises_map:
                    section_exercises_map[solution_target_id] = {"question_elements": [], "solution_elements": []}
                section_exercises_map[solution_target_id]["solution_elements"].append(element)
                current_section_elements = [] # Clear content_elements

            # 5. Regular content, add to current section's main narrative or to general solutions
            else:
                # If a general solution block was detected previously, subsequent elements belong to it
                if general_section_solutions: 
                    general_section_solutions.append(element)
                else:
                    current_section_elements.append(element)
        
        # Finalize the last section's chunks after the loop finishes
        if current_section_elements or section_exercises_map or general_section_solutions:
            final_chunks.extend(self._finalize_book_section(
                current_section_elements, 
                current_section_title, 
                section_exercises_map, 
                general_section_solutions,
                sorted(list(current_section_page_numbers)),
                book_elements[0]['metadata'].get('filename') if book_elements else "unknown_book_file"
            ))
            
        return final_chunks

    def _finalize_book_section(self, 
                                 section_elements: List[Dict[str, Any]], 
                                 section_title: str, 
                                 section_exercises_map: Dict[str, Any], 
                                 general_section_solutions: List[Dict[str, Any]],
                                 page_numbers: List[int],
                                 source_file: str) -> List[Dict[str, Any]]:
        """
        Processes all accumulated elements for a book section and generates granular chunks.
        """
        section_chunks = []
        
        # 1. Chunk the main section title (always create one if present)
        if section_title:
            title_chunk_content = section_title
            # Find page number for the title element itself
            title_page_num = next((e['metadata']['page_number'] for e in section_elements if e['cleaned_content'].strip() == section_title), None)
            section_chunks.append(self._create_chunk_from_block(
                content=title_chunk_content,
                block={"exercise_id": None, "section_title": section_title, 
                       "metadata_accumulator": {"page_numbers": {title_page_num} if title_page_num else set(), 
                                                "element_types": set([e['metadata'].get('type', 'NarrativeText') for e in section_elements if e['cleaned_content'].strip() == section_title]), # Get type for title element
                                                "chunk_type": "book_section_title", 
                                                "is_solution": False}},
                source_file=source_file
            ))
        
        # 2. Chunk the main narrative text of the section (excluding the title itself)
        narrative_content = ""
        narrative_page_numbers = set()
        narrative_element_types = set()
        for elem in section_elements:
            if elem['cleaned_content'].strip() != section_title: 
                narrative_content += elem['cleaned_content'] + "\n"
                p_num = elem['metadata'].get('page_number')
                if p_num: narrative_page_numbers.add(p_num)
                narrative_element_types.add(elem['metadata'].get('type', 'NarrativeText'))
        
        if narrative_content.strip():
            section_chunks.append(self._create_chunk_from_block(
                content=narrative_content.strip(),
                block={"exercise_id": None, "section_title": section_title, 
                       "metadata_accumulator": {"page_numbers": narrative_page_numbers, 
                                                "element_types": narrative_element_types,
                                                "chunk_type": "book_section_narrative", 
                                                "is_solution": False}},
                source_file=source_file
            ))

        # 3. Chunk exercises and their specific solutions within this section
        for exercise_id, data in section_exercises_map.items():
            exercise_content = ""
            current_exercise_page_numbers = set()
            exercise_element_types = set()

            for q_elem in data["question_elements"]:
                exercise_content += q_elem['cleaned_content'] + "\n"
                p_num = q_elem['metadata'].get('page_number')
                if p_num: current_exercise_page_numbers.add(p_num)
                exercise_element_types.add(q_elem['metadata'].get('type', 'NarrativeText'))
            
            if data["solution_elements"]:
                exercise_content += "\nSolución:\n"
                for s_elem in data["solution_elements"]:
                    exercise_content += s_elem['cleaned_content'] + "\n"
                    p_num = s_elem['metadata'].get('page_number')
                    if p_num: current_exercise_page_numbers.add(p_num)
                    exercise_element_types.add(s_elem['metadata'].get('type', 'NarrativeText'))
            
            if exercise_content.strip():
                section_chunks.append(self._create_chunk_from_block(
                    content=exercise_content.strip(),
                    block={"exercise_id": exercise_id, "section_title": section_title, 
                           "metadata_accumulator": {"page_numbers": current_exercise_page_numbers, 
                                                    "element_types": exercise_element_types,
                                                    "chunk_type": "book_exercise_with_solution", 
                                                    "is_solution": bool(data["solution_elements"])}},
                    source_file=source_file
                ))
        
        # 4. Chunk general solutions for the section (if any)
        if general_section_solutions:
            general_solution_content = ""
            general_solution_page_numbers = set()
            general_solution_element_types = set()
            for elem in general_section_solutions:
                general_solution_content += elem['cleaned_content'] + "\n"
                p_num = elem['metadata'].get('page_number')
                if p_num: general_solution_page_numbers.add(p_num)
                general_solution_element_types.add(elem['metadata'].get('type', 'NarrativeText'))

            if general_solution_content.strip():
                section_chunks.append(self._create_chunk_from_block(
                    content="Solución General de Sección:\n" + general_solution_content.strip(),
                    block={"exercise_id": None, "section_title": section_title, 
                           "metadata_accumulator": {"page_numbers": general_solution_page_numbers, 
                                                    "element_types": general_solution_element_types,
                                                    "chunk_type": "book_section_general_solution", 
                                                    "is_solution": True}},
                    source_file=source_file
                ))

        return section_chunks

    def _chunk_generic_document(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        A basic fallback chunker for documents that don't fit specific patterns.
        It attempts to create chunks based on page breaks or major element types (Title/Table).
        """
        chunks = []
        current_content = ""
        current_metadata_acc = self._create_empty_block()["metadata_accumulator"]
        last_page = None
        
        for i, element in enumerate(elements):
            content = element['cleaned_content']
            page_num = element['metadata'].get('page_number')
            filename = element['metadata'].get('filename')
            element_type = element['metadata'].get('type', 'NarrativeText') # CORRECCIÓN
            
            # Heuristic for new chunk: page change OR significant element type AND content accumulated
            if (page_num is not None and last_page is not None and page_num != last_page) or \
               (element_type in ['Title', 'Table'] and current_content.strip()):
                if current_content.strip(): # Finalize previous chunk
                    current_metadata_acc["chunk_type"] = "generic_chunk"
                    chunks.append(self._create_chunk_from_block(
                        content=current_content.strip(),
                        block={"metadata_accumulator": current_metadata_acc},
                        source_file=filename 
                    ))
                # Reset for new chunk
                current_content = ""
                current_metadata_acc = self._create_empty_block()["metadata_accumulator"]
            
            # Accumulate content and metadata for the current block
            current_content += content + "\n"
            if page_num: current_metadata_acc["page_numbers"].add(page_num)
            current_metadata_acc["element_types"].add(element_type)
            current_metadata_acc["source_file"] = filename
            last_page = page_num
        
        # Add the very last chunk if there's any remaining content
        if current_content.strip():
            current_metadata_acc["chunk_type"] = "generic_chunk"
            chunks.append(self._create_chunk_from_block(
                content=current_content.strip(),
                block={"metadata_accumulator": current_metadata_acc},
                source_file=elements[-1]['metadata'].get('filename') if elements else "unknown_file"
            ))
        return chunks

# Código de prueba
if __name__ == "__main__":
    from  data_preparation.text_cleaner import TextCleaner
    import os
    import sys
    import json

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.append(os.path.join(project_root, 'src'))

    raw_elements_file_path = os.path.join(project_root, '.data', 'all_raw_extracted_elements.json') # O 'all_raw_extracted_elements.json'
    
    sample_raw_elements = []
    if os.path.exists(raw_elements_file_path):
        with open(raw_elements_file_path, 'r', encoding='utf-8') as f:
            sample_raw_elements = json.load(f)
        print(f"Cargados {len(sample_raw_elements)} elementos crudos desde {raw_elements_file_path}")
    else:
        print(f"Error: No se encontró el archivo de elementos crudos en {raw_elements_file_path}")
        print("Creando elementos de prueba manuales para demostración...")
        sample_raw_elements = [
            {
                "content": "EXAMEN DE INGRESO A LA EDUCACIÓN SUPERIOR\n\nCurso 2017-2018 1ra convocatoria",
                "metadata": {"source": "exam.pdf", "page_number": 1, "type": "Title", "filename": "exam_1.pdf"}
            },
            {
                "content": "1. Lee detenidamente y responde.\n\n1.1. Clasifica las siguientes proposiciones en verdaderas (V) o falsas (F). Escribe V o F en la línea dada.",
                "metadata": {"source": "exam.pdf", "page_number": 1, "type": "NarrativeText", "filename": "exam_1.pdf"}
            },
            {
                "content": "2. En la figura se muestra una circunferencia de centro O y diámetro AB.",
                "metadata": {"source": "exam.pdf", "page_number": 2, "type": "NarrativeText", "filename": "exam_1.pdf"}
            },
            {
                "content": "Solución :",
                "metadata": {"source": "exam.pdf", "page_number": 3, "type": "Title", "filename": "exam_1.pdf"}
            },
            {
                "content": "La solución al ejercicio 1.1 es V.",
                "metadata": {"source": "exam.pdf", "page_number": 3, "type": "NarrativeText", "filename": "exam_1.pdf"}
            },
            {
                "content": "Solucion Ejercicio 2:",
                "metadata": {"source": "exam.pdf", "page_number": 4, "type": "NarrativeText", "filename": "exam_1.pdf"}
            },
            {
                "content": "La solución al ejercicio 2 es complejo.",
                "metadata": {"source": "exam.pdf", "page_number": 4, "type": "NarrativeText", "filename": "exam_1.pdf"}
            },
            {
                "content": "Capítulo 1: Introducción a la Física",
                "metadata": {"source": "book.pdf", "page_number": 1, "type": "Title", "filename": "book_1.pdf"}
            },
            {
                "content": "La física es la ciencia natural que estudia las propiedades...",
                "metadata": {"source": "book.pdf", "page_number": 1, "type": "NarrativeText", "filename": "book_1.pdf"}
            },
            {
                "content": "Ejercicio 1.1: Calcular la velocidad.",
                "metadata": {"source": "book.pdf", "page_number": 2, "type": "NarrativeText", "filename": "book_1.pdf"}
            },
            {
                "content": "La solución al ejercicio 1.1 es simple.",
                "metadata": {"source": "book.pdf", "page_number": 2, "type": "NarrativeText", "filename": "book_1.pdf"}
            },
            {
                "content": "Capítulo 2: Mecánica Clásica",
                "metadata": {"source": "book.pdf", "page_number": 10, "type": "Title", "filename": "book_1.pdf"}
            },
            {
                "content": "| Item | Descripción |\n|---|---|\n| Masa | 10 kg |",
                "metadata": {"source": "book.pdf", "page_number": 11, "type": "Table", "filename": "book_1.pdf"}
            },
            {
                "content": "Aquí más narrativa sobre mecánica.",
                "metadata": {"source": "book.pdf", "page_number": 11, "type": "NarrativeText", "filename": "book_1.pdf"}
            },
            {
                "content": "Texto genérico sin estructura aparente.",
                "metadata": {"source": "source.txt", "page_number": 1, "type": "NarrativeText", "filename": "generic.txt"}
            },
            {
                "content": "Un segundo párrafo en la misma página.",
                "metadata": {"source": "source.txt", "page_number": 1, "type": "NarrativeText", "filename": "generic.txt"}
            },
            {
                "content": "Otra página con contenido.",
                "metadata": {"source": "source.txt", "page_number": 2, "type": "NarrativeText", "filename": "generic.txt"}
            },
        ]
        # Asegúrate de limpiar los elementos si los estás creando manualmente y no vienen de DocumentLoader
        for elem in sample_raw_elements:
            if 'cleaned_content' not in elem:
                elem['cleaned_content'] = elem['content'] # Dummy clean if not coming from TextCleaner

    cleaner = TextCleaner()
    cleaned_elements = cleaner.clean_documents(sample_raw_elements, apply_lemmatization=False)
    print(f"Se limpiaron {len(cleaned_elements)} elementos de prueba.")
    processed_dir = os.path.join(project_root, '.data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, 'cleaned_elements.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_elements, f, ensure_ascii=False, indent=2)
    print(f"Chunks limpios guardados en {output_path}")
    chunker = SemanticChunker()

    print("\n--- Chunking de Documentos---")
    exam_chunks = chunker.chunk_documents(cleaned_elements)
    for i, chunk in enumerate(exam_chunks):
        print(f"\nCHUNK {i+1} (Tipo: {chunk['metadata'].get('chunk_type', 'N/A')}, Ejercicio: {chunk['metadata'].get('exercise_id', 'N/A')}):")
        print(f"Páginas: {chunk['metadata'].get('page_numbers', 'N/A')}")
        print(f"Es Solución: {chunk['metadata'].get('is_solution', 'N/A')}")
        print(chunk['content'])
        print("-" * 30)
    print(f"Antes del chunk {len(cleaned_elements)} después del chunk {len(exam_chunks)}")
    processed_dir = os.path.join(project_root, '.data', 'processed','chunks')
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, 'exam_chunks.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(exam_chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks guardados en {output_path}")