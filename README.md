# OlivIA: Tu Tutor Inteligente para los Exámenes de Ingreso a la Universidad Cubana

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## ¡Bienvenido a OlivIA! 🤖👩‍🏫

**OlivIA** es un proyecto innovador de tutor inteligente diseñado para transformar la preparación de los estudiantes cubanos para los exámenes de ingreso a la universidad. Utilizando una avanzada arquitectura de **Generación Aumentada por Recuperación (RAG)** y un **Grafo de Conocimiento (KG)**, OlivIA ofrece asistencia pedagógica personalizada en las asignaturas clave de **Matemática, Español-Literatura e Historia**.

Nuestro objetivo es proporcionar una herramienta interactiva y eficiente que complemente el estudio tradicional, ofreciendo explicaciones claras, resolviendo dudas, y guiando en la práctica de ejercicios con un enfoque adaptado al currículo educativo cubano.

---

## Características Principales

* **Dominio Académico Específico**: Especializado en los contenidos y el formato de los exámenes de ingreso a la universidad cubana en:
    * **Matemática**: Álgebra, Geometría, Trigonometría, Lógica y pre-Cálculo.
    * **Español-Literatura**: Gramática, Ortografía, Redacción, Análisis de Textos y Obras Literarias cubanas y universales.
    * **Historia**: Historia de Cuba e Historia Universal relevante para el currículo.
* **Respuestas Inteligentes y Contextuales**: No solo busca información, sino que razona sobre ella para generar explicaciones didácticas y coherentes.
* **Resolución Guiada de Ejercicios**: Asiste a los estudiantes paso a paso en la solución de problemas y ejercicios, ofreciendo retroalimentación y aclaraciones.
* **Interacción Conversacional**: Mantiene el contexto de la conversación para un diálogo fluido y natural, adaptándose al ritmo de aprendizaje del estudiante.
* **Base de Conocimiento Curada**: Construida a partir de libros de texto oficiales, programas de estudio y exámenes de ingreso anteriores del sistema educativo cubano.
* **Arquitectura RAG Avanzada**: Utiliza técnicas de recuperación híbrida, re-ranking y un Grafo de Conocimiento para asegurar la máxima precisión y relevancia en la información.
* **Agentic Supervisor**: Implementa un sistema de agentes inteligentes para la planificación, ejecución y reflexión sobre las interacciones, simulando el comportamiento de un tutor humano.

---

## Pila Tecnológica

OlivIA está construido predominantemente en **Python** y utiliza las siguientes tecnologías clave:

* **Lenguaje de Programación**: `Python`
* **Modelos de Lenguaje (LLMs)**: Flexibilidad para integrar modelos de vanguardia como `GPT-4o`, `Claude 3 Opus` o `Gemini 1.5 Pro` (vía API), o modelos de código abierto optimizados (`Llama 3`, `Mistral`/`Mixtral`) ajustados (`fine-tuned` con LoRA) para despliegue local.
* **Frameworks RAG y NLP**: `LangChain`, `sentence-transformers`, `SpaCy`, `NLTK`.
* **Extracción de Documentos**: `PyPDF2`, `python-docx`, `BeautifulSoup4`, `Pytesseract` (para OCR).
* **Base de Datos Vectorial**: `ChromaDB` (para prototipado y despliegues locales) o `FAISS` + `pgvector` (para soluciones más escalables).
* **Base de Datos de Grafo (Knowledge Graph)**: `Neo4j` para almacenar y consultar el conocimiento estructurado.
* **Gestión de Dependencias**: `pip` y `requirements.txt` (o `Poetry`/`Rye`/`PDM` con `pyproject.toml`).

---

## Estructura del Proyecto

El repositorio está organizado de forma modular para facilitar el desarrollo y mantenimiento:
intelligent-tutor-rag/
├── .gitignore               # Archivos y directorios a ignorar por Git

├── README.md                # Este archivo

├── LICENSE                  # Licencia del proyecto

├── requirements.txt         # Dependencias de Python

├── pyproject.toml           # (Opcional) Para Poetry/Rye/PDM

├── .env.example             # Ejemplo de variables de entorno

├── main.py                  # Punto de entrada principal

├── config.py                # Configuraciones globales

├── data/

│   ├── raw/                 # Documentos originales (PDFs, DOCX, HTML)

│   └── processed/           # Datos procesados: chunks, embeddings, datos del KG

├── src/                     # Código fuente principal

│   ├── data_preparation/    # Ingesta, limpieza, chunking, construcción del KG

│   ├── embedding_models/    # Modelos de embedding y fine-tuning

│   ├── retriever/           # Módulo de recuperación de información (textos y KG)

│   ├── generator/           # Interfaz con LLM y generación de respuestas

│   ├── agents/              # Implementación de la arquitectura de agentes (Supervisor)

│   ├── core/                # Lógica central del sistema tutor

│   ├── utils/               # Utilidades generales (logging, métricas)

│   └── web_crawler/         # Crawler web dinámico

├── tests/                   # Pruebas unitarias, de integración y end-to-end

---

## Configuración e Instalación

Para configurar y ejecutar OlivIA en tu entorno local, sigue estos pasos:

### 1. Requisitos Previos

* **Python 3.9+**: Asegúrate de tener una versión compatible de Python instalada.
* **Tesseract OCR (Opcional, para PDFs escaneados)**: Si planeas usar PDFs escaneados en tu base de conocimiento, necesitarás instalar Tesseract en tu sistema y configurar su ruta en `config.py`.

### 2. Clonar el Repositorio

```bash
git clone [https://github.com/tu-usuario/intelligent-tutor-rag.git](https://github.com/tu-usuario/intelligent-tutor-rag.git)
cd intelligent-tutor-rag
