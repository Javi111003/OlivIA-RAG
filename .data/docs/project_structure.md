## Estructura inicial con aspectos opcionales para OlivIA 🎯📈💭

intelligent-tutor-rag/
├── .gitignore
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml  (Opcional, para Poetry/Rye/PDM, mejora la gestión de dependencias)
├── .env.example
├── main.py             # Punto de entrada principal para iniciar el tutor
├── config.py           # Configuraciones globales (paths, claves de API, etc.)
├── data/
│   ├── raw/
│   │   ├── matematica/  # PDFs, DOCX, etc. originales
│   │   ├── espanol_literatura/
│   │   └── historia/
│   ├── processed/
│   │   ├── chunks/      # Chunks de texto procesados (JSON, Parquet, etc.)
│   │   ├── embeddings/  # Embeddings serializados (Numpy, Faiss index dumps)
│   │   └── kg_data/     # Datos para poblar el KG (CSV, JSON, Cypher scripts)
│   └── docs/            # Documentación adicional del proyecto, reportes
│       ├── system_design.md
│       └── data_sources.md
├── src/
│   ├── __init__.py      # Marca 'src' como un paquete Python
│   ├── data_preparation/
│   │   ├── __init__.py
│   │   ├── document_loader.py    # Carga de PDFs, DOCX, HTML, etc. (incluye OCR)
│   │   ├── text_cleaner.py       # Limpieza y normalización de texto
│   │   ├── chunker.py            # Lógica de segmentación (chunking)
│   │   ├── kg_builder.py         # Extracción de entidades/relaciones y construcción del KG
│   │   └── pipeline.py           # Orquestación del pipeline de preparación de datos
│   ├── embedding_models/
│   │   ├── __init__.py
│   │   ├── embedding_manager.py  # Interfaz para modelos de embedding
│   │   └── fine_tuning.py        # Scripts para fine-tuning del embedding model
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── vector_db_manager.py  # Interfaz con la base de datos vectorial (ChromaDB/FAISS)
│   │   ├── query_processor.py    # Pre-procesamiento y reescritura de consultas
│   │   ├── hybrid_retriever.py   # Lógica de recuperación híbrida (densa + sparse)
│   │   ├── kg_retriever.py       # Lógica de consulta al Grafo de Conocimiento
│   │   └── re_ranker.py          # Módulo para re-ranking de chunks
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── llm_interface.py      # Interfaz para GPT-4o, Claude 3 Opus, Gemini 1.5 Pro
│   │   ├── prompt_builder.py     # Lógica para construir prompts efectivos
│   │   └── post_processor.py     # Limpieza y formato de la respuesta del LLM
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Clase base para agentes
│   │   ├── planning_agent.py     # Agente que planifica la estrategia de respuesta
│   │   ├── reflection_agent.py   # Agente que evalúa y refina la respuesta
│   │   ├── tool_manager.py       # Gestión de herramientas para los agentes
│   │   └── specialised_agents/   # Carpeta para agentes de tarea específica
│   │       ├── math_agent.py
│   │       ├── spanish_lit_agent.py
│   │       └── history_agent.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── conversation_manager.py # Manejo del historial de conversación y contexto
│   │   └── tutor_system.py         # Orquestación principal de la interacción del tutor
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_config.py       # Configuración de logs
│   │   ├── metrics.py              # Funciones para métricas de evaluación
│   │   └── error_handling.py       # Gestión de errores
│   └── web_crawler/              # Módulo para el crawler dinámico (si se integra)
│       ├── __init__.py
│       ├── crawler_manager.py
│       └── content_parser.py
├── tests/
│   ├── __init__.py
│   ├── unit/                     # Pruebas unitarias para funciones/clases individuales
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   └── test_llm_interface.py
│   ├── integration/              # Pruebas de integración entre módulos
│   │   └── test_rag_pipeline.py
│   ├── end_to_end/               # Pruebas de extremo a extremo del tutor
│   │   └── test_tutor_interaction.py
│   └── evaluation/               # Scripts y datos para la evaluación sistemática
│       ├── eval_script.py
│       └── evaluation_data.json  # Preguntas y respuestas esperadas para evaluación
└── notebooks/                  # Para experimentación, prototipado rápido, análisis de datos
    ├── data_exploration.ipynb
    ├── embedding_tuning.ipynb
    └── agent_prototyping.ipynb