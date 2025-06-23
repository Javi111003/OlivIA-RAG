import os
from mistralai import Mistral
from typing import List,  Generator, Optional, Union, Type
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser 
import json

class MistralLLMProvider:
    """
    Provee una instancia configurada de MistralClient y métodos de utilidad
    para interactuar con la API de Mistral AI.
    Cada instancia de esta clase representa un 'cliente LLM' configurable
    para una tarea específica (e.g., chat, expansión de consulta, toma de decisiones).
    """

    def __init__(self, api_key: Optional[str] = None,
                 default_model: str = "mistral-small-latest"):
        """
        Inicializa el proveedor del cliente Mistral.
        Si la API Key no se proporciona, intentará obtenerla de MISTRAL_API_KEY en variables de entorno.

        Args:
            api_key (Optional[str]): Clave API para Mistral AI.
            default_model (str): Nombre del modelo de Mistral a usar por defecto para esta instancia.
        """
        self.api_key = api_key if api_key else os.getenv("MISTRAL_API_KEY")
        self.default_model = default_model
        self._structured_output_schema: Optional[Type[BaseModel]] = None
        self._parser: Optional[JsonOutputParser] = None

        if not self.api_key:
            raise ValueError(
                "La clave API para Mistral no fue proporcionada ni encontrada "
                "en las variables de entorno (MISTRAL_API_KEY)."
            )

        self.client = Mistral(api_key=self.api_key)
        print(f"MistralLLMProvider inicializado para modelo por defecto: {self.default_model}")
        
    def with_structured_output(self, schema: Type[BaseModel]) -> 'MistralLLMProvider':
         """
         Configura el LLM para devolver output estructurado según el schema Pydantic
         
         Args:
             schema: Clase Pydantic que define la estructura esperada
             
         Returns:
             MistralLLMProvider: Nueva instancia configurada para structured output
         """
         # Crear una nueva instancia para no mutar la original
         new_provider = MistralLLMProvider(
             api_key=self.api_key,
             default_model=self.default_model
         )
         new_provider._structured_output_schema = schema
         new_provider._parser = JsonOutputParser(pydantic_object=schema)
         return new_provider
     
    async def ainvoke(self, messages: Union[List[dict[str, str]], str], **kwargs) -> Union[BaseModel, str]:
        """
        Invoca el modelo de forma asíncrona con soporte para structured output
        
        Args:
            messages: Mensajes a enviar al modelo o string único
            **kwargs: Argumentos adicionales para la llamada
            
        Returns:
            BaseModel o str: Respuesta estructurada si se configuró schema, sino string
        """
        # Convertir string a formato de mensajes si es necesario
        if isinstance(messages, str):
            formatted_messages = [{"role": "user", "content": messages}]
        else:
            formatted_messages = messages

        # Si hay schema configurado, agregar instrucciones de formato
        if self._structured_output_schema and self._parser:
            # Agregar instrucciones de formato al último mensaje
            last_message = formatted_messages[-1]["content"]
            format_instructions = self._parser.get_format_instructions()
            
            formatted_messages[-1]["content"] = f"{last_message}\n\n{format_instructions}"

        try:
            # Llamar al modelo (simulamos async con sync por ahora)
            response = self.chat_completion(
                messages=formatted_messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000),
                stream=False
            )

            # Si hay schema configurado, parsear la respuesta
            if self._structured_output_schema and self._parser:
                try:
                    # Intentar parsear como JSON estructurado
                    parsed_dict = self._parser.parse(response)
                    
                    if isinstance(parsed_dict, dict):
                        structured_response = self._structured_output_schema(**parsed_dict)
                        return structured_response
                    else:
                        return parsed_dict
                except Exception as parse_error:
                    print(f"Error parsing structured output: {parse_error}")
                    # Fallback: intentar extraer JSON de la respuesta
                    return self._extract_json_fallback(response)
            
            return response

        except Exception as e:
            print(f"Error en ainvoke: {e}")
            if self._structured_output_schema:
                # Retornar una instancia por defecto del schema
                return self._structured_output_schema()
            return ""
        
    def _extract_json_fallback(self, response: str) -> Union[BaseModel, str]:
        """
        Intenta extraer JSON de una respuesta de texto libre como fallback
        """
        try:
            # Buscar bloques JSON en la respuesta
            import re
            json_pattern = r'\{.*?\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    json_data = json.loads(match)
                    return self._structured_output_schema(**json_data)
                except:
                    continue
                    
            return self._structured_output_schema()
            
        except Exception as e:
            print(f"Error in JSON fallback extraction: {e}")
            return self._structured_output_schema()
    
    def chat_completion(self,
                        messages: List[dict[str, str]],
                        model: Optional[str] = None,
                        temperature: float = 0.7,
                        max_tokens: int = 1000,
                        stream: bool = False) -> Union[Generator[str, None, None], str]:
        """
        Realiza una llamada al chat completion de Mistral AI.
        Puede ser en streaming (Generator) o no (str).

        Args:
            messages (List[dict[str, str]]): Lista de mensajes de chat en formato diccionario.
                                             Ej: [{"role": "user", "content": "Hola!"}]
            model (Optional[str]): Nombre del modelo a usar para esta llamada. Si es None,
                                   usa el default_model de esta instancia.
            temperature (float): Controla la aleatoriedad (0.0 a 1.0).
            max_tokens (int): Límite de tokens en la respuesta.
            stream (bool): Si es True, devuelve un generador para streaming.

        Returns:
            Union[Generator[str, None, None], str]: Un generador para streaming o la cadena de respuesta completa.
        """
        actual_model = model if model else self.default_model
        mistral_api_messages = messages

        try:
            if stream:
                try:
                    response_stream = self.client.chat.stream(
                        model=actual_model,
                        messages=mistral_api_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    def token_generator():
                        for chunk in response_stream:
                            delta = chunk.data.choices[0].delta
                            if delta and getattr(delta, "content", None):
                                yield delta.content
                            else: 
                                print("Chunk sin contenido en delta.")
                                yield ""

                    return token_generator()
                except Exception as e:
                    print(f"Error en stream: {e}")
                    def empty_gen():
                        yield ""
                    return empty_gen()
            else:
                response = self.client.chat.complete(
                    model=actual_model,
                    messages=mistral_api_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error en la llamada a Mistral AI ({'streaming' if stream else 'normal'}) "
                  f"desde MistralLLMProvider para el modelo {actual_model}: {e}")
            if stream:
                def empty_gen():
                    yield ""
                return empty_gen()
            else:
                return ""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Método de compatibilidad para generar respuestas simples"""
        return self.ainvoke([{"role": "user", "content": prompt}], **kwargs)
