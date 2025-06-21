import os
from mistralai import Mistral
from typing import List, Dict, Generator, Optional, Union

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

        if not self.api_key:
            raise ValueError(
                "La clave API para Mistral no fue proporcionada ni encontrada "
                "en las variables de entorno (MISTRAL_API_KEY)."
            )

        self.client = Mistral(api_key=self.api_key)
        print(f"MistralLLMProvider inicializado para modelo por defecto: {self.default_model}")

    def chat_completion(self,
                        messages: List[Dict[str, str]],
                        model: Optional[str] = None,
                        temperature: float = 0.7,
                        max_tokens: int = 1000,
                        stream: bool = False) -> Union[Generator[str, None, None], str]:
        """
        Realiza una llamada al chat completion de Mistral AI.
        Puede ser en streaming (Generator) o no (str).

        Args:
            messages (List[Dict[str, str]]): Lista de mensajes de chat en formato diccionario.
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
