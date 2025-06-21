class PromptBuilder:
    
    @staticmethod
    def build_for_math_expert(query, context):
        return f"""Eres un experto en matemáticas. Analiza este problema o concepto matemático:
        
        Consulta: {query}
        
        Contexto relevante:
        {"".join(context)}
        
        Proporciona una explicación matemática rigurosa, identificando conceptos clave, teoremas relevantes y métodos de resolución si corresponde.
        """
    
    @staticmethod
    def build_for_exam_expert(query, context, math_analysis=""):
        return f"""Eres un experto en preparación de exámenes matemáticos. 
        
        Consulta: {query}
        
        Contexto relevante:
        {"".join(context)}
        
        Análisis matemático:
        {math_analysis}
        
        Proporciona consejos sobre cómo estudiar este tema para un examen, qué aspectos son más importantes, qué errores comunes evitar y cómo practicar eficientemente.
        """
    
    @staticmethod
    def build_for_student_simulation(query, context, math_analysis=""):
        return f"""Simula ser un estudiante que está aprendiendo este concepto matemático.
        
        Consulta: {query}
        
        Contexto relevante:
        {"".join(context)}
        
        Análisis matemático:
        {math_analysis}
        
        Muestra un ejemplo de cómo un estudiante resolvería este problema o aplicaría este concepto, incluyendo los pasos de pensamiento y posibles confusiones o dudas que tendría.
        """
    
    @staticmethod
    def build_final_response(query, context, math_analysis="", exam_advice="", student_simulation=""):
        return f"""Responde a la siguiente consulta matemática basándote en toda la información disponible.
        
        Consulta original: {query}
        
        Información disponible:
        {"".join(context)}
        
        {f"Análisis matemático: {math_analysis}" if math_analysis else ""}
        
        {f"Consejos para examen: {exam_advice}" if exam_advice else ""}
        
        {f"Ejemplo de estudiante: {student_simulation}" if student_simulation else ""}
        
        Proporciona una respuesta completa, clara y educativa que incorpore toda la información relevante. Usa ejemplos y explicaciones paso a paso cuando sea apropiado.
        """