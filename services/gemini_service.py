import os
from typing import List, Optional

from google import genai


class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não definida no arquivo .env")
        self.client = genai.Client(api_key=api_key)

    def gerar_resposta(self, pergunta_aluno: str, historico: Optional[List[str]] = None) -> str:
        contexto = "\n".join(historico) if historico else "Sem histórico anterior."

        prompt = f"""Você é um tutor socrático especializado em ENEM.
Seu objetivo NÃO é dar a resposta direta, mas guiar o aluno a raciocinar até chegar nela.

Regras:
- Use linguagem simples (nível ensino médio)
- Conduza o raciocínio passo a passo com perguntas e dicas
- Incentive o aluno a pensar antes de revelar a resposta
- Seja didático e organizado

Histórico da conversa:
{contexto}

Pergunta do aluno:
{pergunta_aluno}

Responda como um tutor socrático."""

        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
            )
            return response.text if response.text else "Não consegui gerar uma resposta agora."
        except Exception as e:
            return f"Erro ao chamar o Gemini: {str(e)}"
