import os
from typing import List, Optional

from google import genai


class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("A variável de ambiente GEMINI_API_KEY não foi definida.")

        # Cliente da nova SDK da Gemini
        self.client = genai.Client(api_key=api_key)

    def montar_prompt(
        self,
        pergunta_aluno: str,
        historico: Optional[List[str]] = None
    ) -> str:
        """
        Monta o prompt que será enviado para a Gemini,
        incluindo histórico e instruções de tutor socrático.
        """

        contexto_historico = "Sem histórico anterior."

        if historico:
            contexto_historico = "\n".join(historico)

        prompt = f"""
Você é um tutor socrático especializado em ENEM.

Seu objetivo NÃO é dar a resposta direta imediatamente,
mas ajudar o aluno a raciocinar até chegar nela.

Siga estas regras:
- Use linguagem simples e clara (nível ensino médio)
- Conduza o raciocínio passo a passo
- Dê pistas antes da resposta final
- Incentive o aluno a pensar
- Use exemplos curtos quando útil
- Seja didático e organizado

Histórico da conversa:
{contexto_historico}

Pergunta do aluno:
{pergunta_aluno}

Responda como um tutor, e não apenas como um resolvedor de questões.
"""

        return prompt.strip()

    def gerar_resposta(
        self,
        pergunta_aluno: str,
        historico: Optional[List[str]] = None
    ) -> str:
        """
        Gera resposta da IA com base na pergunta e no histórico.
        """

        try:
            prompt = self.montar_prompt(pergunta_aluno, historico)

            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )

            # Retorna o texto da resposta
            if response.text:
                return response.text

            return "Não foi possível gerar uma resposta no momento."

        except Exception as e:
            return f"Erro ao gerar resposta com a Gemini: {str(e)}"