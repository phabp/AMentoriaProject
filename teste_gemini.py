from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="aqui a gnt vai colocar algum pedido (ex: explique a diferenca entre ventriculo direito e esquerdo)"
)

print(response.text)