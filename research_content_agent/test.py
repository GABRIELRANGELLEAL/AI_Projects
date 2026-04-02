import os
from openai import OpenAI
from dotenv import load_dotenv

# Carrega a chave do seu arquivo .env
load_dotenv()

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("Enviando requisição de teste para a OpenAI...")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # Usando um modelo barato/rápido para teste
        messages=[
            {"role": "user", "content": "Diga apenas 'Olá, sistema online!' e nada mais."}
        ],
        max_tokens=10
    )
    
    print("\n✅ SUCESSO! A chave está funcionando.")
    print("Resposta da IA:", response.choices[0].message.content)

except Exception as e:
    print("\n❌ ERRO NA CHAVE OU NA CONTA:")
    print(e)