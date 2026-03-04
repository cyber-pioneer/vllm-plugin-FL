from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8020/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
                )
response = client.chat.completions.create(
        model="qwen3.5",
        messages=[
            {"role": "user", "content": "Give me a short introduction to large language models."},
            ],
        max_tokens=512,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            },
        stream=True,
        )
for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
