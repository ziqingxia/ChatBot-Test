
def test_openai_connection():
    from openai import OpenAI
    client = OpenAI()
    response = client.responses.create(
        model="gpt-4-1106-preview",
        input="Write a one-sentence bedtime story about a unicorn."
    )
    print(f"Connection Result: {response.output_text}")
    return response.output_text