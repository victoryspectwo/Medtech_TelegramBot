import ollama

# Test if Ollama is working
response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is paracetamol used for?"}]
)

# Print the response
print(response)
