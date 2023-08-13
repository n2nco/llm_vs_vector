import openai
import spacy
import os
import time
import requests
from scipy.spatial import distance
import tiktoken

TOKEN_COST_GPT_3_5 = 0.002
TOKEN_COST_ADA = 0.0001

def num_tokens_from_string(string: str, encoding_name: str) -> int:
  encoding = tiktoken.get_encoding(encoding_name)
  return len(encoding.encode(string))

nlp = spacy.load("en_core_web_md")

swap_intent_base = "Swap intent: convert, exchange, trade, flip, transmute, interchange, replace"
send_intent_base = "Send intent: transfer, send to, move, route, forward, pass, pay, remit"

positive_embedding_spacy = nlp(swap_intent_base).vector
negative_embedding_spacy = nlp(send_intent_base).vector

openai.api_key = 'YOUR_API_KEY'

def get_ada_embedding(text):
  headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai.api_key}"}
  data = {"input": text, "model": "text-embedding-ada-002"}
  response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
  return response.json()["data"][0]["embedding"]

swap_embedding_ada = get_ada_embedding(swap_intent_base)
send_embedding_ada = get_ada_embedding(send_intent_base)

sentences = [
  "Put my balance into eth",
  "send this to my friend",
  "i want btc out of my eth",  # Missing comma here
  "transfer 10 eth to prala",
  "let's go all in on eth",
  "i want to buy eth",
  "put it all in eth",
  "give it to langwallet.eth",
  "send 10 eth to langwallet.eth",
  "pass it to langwallet.eth",
]

results = []

for sentence in sentences:
  # Classification using OpenAI and spaCy goes here
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
        "role": "system",
        "content": "You are a helpful assistant that only responds as 'swap' or 'send'."
        }, {
        "role": "user",
        "content": f"Is the following sentence swap or send intent: (options:swap/send)?\n\n{sentence}"
        }])

    davinci_duration = time.time() - start_time
    davinci_classification = response.choices[0].message['content'].strip().lower()

    start_time = time.time()
    sentence_embedding_ada = get_ada_embedding(sentence)
    swap_distance_ada = distance.cosine(sentence_embedding_ada, swap_embedding_ada)
    send_distance_ada = distance.cosine(sentence_embedding_ada, send_embedding_ada)
    embedding_classification_ada = "swap" if swap_distance_ada < send_distance_ada else "send"
    embedding_duration_ada = time.time() - start_time

    # Token calculations
    tokens_gpt_3_5 = num_tokens_from_string(sentence, "cl100k_base")
    tokens_ada = tokens_gpt_3_5  # Assuming the same tokenization method
    cost_gpt_3_5 = tokens_gpt_3_5 * TOKEN_COST_GPT_3_5
    cost_ada = tokens_ada * TOKEN_COST_ADA

    results.append(
        (davinci_classification, davinci_duration, tokens_gpt_3_5, cost_gpt_3_5,
        embedding_classification_ada, swap_distance_ada, send_distance_ada, embedding_duration_ada, tokens_ada, cost_ada)
    )





# Define some ANSI escape codes for colors
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'


# Print results
header = "| Sentence | gpt-3.5 Class. | gpt-3.5 Time | gpt-3.5 Tokens | gpt-3.5 Cost | ada-002 Class. | ada Dist. Swap | ada Dist. Send | ada-002 Time | ada-002 Tokens | ada-002 Cost |"
separator = "|----------|---------------|--------------|---------------|--------------|----------------|----------------|----------------|--------------|---------------|--------------|"

print(header)
print(separator)
for i, (davinci_classification, davinci_duration, davinci_tokens, cost_gpt_3_5, embedding_classification_ada, swap_distance_ada, send_distance_ada, embedding_duration_ada, ada_tokens, cost_ada) in enumerate(results):
    print(
        f"| {sentences[i][:30]}.. | {davinci_classification} | {davinci_duration:.2f} sec | {davinci_tokens} tokens | ${cost_gpt_3_5:.2f} | {embedding_classification_ada} | {swap_distance_ada:.3f} | {send_distance_ada:.3f} | {embedding_duration_ada:.2f} sec | {ada_tokens} tokens | ${cost_ada:.2f} |"
    )


avg_davinci_time = sum([res[1] for res in results]) / len(results)
avg_ada_time = sum([res[5] for res in results]) / len(results)
avg_gpt_3_5_cost = sum([res[3] for res in results]) / len(results)
avg_ada_cost = sum([res[7] for res in results]) / len(results)

# Colorize average results:
print(f"\n{Colors.GREEN}Average gpt-3.5 Time:{Colors.END}", avg_davinci_time)
print(f"{Colors.GREEN}Average gpt-3.5 Cost:{Colors.END} $", avg_gpt_3_5_cost)
print(f"{Colors.BLUE}Average ada-002 Time:{Colors.END}", avg_ada_time)
print(f"{Colors.BLUE}Average ada-002 Cost:{Colors.END} $", avg_ada_cost)
