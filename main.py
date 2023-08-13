import openai
import spacy
import os
import time
import requests
from scipy.spatial import distance
import tiktoken

# Cost per token for the respective models (hypothetical costs, check OpenAI's pricing page)
TOKEN_COST_GPT_3_5 = 0.002
TOKEN_COST_ADA = 0.0001


def num_tokens_from_string(string: str, encoding_name: str) -> int:
  """Returns the number of tokens in a text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(string))
  return num_tokens


# Load spaCy's English NLP model for embeddings
nlp = spacy.load("en_core_web_md")


# positive_sentiment_base = "Positive:  Joyful Elated Ecstatic Content Jubilant Optimistic Serene Euphoric Radiant Thrilled Positive Bliss Elation Jubilation Serenity Triumph Delight Exuberance Harmony Reverie Zenith"
# negative_sentiment_base = "Negative:  Despondent Morose Disheartened Forlorn Melancholic Pessimistic Dismayed Frustrated Anguished Apprehensive Negative Despair Gloom Dismay Angst Malaise Turmoil Woe Heartbreak Affliction Abyss"

swap_intent_base = "Swap intent: convert, exchange, trade, flip, transmute, interchange, replace"
send_intent_base = "Send intent: transfer, send to, move, route, forward, pass, pay, remit"

# Embed the words "positive" and "negative" using spaCy
swap_embedding_spacy = nlp(swap_intent_base).vector
send_embedding_space = nlp(send_intent_base).vector
def spacy_similarity(sentence, base_text):
    return nlp(sentence).similarity(nlp(base_text))

# Set up the OpenAI API
openai.api_key = print('your key here')


# Function to get embeddings using OpenAI's API
def get_ada_embedding(text):
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}"
  }
  data = {"input": text, "model": "text-embedding-ada-002"}
  response = requests.post("https://api.openai.com/v1/embeddings",
                           headers=headers,
                           json=data)
  return response.json()["data"][0]["embedding"]




swap_embedding_ada = get_ada_embedding(swap_intent_base)
send_embedding_ada = get_ada_embedding(send_intent_base)


# Example sentences
sentences = [
  "Put my balance into eth",
  "send this to my friend",
  "i want btc out of my eth"
  "transfer 10 eth to prala",
  "let's go all in on eth",
  "i want to buy eth",
  "put it all in eth",
  "give it to langwallet.eth",
  "send 10 eth to langwallet.eth",
  "pass it to langwallet.eth",
]

results = []
spacy_results = []

for sentence in sentences:
    print(sentence)
    # Time measurement for GPT-3.5
    start_time_gpt = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant that only responds as 'swap' or 'send'."
        }, {
            "role": "user",
            "content": f"Is the following user intent to swap or send: (options:swap/send)?\n\n{sentence}"
        }])
    davinci_duration = time.time() - start_time_gpt
    davinci_classification = response.choices[0].message['content'].strip().lower()
    print('gpt says', davinci_classification)



    start_time_spacy = time.time()
    swap_similarity_spacy = spacy_similarity(sentence, swap_intent_base)
    send_similarity_spacy = spacy_similarity(sentence, send_intent_base)
    
    if swap_similarity_spacy > send_similarity_spacy:
        embedding_classification_spacy = "swap"
    else:
        embedding_classification_spacy = "send"
        
    spacy_duration = time.time() - start_time_spacy
    
    spacy_results.append((embedding_classification_spacy, spacy_duration))
    
    # Time measurement for ADA embeddings
    start_time_ada = time.time()
    sentence_embedding_ada = get_ada_embedding(sentence)
    swap_distance_ada = distance.cosine(sentence_embedding_ada, swap_embedding_ada)
    send_distance_ada = distance.cosine(sentence_embedding_ada, send_embedding_ada)
    if swap_distance_ada < send_distance_ada:
        embedding_classification_ada = "swap"
    else:
        embedding_classification_ada = "send"
    ada_duration = time.time() - start_time_ada

    tokens_gpt_3_5 = num_tokens_from_string(sentence, "cl100k_base")
    tokens_ada = num_tokens_from_string(sentence,"cl100k_base")  # Assuming the same tokenization method

    cost_gpt_3_5 = tokens_gpt_3_5 * TOKEN_COST_GPT_3_5
    cost_ada = tokens_ada * TOKEN_COST_ADA

    results.append((davinci_classification, davinci_duration, tokens_gpt_3_5, cost_gpt_3_5,
                    embedding_classification_ada, swap_distance_ada, send_distance_ada, ada_duration, tokens_ada, cost_ada))


# Print results
header = "| Sentence | gpt-3.5 Class. | gpt-3.5 Time | gpt-3.5 Tokens | gpt-3.5 Cost | ada-002 Class. | ada Dist. Swap | ada Dist. Send | ada-002 Time | ada-002 Tokens | ada-002 Cost |"
separator = "|----------|---------------|--------------|---------------|--------------|----------------|----------------|----------------|--------------|---------------|--------------|"
header += "| spaCy Class. | spaCy Time |"
separator += "|--------------|-----------|"
print(header)
print(separator)

print(header)
print(separator)
for i, (davinci_classification, davinci_duration, davinci_tokens, cost_gpt_3_5,
        embedding_classification_ada, swap_distance_ada, send_distance_ada, embedding_duration_ada, ada_tokens, cost_ada) in enumerate(results):
    
    ada_color = '\033[94m' if 'swap' in embedding_classification_ada else '\033[93m'

    spacy_classification, spacy_duration = spacy_results[i]
    
    print(
        f"| {sentences[i][:35]}.. | {davinci_classification} | {davinci_duration:.2f} sec | {davinci_tokens} tokens | ${cost_gpt_3_5:.2f} | {embedding_classification_ada} | {swap_distance_ada:.3f} | {send_distance_ada:.3f} | {embedding_duration_ada:.2f} sec | {ada_tokens} tokens | ${cost_ada:.2f} | {spacy_classification} | {spacy_duration:.2f} sec |"
    )
# Average results
avg_davinci_time = sum([res[1] for res in results]) / len(results)
avg_gpt_3_5_cost = sum([res[3] for res in results]) / len(results)
avg_ada_time = sum([res[7] for res in results]) / len(results)
avg_ada_cost = sum([res[9] for res in results]) / len(results)
avg_spacy_time = sum([res[1] for res in spacy_results]) / len(spacy_results) #cost = free


print("\nAverage gpt-3.5 Time:", avg_davinci_time)
print("Average gpt-3.5 Cost: $",avg_gpt_3_5_cost)
print("Average spaCy Time:", avg_spacy_time)
print("Average ada-002 Time:", avg_ada_time)
print("Average ada-002 Cost: $",avg_ada_cost)
