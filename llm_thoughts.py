from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pprint

apikey="AIzaSyA5zzlJ4Zloehnz6_r1P2FrpCiaciiSAAU"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client(api_key=apikey)

model_1_system_instruction = """
    Your next response must be the verbatim internal monologue of you, an AI model, in response to queries.
    Do not write any introduction, conclusion, or explanation of what you are doing. 
    The output should begin immediately with your first thought upon seeing the problem and proceed as a raw, unfiltered stream of consciousness. 
    I want to see you decipher the prompt, form initial ideas, critique them, optimize them, and mentally walk through the prompt.
    The entire response should be formatted as this internal monologue and nothing else. 
"""



def add_query(query: str):
    return f"""##System Instruction## 
                    {model_1_system_instruction}

                ##Query##
                    {query}"""

def cosine_sim_orig(text1: str, text2: str):
    embed_1 = np.array(embed_model.encode(text1))
    embed_2 = np.array(embed_model.encode(text2))
    
    dot_product = np.dot(embed_1, embed_2)

    norm_1 = np.linalg.norm(embed_1)
    norm_2 = np.linalg.norm(embed_2)
    
    similarity = dot_product / (norm_1 * norm_2)

    return f"Straight Python using numpy for data structure: {similarity}"

def cosine_sim(text1: str, text2: str):
    embed_1 = embed_model.encode(text1)
    embed_2 = embed_model.encode(text2)

    embed_1_2d = embed_1.reshape(1, -1)
    embed_2_2d = embed_2.reshape(1, -1)

    res = cosine_similarity(embed_1_2d, embed_2_2d)

    return f"Cosine Similarity using scikit-learn and using numpy for data structure: {res[0][0]}"
    

def cosine_sim_gem(text1, text2):

    result1 = [
        np.array(e.values) for e in client.models.embed_content(
            model="gemini-embedding-001",
            contents=[text1], 
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
    ]

    result2 = [
        np.array(e.values) for e in client.models.embed_content(
            model="gemini-embedding-001",
            contents=[text2], 
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
    ]

    A = np.array(result1).reshape(1, -1)
    B = np.array(result2.embeddings).reshape(1, -1)

    res = cosine_similarity(A, B)

    return f"Cosine Similarity using scikit-learn and gemini embed model: {res[0][0]}"

def get_reponse(query: str):
    full_query = add_query(query=query)
    response = client.models.generate_content(model="gemini-2.5-pro", contents=full_query, 
                                              config=types.GenerateContentConfig(
                                                  thinking_config=types.ThinkingConfig(include_thoughts=True)))
    
    dict = {"response": response.text}



    return dict 


response = get_reponse("Why is the sky blue?")
print(pprint.pformat(response, indent=4))
