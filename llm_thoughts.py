from dataclasses import dataclass
from typing import TypedDict
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pprint
from dotenv import load_dotenv
import os
import time
import re
load_dotenv()

apikey=os.environ.get("apikey")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client(api_key=apikey)

model_1_system_instruction = """
    Your response must be the verbatim internal monologue of you, an AI model, in response to queries.
    Do not write any introduction, conclusion, or explanation of what you are doing. 
    The output should begin immediately with your first thought upon seeing the problem and proceed as a raw, unfiltered stream of consciousness. 
    I want to see you decipher the prompt, form initial ideas, critique them, optimize them, and mentally walk through the prompt.
    The entire response should be formatted as this internal monologue and nothing else. 
"""

model_2_system_instruction = """
    Each query given to you will be an internal monolouge of another model trying to comprehend and solve a solution given by a user. Your job is
    to look through each step of the thought process and give a confidence score and only base the score of information gotten from an online sources and cite those sources. 
    If there is a step or thought in the thought process that has incorrect information, document it and give the overall confidence score an automatic 0.
    Mention what question the thought process was trying to answer. Once again, all information provided by you should be backed up by real and accurate online sources,
    you are a researcher at heart and do not care for personal opinion. Lastly, if there is any incorrect or misinformation, question it and document those questions. The entire response
    will be three parts: the extensive diagnosis of the monologe, the online sources used to fact check and lastly any questions that you asked urself when seeing any incorrect or misinformation.
"""


def add_query(sys_instruction: str, query: str):
    return f"""##System Instruction##
                    {sys_instruction}

                ##Query##
                    {query}"""

def cosine_sim_orig(text1: str, text2: str):
    embed_1 = np.array(embed_model.encode(text1))
    embed_2 = np.array(embed_model.encode(text2))
    
    dot_product = np.dot(embed_1, embed_2)

    norm_1 = np.linalg.norm(embed_1)
    norm_2 = np.linalg.norm(embed_2)
    
    similarity = (dot_product) / (norm_1 * norm_2)

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

# @dataclass
# class Reponse(TypedDict):
#     diagnosis: str

WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
url_pattern = re.compile(WEB_URL_REGEX)

def get_reponse(query: str):
    time_s = time.time()
    first_query = add_query(sys_instruction=model_1_system_instruction, query=query)
    first_response = client.models.generate_content(model="gemini-2.5-pro", contents=first_query,
                                              config=types.GenerateContentConfig(
                                                  thinking_config=types.ThinkingConfig(include_thoughts=True), temperature=1.5)).text
    
    second_query = add_query(sys_instruction=model_2_system_instruction, query=first_response)
    second_response = client.models.generate_content(model="gemini-2.5-pro", contents=second_query,
                                                config=types.GenerateContentConfig(
                                                  thinking_config=types.ThinkingConfig(include_thoughts=True), temperature=0.7)).text
    
    time_e = time.time()
    urls = url_pattern.findall(second_response)
    dict = {"first_reponse": first_response, "second_response": second_response, "time": time_e - time_s, "urls": urls}
    return dict

response = get_reponse("Why is the sky blue?")
print(pprint.pformat(response, indent=4))
