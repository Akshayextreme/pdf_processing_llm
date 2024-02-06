import json
import sys
import os
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import torch
import re
from unstructured.partition.pdf import partition_pdf
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import logging
logging.basicConfig(level=logging.INFO)

from llm_chains import answer_chain, application_chain


def multi_mapp_res(pdf_data, que):
    ans_idx = []
    for idx, mapper in enumerate(que['multiple_mapping']):
        if pdf_data[mapper]['/V'] == "/Yes":
            ans_idx.append(idx)

    if len(ans_idx) == 0:
        return "Information not provided"
    if len(ans_idx) == 1:
        return que['choices'][ans_idx[0]]
    else:
        return ", ".join(list(que['choices'][i] for i in ans_idx))    


def parse_limit_pdf(limit_pdf):
    reader = PdfReader(limit_pdf)
    fields = reader.get_fields()

    with open("mapping.json", 'r') as fp:
        limit_mapper = json.load(fp)

    for que in limit_mapper['questions']:
        if que['mapping'] == 'multiple_mapping':
            answer = multi_mapp_res(fields, que)
        else:
            answer = fields[que['mapping']]['/V']
            if fields[que['mapping']]["/FT"] == "/Ch":
                multi_choices = fields[que['mapping']]["/Opt"]
                choices = list(i[0] for i in multi_choices)
                que['choices'] = choices
        que['answer'] = answer

    with open("limit.json", "w") as fp:
        json.dump(limit_mapper, fp)
        logging.info(f"Saving parsed limit application file at {os.path.join(os.getcwd(), 'limit.json')}")

    return limit_mapper


def parse_other_pdf(other_pdf):
    elements = partition_pdf(other_pdf, include_page_breaks=True)
    app_data = "\n\n".join([str(el) for el in elements]).split("\n\n\n\n")
    logging.info(f"Total pages in other pdf : {len(app_data)}")
    json_respsonse = []
    for p_idx, page in enumerate(tqdm(app_data)):
        try:
            # send page to llm for extraction
            response = application_chain.invoke({
                "application" : page})
            json_respsonse.extend(response.get('questions', None))
        except Exception as e:
            logging.exception(f"Exception in parsing application: {e}\n Page: {page}\n LLM response: {response}\n")

        # if p_idx > 1:
        #     break

    json_respsonse = {"application" : json_respsonse}
    other_pdf_json = json_respsonse

    with open("other_pdf.json", "w") as fp:
        json.dump(other_pdf_json, fp)
        logging.info(f"Saving parsed other application file at {os.path.join(os.getcwd(), 'other_pdf.json')}")

    return other_pdf_json


def get_similar_questions(query, corpus):
    embedder = SentenceTransformer("all-mpnet-base-v2")
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(corpus))

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=False)
    query_embedding = embedder.encode(query, convert_to_tensor=True, show_progress_bar=False)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    return top_results


def context_generation(limit_data, other_data, query_index, top_results):
    # logging.info("\n\n======================\n\n")
    query = other_data['application'][query_index].get('question')
    # logging.info("Query:", query)
    if (other_data['application'][query_index].get('choices', None)):
        choices = other_data['application'][query_index]['choices']
    else:
        choices = "[]"
    # logging.info("Choices:", choices)
    # logging.info("\nTop 5 most similar sentences in corpus:")

    context = ""
    for score, idx in zip(top_results[0], top_results[1]):
        context += f"Question: {limit_data['questions'][idx]['question']} (Score: {score:.4f})" + "\n" + f"Choices: {limit_data['questions'][idx]['choices']}" + "\n" + f"User provide answer: {limit_data['questions'][idx]['answer']}" + "\n\n"
    # logging.info(context)
    return query, choices, context


def run(limit_data, other_data):
    corpus = [que['question'] for que in limit_data['questions']]
    corpus[0] = "Name of Organization (Applicant)"

    def clean_query(query):
        query = re.sub(r"(\d+. )", "", query)
        query = re.sub(r"(\d+.\d )", "", query)
        return query

    queries = [clean_query(que['question']) for que in other_data['application']]
    logging.info(f"Total questions in other pdf: {len(queries)}")

    answers = []
    for q_idx, query in enumerate(tqdm(queries)):
        top_results = get_similar_questions(query, corpus)
        _, choices, context = context_generation(limit_data, other_data, q_idx, top_results)
        try:
            llm_resp = answer_chain.invoke({
                "query" : query,
                "choices" : choices,
                "context" : context})
            
            answers.append(
                f"{query}\n{choices}\n{llm_resp.dict().get('answer', None)}"
            )
            # logging.info(f"Generated answer: {llm_resp.dict().get('answer', None)}")
        except Exception as e:
            logging.exception(f"Exception in generating answer: {e}\n LLM response: {llm_resp}\n")

        # if q_idx > 1:
        #     break
    return answers


if __name__ == "__main__":
    print ('argument list', sys.argv)
    limit_pdf = sys.argv[1]
    other_pdf = sys.argv[2]

    logging.info(f"Parsing {limit_pdf}")
    limit_data = parse_limit_pdf(limit_pdf)

    logging.info(f"Parsing {other_pdf}")
    other_data = parse_other_pdf(other_pdf)

    logging.info("Generating answers")
    answers = run(limit_data, other_data)

    with open('answers.txt', 'w') as f:
        f.write('\n\n'.join(answers))
        logging.info(f"Saving generating answers at {os.path.join(os.getcwd(), 'answers.txt')}")
