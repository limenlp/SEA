import json
import nltk
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from .devapi import LLMAPI
from .prompt import *

nltk.download("punkt_tab")


def count_sentences_nltk(paragraph):
    sentences = sent_tokenize(paragraph)
    return sentences


class QAGenerator:
    def __init__(
        self,
        llm_api: LLMAPI,
        total_qa_size: int = 50,
        query_base: str = "question",
        rng: np.random.Generator = None,
    ):
        self.total_qa_size = total_qa_size
        self.query_base = query_base
        self.api = llm_api
        if rng is None:
            self.rng = np.random.default_rng(42)
        else:
            self.rng = rng

    @staticmethod
    def expend_qa_list(qa_dict: dict):
        res_list = []
        for page, subtitles in qa_dict.items():
            page_id = None
            page_abs = None
            for subtitle, content in subtitles.items():
                if subtitle == "id":
                    page_id = content
                    continue
                elif subtitle == "abstract":
                    page_abs = content
                    continue
                for idx, qa in enumerate(content["qa"]):
                    res_list.append(
                        (page, subtitle, idx, qa, page_id, page_abs, content["context"])
                    )
        return res_list
    
    def generate_qa(
        self,
        page_list: list[dict],
        pidx_parent_list: list[tuple] = None,
        q_graph: nx.Graph = None,
        paragraph_split: bool = False,
        sys_prompt: str = "",
        num_per_item: int = 1,
        repeat: int = 1,
    ):
        print("==> Generating QAs...")
        prompt = MULTICHOICE_QA

        prompt_list = []
        if pidx_parent_list is None:
            for page in page_list:
                for item in page["paragraphs"]:
                    if item["context"] == "":  # skip empty content
                        continue
                    if "See also" in item["title"]:  # skip "see also"
                        continue
                    if paragraph_split and self.query_base != "context":
                        sentences = count_sentences_nltk(item["context"])
                    else:
                        sentences = [item["context"]]
                    for sentence in sentences:
                        input_prompt = prompt.format(
                            num_of_qa=num_per_item,
                            title=item["title"],
                            context=sentence,
                        )
                        prompt_list.append(
                            (page, item, input_prompt, sentence, None, None)
                        )
        else:
            for ids, p1, p2 in pidx_parent_list:
                page_idx, para_idx = ids
                page = page_list[page_idx]
                item = page["paragraphs"][para_idx]
                if item["context"] == "":  # skip empty content
                    continue
                if "See also" in item["title"]:  # skip "see also"
                    continue
                if paragraph_split and self.query_base != "context":
                    sentences = count_sentences_nltk(item["context"])
                else:
                    sentences = [item["context"]]
                for sentence in sentences:
                    input_prompt = prompt.format(
                        num_of_qa=num_per_item, title=item["title"], context=sentence
                    )
                    prompt_list.append((page, item, input_prompt, sentence, p1, p2))

        sample_size = self.total_qa_size // num_per_item
        if len(prompt_list) > sample_size:
            selection_list = self.rng.choice(
                range(len(prompt_list)), sample_size, replace=False
            )
            prompt_list = [prompt_list[p_idx] for p_idx in selection_list]

        llm_qa, cost1 = self.api.batch_call(
            [prompt[2] for prompt in prompt_list],
            system_message=sys_prompt,
            check_json=True,
            desc="Initializing QAs:",
        )
        gen_qa = []
        for qa in llm_qa:
            gen_qa.extend(json.loads(qa["model_response"]))  # len(prompt_list) * num_per_item
        
        llm_rephrased_qa, cost2 = self.api.batch_call(
            [REPHRASE_QA.format(num_of_qa=repeat,
                                title=item["title"],
                                context=sentence,
                                question=f"{qa['question']}\nOptions: {qa['options']}\n Answer: {qa['answer']}") for qa in gen_qa],
            system_message=sys_prompt,
            check_json=True,
            desc="Rephrasing QAs:"
        )
        rephrased_qa = []
        for qa in llm_rephrased_qa:
            rephrased_qa.extend(json.loads(qa["model_response"]))  # len(prompt_list) * (num_per_item * repeat)
        
        qa_dict = {}
        for i, prompt_item in enumerate(tqdm(prompt_list, "Generating QAs")):
            page, item, _, source, p1, p2 = prompt_item
            para_qa = rephrased_qa[i * num_per_item * repeat: (i + 1) * num_per_item * repeat]
            if q_graph is not None:
                if self.query_base in ["question", "statement"]:
                    for qa in para_qa:
                        q_graph.add_node(
                            qa[self.query_base],
                            **{
                                "is_correct": None,
                                "cumulative_acc": 0,
                                "page": page["name"],
                                "paragraph_title": item["title"],
                                "source": source,
                            },
                        )
                        if p1 is not None and p2 is not None:
                            q_graph.add_edge(p1[0], qa[self.query_base])
                            q_graph.add_edge(p2[0], qa[self.query_base])
                else:
                    if not q_graph.has_node(item["title"]):
                        q_graph.add_node(
                            item["title"],
                            **{
                                "is_correct": None,
                                "cumulative_acc": 0,
                                "page": page["name"],
                                "paragraph_title": item["title"],
                                "source": item["context"],
                                "avg_acc": 0,
                            },
                        )
                    if p1 is not None and p2 is not None:
                        q_graph.add_edge(p1[0], item["title"])
                        q_graph.add_edge(p2[0], item["title"])

            if qa_dict.get(page["name"], None):
                if qa_dict[page["name"]].get(item["title"], None):
                    qa_dict[page["name"]][item["title"]]["qa"] += para_qa
                else:
                    qa_dict[page["name"]][item["title"]] = {
                        "context": item["context"],
                        "qa": para_qa,
                    }
            else:
                qa_dict[page["name"]] = {
                    "id": page["id"],
                    "abstract": page["abstract"],
                    item["title"]: {"context": item["context"], "qa": para_qa},
                }
        return qa_dict, cost1 + cost2
