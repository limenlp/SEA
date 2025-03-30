import faiss
import glob
import copy
import json
import torch
import asyncio
import aiolimiter
import requests
import networkx as nx
import datasets
import numpy as np
from PIL import Image
from functools import partial
from typing import Tuple, Union
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from sentence_transformers import SentenceTransformer
from .wiki2md import (
    generate_markdown,
    format_markdown_content,
)

# Customize filter for ProVision-10M dataset
def comp_img_name(y: list, x):
    return x['data_path'][0].split("/")[-1] in y

def comp_qa(y: str, x):
    return y == x['question']

def wiki_login():
    with open("wiki.key", "r") as f:
        S = requests.Session()
        URL = "https://www.mediawiki.org/w/api.php"

        # Retrieve login token first
        PARAMS_0 = {
            "action": "query",
            "meta": "tokens",
            "type": "login",
            "format": "json",
        }

        R = S.get(url=URL, params=PARAMS_0)
        DATA = R.json()

        LOGIN_TOKEN = DATA["query"]["tokens"]["logintoken"]

        # print(LOGIN_TOKEN)

        # Send a post request to login. Using the main account for login is not
        # supported. Obtain credentials via Special:BotPasswords
        # (https://www.mediawiki.org/wiki/Special:BotPasswords) for lgname & lgpassword
        lines = f.readlines()
        PARAMS_1 = {
            "action": "login",
            "lgname": lines[0].strip(),
            "lgpassword": lines[1].strip(),
            "lgtoken": LOGIN_TOKEN,
            "format": "json",
        }

        R = S.post(URL, data=PARAMS_1)
        DATA = R.json()

        # print(DATA)
        assert DATA["login"]["result"] == "Success"

def top_k_indices_argpartition(a, k, axis=-1):
    """
    Return the indices of the top-k elements of `a` along a given axis,
    ordered from largest to smallest by their corresponding values.

    Parameters
    ----------
    a : np.ndarray
        Input array.
    k : int
        Number of top elements to retrieve.
    axis : int, optional
        The axis along which to find the top-k elements. Default is -1 (last axis).

    Returns
    -------
    top_k_sorted_indices : np.ndarray
        Indices of the top-k elements, sorted by descending value along `axis`.
        The shape is the same as `a` except along `axis`, where it is `k`.
    """
    k = min(k, a.shape[axis])
    partitioned = np.argpartition(a, -k, axis=axis)
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(-k, None)  # from (size-k) to end
    top_k_unsorted_indices = partitioned[tuple(idx)]
    top_k_unsorted_values = np.take_along_axis(a, top_k_unsorted_indices, axis=axis)
    sorter = np.argsort(-top_k_unsorted_values, axis=axis)
    top_k_sorted_indices = np.take_along_axis(top_k_unsorted_indices, sorter, axis=axis)

    return top_k_sorted_indices


class PageRetriever:
    data_per_file: int = 10000
    total_pages: int = 7066830
    vg_imgs: int = 108079

    def __init__(
        self,
        data_path: str,
        faiss_idx_path: str,
        requests_per_minute: int = 10,
        error_thres: float = 0,
        query_base: str = "statement",
        rng: np.random.Generator = None,
        text_model_name: str = None,
        img_model_name: str = None,
        device: str = "cuda:0",
        vision_dataset: Union[str, Tuple[str, str]] = None,
    ):
        if rng is None:
            self.rng = np.random.default_rng(42)
        else:
            self.rng = rng
        self.data_path = (
            data_path  # /linxindata/wiki/wikidata/processed_files/id_name_abs_url/*
        )
        if text_model_name is not None:
            self.tmodel = SentenceTransformer(
                text_model_name,
                trust_remote_code=True,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                },
                device=device,
            )
        if img_model_name is not None:
            self.imodel = SentenceTransformer(
                img_model_name,
                trust_remote_code=True,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                },
                device=device,
            )
            
        self.rng = rng
        self.idx_path = faiss_idx_path
        self.faiss_index = faiss.read_index(self.idx_path)
        self.file_list = glob.glob(f"{self.data_path}/*")
        self.requests_per_minute = requests_per_minute
        self.error_thres = error_thres
        self.query_base = query_base
        
        if vision_dataset is not None:
            if isinstance(vision_dataset, str):
                self.vision_dataset = load_dataset(vision_dataset)
            else:
                d_name, split = vision_dataset
                self.vision_dataset = load_dataset(d_name, split=split)

    async def _get_page_md(self, pages):
        limiter = aiolimiter.AsyncLimiter(self.requests_per_minute)
        async_md_list = [
            generate_markdown(page["id"], page["name"], limiter=limiter)
            for page in pages
        ]
        page_md_list = await tqdm_asyncio.gather(*async_md_list, desc="Retrieving")

        return page_md_list

    def _retrieve_paragraph_from_pages(
        self, pages: list[dict], para_cutoff: int = None, from_online: bool = False
    ):
        pages = copy.deepcopy(pages)
        if from_online:
            page_md_list = asyncio.run(self._get_page_md(pages))
        else:
            page_md_list = [
                page["wikitext"]
                for page in pages
            ]

        pop_pages = []
        for page_idx, page in enumerate(pages):
            page_md = page_md_list[page_idx]
            if page_md is None:
                pop_pages.append(page)
                continue
            item_list = format_markdown_content(page_md)
            if len(item_list) == 0:
                pop_pages.append(page)
                continue
            if para_cutoff:
                pages[page_idx]["paragraphs"] = item_list[:para_cutoff]
            else:
                pages[page_idx]["paragraphs"] = item_list
        for p in pop_pages:
            pages.remove(p)
        return pages

    def _read_page_from_file(self, file_ids: np.ndarray):
        total_page_list = []
        for fid in file_ids:
            if fid == -1:
                continue
            with open(f"{self.data_path}/{fid}.jsonl", "r") as f:
                total_page_list += f.readlines()
        return total_page_list

    def page_random_sampling(
        self,
        page_size: int = 5,
        num_files: int = 3,
        mode: str = "text",  # vision, text
        qa_per_img: int = 3,
        img_size: int = 50,
    ) -> datasets.Dataset:
        if mode == "text":
            selected_file_id = self.rng.choice(
                len(self.file_list), num_files, replace=False
            )
            self.faiss_index.remove_ids(selected_file_id)
            page_list = self._read_page_from_file(selected_file_id)
            sp = self.rng.choice(page_list, page_size, replace=False).tolist()
            selected_pages = [
                json.loads(page) for page in list(map(lambda x: x.strip(), sp))
            ]
            selected_pages = self._retrieve_paragraph_from_pages(selected_pages)
            return selected_pages
        elif mode == "vision":
            selected_file_id = self.rng.choice(
                len(self.file_list), page_size, replace=False
            )
            self.faiss_index.remove_ids(selected_file_id)
            img_list = [self.file_list[iid].split("/")[-1] for iid in selected_file_id]
            qa_list = self.vision_dataset.filter(partial(comp_img_name, img_list), num_proc=24)
            
            final_set = []
            datasets.disable_progress_bars()
            for img in img_list:
                subset = qa_list.filter(partial(comp_img_name, img))
                qpi = len(subset)
                if qpi >= qa_per_img:
                    selected_qa = self.rng.choice(range(qpi), qa_per_img, replace=False)
                    final_set.append(subset.select(selected_qa))
                    if len(final_set) == img_size:
                        break
                else:
                    continue
            datasets.enable_progress_bars()
            return concatenate_datasets(final_set)

    def init_sampling(
        self,
        page_size: int = 5,
        init_topics: list[str] = None,
    ):
        query = [
            f"Title: {topic}"
            for topic in init_topics
        ]
        query_vec = self.tmodel.encode(
            query, convert_to_numpy=True, batch_size=16
        )
        faiss.normalize_L2(query_vec)
        _, page_indices = self.faiss_index.search(
            query_vec, page_size // len(init_topics)
        )  # distance, page_indices
        page_indices = page_indices.reshape(-1)
        self.faiss_index.remove_ids(page_indices)
        page_indices = [
            (idx // self.data_per_file, str(idx)[-4:])  # file_id, real_page_idx
            for idx in page_indices
        ]

        selected_pages = []
        for file_idx, real_page_idx in page_indices:
            try:
                selected_pages.append(
                    self._read_page_from_file([file_idx])[int(real_page_idx)]
                )
            except IndexError:
                print(f"IndexError: {file_idx}_{real_page_idx}")
        selected_pages = self._retrieve_paragraph_from_pages(
            [json.loads(page) for page in selected_pages]
        )
        return selected_pages

    def page_related_sampling(
        self,
        top_k_page: int = 5,
        paragraph_size: int = 3,
        q_graph: nx.DiGraph = None,
        random_mix_ratio: float = 0,
    ):
        print("==> Relation based sampling")
        if self.query_base in ["question", "statement"]:
            selected_query = [
                (x, y) for x, y in q_graph.nodes(data=True) if not y["is_correct"]
            ]
        else:
            selected_query = [
                (x, y)
                for x, y in q_graph.nodes(data=True)
                if y["avg_acc"] < self.error_thres
            ]

        query1 = [
            f"Title: {sq[1]['paragraph_title']}\nAbstract: {sq[1]['source']}"
            for sq in selected_query
        ]
        query2 = [sq[1]["source"] for sq in selected_query]

        # page level retrieval
        query_vec = self.tmodel.encode(
            query1, convert_to_numpy=True, batch_size=16
        )
        faiss.normalize_L2(query_vec)
        _, page_indices = self.faiss_index.search(
            query_vec, top_k_page
        )  # distance, page_indices
        self.faiss_index.remove_ids(np.array(page_indices).reshape(-1))
        page_indices = np.array(page_indices).reshape(-1)  # (len(query_vec) * top_k_page)
        
        page_indices_ids = list(range(len(page_indices)))
        page_indices = [
            (idx // self.data_per_file, str(idx)[-4:])  # file_id, real_page_idx
            for idx in page_indices
        ]

        selected_pages = []
        for file_idx, real_page_idx in page_indices:
            try:
                selected_pages.append(
                    self._read_page_from_file([file_idx])[int(real_page_idx)]
                )
            except IndexError:
                print(f"IndexError: {file_idx}_{real_page_idx}")
        selected_pages = self._retrieve_paragraph_from_pages(
            [json.loads(page) for page in selected_pages]
        )
        first_parents = [
            selected_query[mat_idx // top_k_page] for mat_idx in page_indices_ids
        ]

        # Optional: add random mixin pages to expend the search space, but add noices.
        mix_pages = []
        if random_mix_ratio > 0:
            mix_indices = self.rng.choice(
                range(self.total_pages),
                int(len(page_indices) * random_mix_ratio),
                replace=False,
            )
            self.faiss_index.remove_ids(mix_indices)
            mix_indices = [
                (idx // self.data_per_file, str(idx)[-4:]) for idx in mix_indices
            ]
            for file_idx, real_page_idx in mix_indices:
                mix_pages.append(
                    self._read_page_from_file([file_idx])[int(real_page_idx)]
                )
            mix_pages = self._retrieve_paragraph_from_pages(
                [json.loads(page) for page in mix_pages], para_cutoff=paragraph_size
            )

        # expend pages and paragraphs
        query_paragraphs_ids = []
        query_paragraphs = []
        for page_idx, page in enumerate(selected_pages):
            for para_idx, para in enumerate(page["paragraphs"]):
                if para["context"] == "":
                    continue
                if (page_idx, para_idx) not in query_paragraphs_ids:
                    query_paragraphs_ids.append((page_idx, para_idx))
                    query_paragraphs.append(para["context"])
        
        # paragraph level retrieval
        input_content_embd = self.tmodel.encode(
            query2, convert_to_numpy=True,
            batch_size=16, normalize_embeddings=True
        )
        query_paragraphs_embd = self.tmodel.encode(
            query_paragraphs, convert_to_numpy=True,
            batch_size=16, normalize_embeddings=True
        )
        sim_mat = np.abs(
            input_content_embd @ query_paragraphs_embd.T
        )  # (inputsize, outputsize)

        selected_indices = top_k_indices_argpartition(
            sim_mat, paragraph_size, axis=1
        ).reshape(-1)  # (inputsize, paragraph_size)
        selected_indices = list(set(selected_indices))
        selected_paragraph_ids = [
            (
                query_paragraphs_ids[idx],
                first_parents[query_paragraphs_ids[idx][0]],  # first parent
                selected_query[mat_idx // paragraph_size],  # second parent
            )  
            for mat_idx, idx in enumerate(selected_indices)
        ]

        return selected_pages, selected_paragraph_ids
    
    def image_related_sampling(
        self,
        q_graph: nx.DiGraph = None,
        top_k_imgs: int = 50,
        qa_per_img: int = 3,
        qa_size: int = 50,
        incorr_list: datasets.Dataset = None,
    ):
        print("==> Image relation based sampling")
        selected_query = [
            (x, y)
            for x, y in q_graph.nodes(data=True)
            if y["avg_acc"] < self.error_thres
        ]
        query1 = [Image.open(sq[1]["source"]) for sq in selected_query]

        # Image-level retrieval using FAISS (Batch Processing)
        query_vec = self.imodel.encode(query1, convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        _, img_indices = self.faiss_index.search(query_vec, top_k_imgs)
        
        img_indices = img_indices.flatten()  # Convert to 1D array
        self.faiss_index.remove_ids(img_indices)
        
        selected_imgs = {self.file_list[i].split("/")[-1] for i in img_indices if i < len(self.file_list)}

        # Filter dataset efficiently using set lookup
        qa_list = self.vision_dataset.filter(lambda x: x["data_path"][0].split("/")[-1] in selected_imgs, num_proc=24)

        # Parent tracking
        parent_mapping = {img: selected_query[i // top_k_imgs] for i, img in enumerate(selected_imgs)}

        # Question-level retrieval using FAISS
        query_questions = [il["question"] for il in incorr_list]
        qa_questions = [fs["question"] for fs in qa_list]

        query_vec = self.tmodel.encode(query_questions, convert_to_numpy=True)
        input_vec = self.tmodel.encode(qa_questions, convert_to_numpy=True)
        sim_mat = np.abs(query_vec @ input_vec.T)

        # Select top-k QA pairs
        selected_qa_set = np.unique(
            top_k_indices_argpartition(sim_mat, qa_size // 2, axis=1).reshape(-1)
        )
        candidate_qa = qa_list.select(selected_qa_set)

        # Efficiently group by image and limit QA per image
        img_qa_map = {}
        for qa in candidate_qa:
            img_name = qa["data_path"][0].split("/")[-1]
            if img_name not in img_qa_map:
                img_qa_map[img_name] = []
            if len(img_qa_map[img_name]) < qa_per_img:
                img_qa_map[img_name].append(qa)
        
        # Randomly select (qa_size // qa_per_img) images with exactly qa_per_img questions
        selected_imgs = [img for img in img_qa_map.keys() if len(img_qa_map[img]) == qa_per_img]
        selected_imgs = self.rng.choice(selected_imgs, min(len(selected_imgs), qa_size // qa_per_img), replace=False)
        
        final_subset = [datasets.Dataset.from_list(img_qa_map[img]) for img in selected_imgs]
        final_parents = [parent_mapping[img] for img in selected_imgs]
        
        # Flatten final QA set
        final_set = concatenate_datasets(final_subset)
        
        return final_set, final_parents
