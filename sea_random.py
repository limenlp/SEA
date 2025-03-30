import json
import os
import numpy as np
from tqdm import tqdm
from utils.sample import PageRetriever
from utils.qa_generator import QAGenerator
from utils.devapi import LLMAPI, MODEL_MAP
from utils.prompt import TESTEE
from utils.utils import ans_extractor, judge_ans


TESTEE_MODEL = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-72B-Instruct",
    "gpt-4o",
]
EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
QUERY_BASE = "context"  # context(paragraph), statement, question
ROUND = 1
RANDOM_PAGE_SIZE = 800
QA_SIZE = 4000
NUM_PER_ITEM = 5
GEN_MODEL = "gpt-4o"
DEBUG = False
REPEAT = 5
SEED = 42

for model in TESTEE_MODEL:
    step_cost = []
    rng = np.random.default_rng(SEED)
    
    if 'gpt' in model or 'o1' in model:
        model_name = model
    else:
        model_name = model.split('/')[1]

    ANS_SAVE_PATH = f"res_random/{model_name}\
_rps{RANDOM_PAGE_SIZE}\
_qs{QA_SIZE}\
_npi{NUM_PER_ITEM}\
_rep{REPEAT}\
_seed{SEED}"

    FIG_SAVE_PATH = ANS_SAVE_PATH.replace("res", "figs")

    api = LLMAPI(
        model_path=model,
        request_per_minute=120,
        base_url=MODEL_MAP[model][0],
        key_path=MODEL_MAP[model][1],
    )
    gen_api = LLMAPI(
        model_path=GEN_MODEL,
        request_per_minute=120,
        base_url=MODEL_MAP[GEN_MODEL][0],
        key_path=MODEL_MAP[GEN_MODEL][1],
    )
    pr = PageRetriever(
        data_path="/linxindata/wiki/wikidata/processed_files/id_name_abs_url_para",  # glob format
        faiss_idx_path="/linxindata/wiki/wikidata/processed_files/faiss_index_gte_para.ivf",
        query_base=QUERY_BASE,
        text_model_name=EMBEDDING_MODEL,
        rng=rng,
        device="cuda:0"
    )
    qa_gen = QAGenerator(gen_api, total_qa_size=QA_SIZE, query_base=QUERY_BASE, rng=rng)

    page_list = None
    pidx_parent_list = None
    acc_list = []
    
    if not os.path.exists(ANS_SAVE_PATH):
        os.mkdir(ANS_SAVE_PATH)

    page_list = pr.page_random_sampling(page_size=RANDOM_PAGE_SIZE)

    qa_dict, cost1 = qa_gen.generate_qa(
        page_list=page_list,
        pidx_parent_list=pidx_parent_list,
        num_per_item=NUM_PER_ITEM,
        repeat=REPEAT,
    )
    qa_list = qa_gen.expend_qa_list(
        qa_dict
    )  # (page, subtitle, idx, qa, page_id, page_abs, content['context'])
    input_list = []
    for q in qa_list:
        input_list.append(
            TESTEE.format(
                topic=q[1], que=q[3]["question"], opts="\n".join(q[3]["options"])
            )
        )
    results, cost2 = api.batch_call(
        input_list,
        system_message="",
    )
    step_cost.append(cost1 + cost2)

    correct = 0
    for i, qa_item in enumerate(tqdm(qa_list, desc="Answer: ")):
        page, subtitle, real_idx, qa, _, _, context = qa_item
        llm_ans = results[i]["model_response"]
        qa_dict[page][subtitle]["qa"][real_idx]["llm_answer"] = {
            "answer": llm_ans,
            "input_token_usage": results[i]["input_token_usage"],
            "output_token_usage": results[i]["output_token_usage"],
            "total_price": results[i]["total_price"],
        }
        ext_llm_ans = ans_extractor(llm_ans)
        is_correct = judge_ans(ext_llm_ans, qa["answer"])
        if is_correct:
            correct += 1
        qa_dict[page][subtitle]["qa"][real_idx]["is_correct"] = is_correct

    acc_list.append(correct / len(qa_list))
    json.dump(qa_dict, open(f"{ANS_SAVE_PATH}/ans.json", "w"), indent=4)

    print(acc_list)
    print(step_cost)
    with open(f"{ANS_SAVE_PATH}/acc_list.txt", "w") as f:
        f.write(json.dumps(acc_list))
        
    with open(f"{ANS_SAVE_PATH}/cost_list.txt", "w") as f:
        f.write(json.dumps(step_cost))
