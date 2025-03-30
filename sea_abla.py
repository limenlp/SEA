import json
import os
import copy
import networkx as nx
import numpy as np
from tqdm import tqdm
from utils.sample import PageRetriever
from utils.qa_generator import QAGenerator
from utils.devapi import LLMAPI, MODEL_MAP
from utils.prompt import TESTEE
from utils.graph import QueryGraph
from utils.utils import ans_extractor, judge_ans


TESTEE_MODEL = [
    "meta-llama/Llama-3.3-70B-Instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",  # llama 3.3 distilled
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-72B-Instruct",
    # "deepseek-ai/DeepSeek-R1"
    # "gpt-4o-mini",
    "gpt-4o",
    # "o1-mini",
]
EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
QUERY_BASE = "context"  # context(paragraph), statement, question
ROUND = 20
INIT_THRESHOLD = 0.85
RANDOM_PAGE_SIZE = 100
QA_SIZE = 200
NUM_PER_ITEM = 5
TOP_K_PAGE = 50
PARA_SIZE_PER_PAGE = 10
RANDOM_MIX_RATIO = 0
PRUNING_THRES = 0.5
PRUNING_METHOD = "avgacc"  # cumacc, avgacc(context only)
GEN_MODEL = "gpt-4o"
SEED = 42
DEBUG = False
REPEAT = 5

INIT_TOPICS = [
    "General reference",
    "Culture and the arts",
    "Geography and places",
    "Health and fitness",
    "History and events",
    "Human activities",
    "Mathematics and logic",
    "Natural and physical sciences",
    "People and self",
    "Philosophy and thinking",
    "Religion and belief systems",
    "Society and social sciences",
    "Technology and applied sciences"
]

RESUME_STEP = None

for model in TESTEE_MODEL:
    step_cost = []
    rng = np.random.default_rng(SEED)
    
    if 'gpt' in model or 'o1' in model or model == 'deepseek-reasoner':
        model_name = model
    else:
        model_name = model.split('/')[1]

    ANS_SAVE_PATH = f"res_same_topic_abla/{model_name}_search\
_r{ROUND}\
_rps{RANDOM_PAGE_SIZE}\
_qs{QA_SIZE}\
_npi{NUM_PER_ITEM}\
_tkp{TOP_K_PAGE}\
_tps{PARA_SIZE_PER_PAGE}\
_rmr{RANDOM_MIX_RATIO}\
_pts{PRUNING_THRES}\
_rep{REPEAT}\
_seed{SEED}\
_{PRUNING_METHOD}\
_{QUERY_BASE}"

    FIG_SAVE_PATH = ANS_SAVE_PATH.replace("res", "figs")

    api = LLMAPI(
        model_path=model,
        request_per_minute=130,
        base_url=MODEL_MAP[model][0],
        key_path=MODEL_MAP[model][1],
    )
    gen_api = LLMAPI(
        model_path=GEN_MODEL,
        request_per_minute=130,
        base_url=MODEL_MAP[GEN_MODEL][0],
        key_path=MODEL_MAP[GEN_MODEL][1],
    )
    pr = PageRetriever(
        data_path="/linxindata/wiki/wikidata/processed_files/id_name_abs_url_para",  # glob format
        faiss_idx_path="/linxindata/wiki/wikidata/processed_files/faiss_index_gte_para.ivf",
        query_base=QUERY_BASE,
        pruning_thres=PRUNING_THRES,
        text_model_name=EMBEDDING_MODEL,
        rng=rng,
        device="cuda:0"
    )
    qa_gen = QAGenerator(gen_api, total_qa_size=QA_SIZE, query_base=QUERY_BASE, rng=rng)
    G = QueryGraph(
        query_base=QUERY_BASE,
        pruning_thres=PRUNING_THRES,
    )

    step = 0
    page_list = None
    pidx_parent_list = None
    acc_list = []
    while step < ROUND:
        
        if not os.path.exists(ANS_SAVE_PATH):
            os.mkdir(ANS_SAVE_PATH)
        if not os.path.exists(FIG_SAVE_PATH):
            os.mkdir(FIG_SAVE_PATH)

        if step == 0 and not INIT_TOPICS:
            page_list = pr.page_random_sampling(page_size=RANDOM_PAGE_SIZE)
        elif step == 0 and INIT_TOPICS:
            page_list = pr.init_sampling(page_size=RANDOM_PAGE_SIZE, init_topics=INIT_TOPICS)
            

        qa_dict, cost1 = qa_gen.generate_qa(
            page_list=page_list,
            pidx_parent_list=pidx_parent_list,
            num_per_item=NUM_PER_ITEM,
            q_graph=G,
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
        step_cost.append((cost1, cost2))

        correct = 0
        ctx_acc = None
        if QUERY_BASE == "context":
            ctx_acc = {qi[1]: [0, 0] for qi in qa_list}  # title: (acc, total)
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

            if QUERY_BASE in ["question", "statement"]:
                if step == 0 and is_correct and G.has_node(qa[QUERY_BASE]):
                    G.remove_node(qa[QUERY_BASE])
                else:
                    nx.set_node_attributes(G, {qa[QUERY_BASE]: is_correct}, "is_correct")
            elif QUERY_BASE == "context":
                ctx_acc[subtitle][0] = (
                    ctx_acc[subtitle][0] + 1 if is_correct else ctx_acc[subtitle][0]
                )
                ctx_acc[subtitle][1] += 1

            if is_correct:
                correct += 1
            qa_dict[page][subtitle]["qa"][real_idx]["is_correct"] = is_correct
        
        print(f"==> Step {step} Acc: {correct / len(qa_list)}")

        if step == 0 and (correct / len(qa_list)) > INIT_THRESHOLD:
            continue

        if QUERY_BASE == "context":
            nx.set_node_attributes(
                G,
                {title: acc_num[0] / acc_num[1] for title, acc_num in ctx_acc.items()},
                "avg_acc",
            )
        
        if PRUNING_THRES > 0 and step > 0:
            G.pruning_by_avg_acc()
            G.clean_graph()

        acc_list.append(correct / len(qa_list))
        json.dump(qa_dict, open(f"{ANS_SAVE_PATH}/ans_step{step}.json", "w"), indent=4)

        wrong_qa_dict = copy.deepcopy(qa_dict)
        for page_name in list(wrong_qa_dict):
            page = wrong_qa_dict[page_name]
            for subtitle in list(page):
                if subtitle in ["id", "abstract"]:
                    continue
                content = page[subtitle]
                content["qa"] = [qa for qa in content["qa"] if qa["is_correct"] is False]
                if len(content["qa"]) == 0:
                    del wrong_qa_dict[page_name][subtitle]
            if len(wrong_qa_dict[page_name].keys()) == 2:
                del wrong_qa_dict[page_name]

        json.dump(
            wrong_qa_dict, open(f"{ANS_SAVE_PATH}/wrong_ans_step{step}.json", "w"), indent=4
        )
        if DEBUG:
            break
        page_list, pidx_parent_list = pr.page_related_sampling(
            top_k_page=TOP_K_PAGE,
            paragraph_size=PARA_SIZE_PER_PAGE,
            random_mix_ratio=RANDOM_MIX_RATIO,
            q_graph=G,
        )
        step += 1

    print(acc_list)
    print(step_cost)
    with open(f"{ANS_SAVE_PATH}/acc_list.txt", "w") as f:
        f.write(json.dumps(acc_list))
        
    with open(f"{ANS_SAVE_PATH}/cost_list.txt", "w") as f:
        f.write(json.dumps(step_cost))
