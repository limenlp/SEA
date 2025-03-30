import json
import glob
from tqdm import tqdm


model_name = '*'
para_thres = 0.5
rnd = 20
suffix = "_same_topic"
# file_path = f'res/{model_name}_search_r50_rps20_qs100_npi5_tkp3_tps5_rmr0_pts0.6_cumacc_context'
# save_path = f"./res/{model_name}_search_r50_rps20_qs100_npi5_tkp3_tps5_rmr0_pts06_cumacc_context.jsonl"
folder_list = glob.glob(f'./res{suffix}/{model_name}_search_r{rnd}*')
for folder in folder_list:
    qa_idx = 0
    
    statement_data_list = []  # (question, label)
    para_data_list = []
    
    file_list = glob.glob(f'{folder}/ans_step*.json')
    qa_save_path = f"datasets{suffix}/{folder.split('/')[-1]}_qaset.jsonl"
    para_save_path = f"datasets{suffix}/{folder.split('/')[-1]}_paraset.jsonl"
    
    for fp in tqdm(file_list):
        with open(fp, 'r') as f:
            data = json.load(f)
        for page, pvalue in data.items():
            for pc, pcvalue in pvalue.items():
                if pc in ["id", "abstract"]:
                    continue
                corr = 0
                qa_ids = []
                for qa in pcvalue['qa']:
                    if qa.get('llm_answer', None) is None:
                        continue
                    is_correct = 0 if qa['is_correct'] == False else 1
                    corr += is_correct
                    if isinstance(qa['llm_answer'], str):
                        cost = 0
                        input_token = 0
                        output_token = 0
                    else:
                        cost = qa['llm_answer'].get('total_price', 0)
                        input_token = qa['llm_answer'].get('input_token_usage', 0)
                        output_token = qa['llm_answer'].get('output_token_usage', 0)
                    qa_ids.append(qa_idx)
                    qa_idx += 1
                    statement_data_list.append(json.dumps({
                        "title": pc,
                        "context": pcvalue['context'],
                        "question": qa['question'],
                        "input": qa['statement'],
                        "options": qa['options'],
                        "answer": qa['answer'],
                        "llm_ans": qa['llm_answer']['answer'],
                        "input_tokens": input_token,
                        "output_tokens": output_token,
                        "cost": cost,
                        "label": is_correct
                    }))
                acc = corr / len(pcvalue['qa'])
                para_data_list.append(json.dumps({
                    "title": pc,
                    "input": pcvalue['context'],
                    "acc": acc,
                    "label": 0 if acc <= para_thres else 1,
                    "qa_ids": qa_ids
                }))

    with open(qa_save_path, 'w') as f:
        for i, data in enumerate(statement_data_list):
            if i == len(statement_data_list) - 1:
                f.write(data)
            else:
                f.write(data + '\n')
    
    with open(para_save_path, 'w') as f:
        for i, data in enumerate(para_data_list):
            if i == len(para_data_list) - 1:
                f.write(data)
            else:
                f.write(data + '\n')
                