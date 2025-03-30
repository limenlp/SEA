# Stochastic Error Ascent
![Overall workflow of SEA](assets/teaser.jpg)
Official Implimentation of Stochastic Error Ascent (SEA) of our paper: **Discovering Knowledge Deficiencies of Language Models on Massive Knowledge Base**

If you feel our work is helpful, feel free to ⭐ our project!

# How does SEA Work?
SEA includs six steps:
1. Retrieve top-k error related paragraphs from knowledge base (see `utils/sample.py:page_related_sampling`)
2. Random sample from the retrieved batch
3. Evaluate the Testee LLM on the batch (see `utils/qa_generator.py`)
4. Construct a Relation DAG (see `utils/qa_generator.py`)
5. Prune the source error (see `utils/graph.py`)

# How to Use?

## Preparation
1. Create a new conda environment by
    ```
    conda create -n sea python=3.10
    ```
2. Install requirement package by
    ```
    conda activate sea
    pip install -r requirements.txt
    ```
3. Download the preprocessed Wikipedia data through [**this link**](https://drive.google.com/file/d/1Xj1EO9coL8cF0Tud3Z21_wT0DGkUmSMH/view?usp=drivesdk). You can check the data format in `examples/id_name_abs_url_example.json`

4. Specify the key file (like `keys(examle)/example.key`) and base url for the closed-source model. Check the `MODEL_MAP` in `utils/devapi.py` for details.

## Usage

All the following scripts will log the llm's per-step answer in `res_same_topic[_abla]` and `res_same_seed`. You can check the output format in `examples/qa_example.json`

### Result for Figure 2 (main result) in our paper:
Run the following code:
```
conda activate sea
python sea.py
```
In `text.py`, you need to specify the data path (`DATA_PATH`) to the preprocessed Wikipedia data. It includes all preprocessed pages (`id_name_abs_url_para`) and a faiss index file (`faiss_index_gte_para.ivf`). 

You can also adjust the total budget (`TOTAL_BUDGET`), per-step QA size (`QA_SIZE`), rephrase times (`REPHRASE`), top k value (`TOP_K_PAGE`), embedding model (`EMBEDDING_MODEL`), error and pruning threshold (`ERROR_THRES` and `PRUNING_THRES`), pruning method (`cumacc` and `avgacc`, where `avgacc` is an ablation of our method), QA generator model (default is `gpt-4o`), and seed (default is 42).

### Result for Figure 3 (ablation studies) in our paper:
Run the following code:
```
conda activate sea
python sea_abla.py
python sea_random.py
```

### How to analysis from SEA's results (Fig)?
We leave a post process code in `process_exist.py`, where you can specify a result you want to format. After that you can run the code in `analysis.ipynb` for all the ablation and analysis result visualization.

# Contact
- Linxin Song (linxinso@usc.edu)

# Citation
### BibTeX:
```
TBD
```