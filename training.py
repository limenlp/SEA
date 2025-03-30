from train.trainer import LLMTrainer


trainer = LLMTrainer(
    model_name="FacebookAI/roberta-large",
    dataset_path="datasets/gpt-4o_search_r50_rps20_qs100_npi5_tkp3_tps5_rmr0_pts0.6_cumacc_context_paraset.jsonl",
)
trainer.train(
    output_dir="./train_res",
    batch_size=16,
    evaluation_steps=30,
    learning_rate=1e-5,
    num_train_epochs=10
)
trainer.evaluate(
    dataset_path="datasets/gpt-4o_search_r50_rps20_qs100_npi5_tkp3_tps5_rmr0_pts0.6_cumacc_context_paraset.jsonl"
)
trainer.finalize()
