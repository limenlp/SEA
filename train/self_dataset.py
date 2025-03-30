from datasets import load_dataset, DatasetDict


def load_and_process_dataset(dataset_path: str, file_type: str = "json") -> DatasetDict:
    """
    Load a dataset from a CSV or JSON file and process it into Hugging Face's Dataset format.

    Args:
        dataset_path (str): Path to the dataset file.
        file_type (str): Type of the file, either 'csv' or 'json'. Default is 'csv'.

    Returns:
        DatasetDict: A Hugging Face DatasetDict containing train, validation, and test splits.
    """

    if file_type not in ["csv", "json"]:
        raise ValueError("Unsupported file type. Use 'csv' or 'json'.")

    # Load dataset
    dataset = load_dataset(file_type, data_files={"train": dataset_path})

    # Split dataset into train, validation, and test
    split_dataset = dataset["train"].train_test_split(test_size=0.2)
    test_valid = split_dataset["test"].train_test_split(test_size=0.5)

    final_dataset = DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"],
        }
    )

    return final_dataset

