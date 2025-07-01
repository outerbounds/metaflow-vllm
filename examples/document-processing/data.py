DOCUMENT_TOPICS = [
    "Company Policies",
    "Cybersecurity News",
    "Local Technology and Innovation",
    "Local Environmental Issues",
    "Regional Folklore and Myths",
    "Local Politics and Governance",
    "News Stories",
    "Local Economy and Market",
    "Local Education Systems",
    "Local Arts and Culture",
    "Local News",
    "Small and Medium Enterprises",
    "Incident Report",
    "Regional Cuisine and Recipes",
    "Neighborhood Stories",
    "Local Sports and Activities",
    "Local Health and Wellness",
]


def load_and_prepare_dataset(dataset_config):
    """Load RepliQA dataset and prepare for batch processing"""
    from datasets import load_dataset
    import random

    # Load dataset
    dataset_name = dataset_config["name"]
    splits = dataset_config["splits"]
    sample_size = dataset_config.get("sample_size", None)

    all_data = []
    for split in splits:
        split_data = load_dataset(dataset_name, split=split)
        all_data.extend(split_data)

    # Sample if specified
    if sample_size and len(all_data) > sample_size:
        all_data = random.sample(all_data, sample_size)

    return all_data


def group_by_document_id(dataset):
    """Group questions by document_id for batch processing"""
    document_groups = {}

    for item in dataset:
        doc_id = item["document_id"]
        if doc_id not in document_groups:
            document_groups[doc_id] = {
                "document_id": doc_id,
                "document_topic": item["document_topic"],
                "document_extracted": item["document_extracted"],
                "questions": [],
            }

        document_groups[doc_id]["questions"].append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": item["answer"],
                "long_answer": item["long_answer"],
            }
        )

    return list(document_groups.values())


def create_processing_batches(document_groups, batch_size=50, max_num_batches=None):
    """Create batches for parallel processing with optional max limit"""
    batches = []
    for i in range(0, len(document_groups), batch_size):
        batches.append(document_groups[i : i + batch_size])

        # Stop if we've reached the maximum number of batches
        if max_num_batches and len(batches) >= max_num_batches:
            break

    return batches


def calculate_metrics(predictions, ground_truth):
    """Calculate performance metrics"""
    topic_correct = 0
    answerable_correct = 0
    exact_match = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truth):
        # Topic classification accuracy
        if pred.get("predicted_topic") == gt["document_topic"]:
            topic_correct += 1

        # Answerability prediction
        gt_answerable = gt["answer"] != "UNANSWERABLE"
        pred_answerable = pred.get("is_answerable", True)
        if gt_answerable == pred_answerable:
            answerable_correct += 1

        # Exact match for answers
        if pred.get("answer", "").strip().lower() == gt["answer"].strip().lower():
            exact_match += 1

    return {
        "topic_accuracy": topic_correct / total if total > 0 else 0,
        "answerability_accuracy": answerable_correct / total if total > 0 else 0,
        "exact_match": exact_match / total if total > 0 else 0,
        "total_samples": total,
    }
