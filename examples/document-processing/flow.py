from metaflow import (
    FlowSpec,
    step,
    vllm,
    pypi,
    kubernetes,
    secrets,
    Config,
    config_expr,
    current,
    card,
    environment,
    IncludeFile,
    huggingface_hub,
    model,
    retry
)
from metaflow.profilers import gpu_profile
from metaflow.cards import Table, ProgressBar, Markdown
import json


class BatchDocumentProcessing(FlowSpec):

    config = Config("config", default="smol-config.json")
    prompt = IncludeFile("prompt", default="prompt.txt")

    @retry(times=3)
    @pypi(**config_expr("config.huggingface_hub.environment"))
    @secrets(sources=[config_expr("config.huggingface_hub.secret")])
    @huggingface_hub
    @step
    def start(self):
        """Load and prepare RepliQA dataset for batch processing"""
        import math
        import time
        from data import load_and_prepare_dataset, group_by_document_id

        print("Loading RepliQA dataset...")

        num_retries = 0
        while True:
            try:
                dataset = load_and_prepare_dataset(self.config.dataset)
                break
            except Exception as e:
                num_retries += 1
                if num_retries > 3:
                    raise e
                time.sleep(1)
            
        print(f"Loaded {len(dataset)} samples")

        self.document_groups = group_by_document_id(dataset)
        print(f"Grouped into {len(self.document_groups)} documents")

        # Distribute work evenly across available batch inference workers
        max_num_batches = self.config.dataset.get("max_num_batches", 1)
        total_documents = len(self.document_groups)
        batch_size = math.ceil(total_documents / max_num_batches)

        print(f"Distributing {total_documents} documents across {max_num_batches} nodes")
        print(f"Optimal batch size: {batch_size} documents per node")

        self.processing_batches = []
        for i in range(0, total_documents, batch_size):
            batch = self.document_groups[i : i + batch_size]
            if batch:
                self.processing_batches.append(batch)

        print(f"Created {len(self.processing_batches)} processing batches")
        for i, batch in enumerate(self.processing_batches):
            total_questions = sum(len(doc["questions"]) for doc in batch)
            print(f"  Batch {i+1}: {len(batch)} documents, {total_questions} questions")

        # Store for evaluation
        self.sample_questions = []
        for doc_group in self.document_groups:
            for question in doc_group["questions"]:
                self.sample_questions.append({
                    "document_topic": doc_group["document_topic"],
                    "question_id": question["question_id"],
                    "answer": question["answer"],
                })

        self.hf_model_for_vllm = current.huggingface_hub.snapshot_download(
            repo_id=self.config.huggingface_hub.repo_id,
            force_download=self.config.huggingface_hub.force_download,
            allow_patterns=(
                "*"
                if self.config.huggingface_hub.allow_patterns is None
                else self.config.huggingface_hub.allow_patterns.split(",")
            ),
        )

        self.next(self.process_documents, foreach="processing_batches")

    @card(id="inference_status")
    @secrets(sources=[config_expr("config.huggingface_hub.secret")])
    @gpu_profile(interval=1)
    @kubernetes(**config_expr("config.inference_environment_resources"))
    @model(
        load=[
            (
                "hf_model_for_vllm",
                config_expr(
                    "config.huggingface_hub.hf_home + '/models--' + config.huggingface_hub.repo_id.replace('/', '--')"
                ),
            )
        ]
    )
    @environment(vars={"HF_HOME": config_expr("config.huggingface_hub.hf_home")})
    @vllm(model=config_expr("config.huggingface_hub.repo_id"))
    @pypi(packages={"pydantic": "2.11.7"})
    @step
    def process_documents(self):
        """Process document batches with LLM using native vLLM engine and guided decoding"""
        
        import time
        from data import DOCUMENT_TOPICS
        from pydantic import BaseModel # pylint: disable=import-error
        from vllm.sampling_params import SamplingParams, GuidedDecodingParams # pylint: disable=import-error

        class GenerationSchema(BaseModel):
            predicted_topic: str
            topic_confidence: float
            is_answerable: bool
            answer: str
            supporting_evidence: str
            reasoning: str

        json_schema = GenerationSchema.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
        sampling_params = SamplingParams(
            temperature=self.config.processing.temperature,
            max_tokens=self.config.processing.max_tokens,
            guided_decoding=guided_decoding_params
        )
        llm_engine = current.vllm.llm

        m = Markdown("## Native vLLM Inference in Progress...")
        p_doc_groups = ProgressBar(max=len(self.input), label="Doc groups completed")
        current.card["inference_status"].append(m)
        current.card["inference_status"].append(p_doc_groups)
        current.card["inference_status"].refresh()

        batch_results = []
        batch_start_time = time.time()
        
        total_questions = sum(len(doc_group["questions"]) for doc_group in self.input)
        print(f"Processing batch with {len(self.input)} documents, {total_questions} questions")

        question_count = 0
        for ii, doc_group in enumerate(self.input):
            document_text = doc_group["document_extracted"][
                : self.config.processing.max_document_length
            ]

            for jj, question_data in enumerate(doc_group["questions"]):
                question_count += 1
                question_start_time = time.time()
                
                prompt = self.prompt.format(
                    topics=", ".join(DOCUMENT_TOPICS),
                    document_text=document_text,
                    question=question_data["question"],
                )

                try:
                    # Use native vLLM engine with guided decoding
                    outputs = llm_engine.generate([prompt], sampling_params)
                    response_text = outputs[0].outputs[0].text.strip()
                    
                    result = json.loads(response_text)
                    result.update({
                        "question_id": question_data["question_id"],
                        "document_id": doc_group["document_id"],
                        "ground_truth_topic": doc_group["document_topic"],
                        "ground_truth_answer": question_data["answer"],
                        "status": "success",
                        "processing_time_ms": (time.time() - question_start_time) * 1000
                    })

                except (json.JSONDecodeError, Exception) as e:
                    result = {
                        "question_id": question_data["question_id"],
                        "document_id": doc_group["document_id"],
                        "predicted_topic": doc_group["document_topic"],
                        "topic_confidence": 0.5,
                        "is_answerable": True,
                        "answer": "PROCESSING_ERROR",
                        "supporting_evidence": "",
                        "reasoning": f"Native vLLM processing failed: {str(e)}",
                        "ground_truth_topic": doc_group["document_topic"],
                        "ground_truth_answer": question_data["answer"],
                        "status": "error",
                        "error_details": str(e),
                        "processing_time_ms": (time.time() - question_start_time) * 1000
                    }

                batch_results.append(result)
                
                if question_count % 10 == 0:
                    elapsed = time.time() - batch_start_time
                    avg_time_per_question = elapsed / question_count
                    estimated_remaining = (total_questions - question_count) * avg_time_per_question
                    print(f"  Progress: {question_count}/{total_questions} questions "
                          f"({question_count/total_questions:.1%}) - "
                          f"ETA: {estimated_remaining:.1f}s")

            p_doc_groups.update(ii + 1)
            current.card["inference_status"].refresh()

        batch_total_time = time.time() - batch_start_time
        self.batch_results = batch_results
        self.batch_processing_time = batch_total_time
        
        # Update progress card
        success_count = sum(1 for r in batch_results if r["status"] == "success")
        m.update(f"Completed: {len(batch_results)} questions, "
                f"{success_count} successful ({success_count/len(batch_results):.1%}), "
                f"Time: {batch_total_time:.1f}s")
        current.card["inference_status"].refresh()
        
        print(f"Batch completed: {len(batch_results)} questions in {batch_total_time:.2f}s")
        print(f"Success rate: {success_count/len(batch_results):.3f}")
        print(f"Average time per question: {batch_total_time/len(batch_results):.2f}s")

        self.next(self.join_results)

    @pypi(disabled=True)
    @step
    def join_results(self, inputs):
        """Combine results from all parallel processing batches"""
        self.all_results = []
        total_processing_time = 0
        total_success = 0
        n_batches = 0
        
        for i, input_obj in enumerate(inputs):
            self.all_results.extend(input_obj.batch_results)
            total_processing_time += input_obj.batch_processing_time
            batch_success = sum(1 for r in input_obj.batch_results if r["status"] == "success")
            total_success += batch_success
            n_batches += 1
            print(f"Batch {i+1}: {len(input_obj.batch_results)} questions, "
                  f"{batch_success} successful, {input_obj.batch_processing_time:.2f}s")
        

        print(f"Combined {len(self.all_results)} total results")
        print(f"Overall success rate: {total_success/len(self.all_results):.3f}")
        print(f"Total processing time across all batches: {total_processing_time:.2f}s")
        print(f"Average processing time per batch: {total_processing_time/n_batches:.2f}s")
        
        self.sample_questions = inputs[0].sample_questions
        self.total_processing_time = total_processing_time
        
        self.next(self.evaluate_performance)

    @card
    @pypi(
        packages={"pandas": "2.3.0", "matplotlib": "3.10.3", "seaborn": "0.13.2"},
        python="3.12.0",
    )
    @step
    def evaluate_performance(self):
        """Calculate performance metrics and create visualizations"""
        import pandas as pd
        from data import calculate_metrics

        predictions = self.all_results
        ground_truth = self.sample_questions
        gt_lookup = {item["question_id"]: item for item in ground_truth}
        
        aligned_predictions = []
        aligned_ground_truth = []

        for pred in predictions:
            question_id = pred["question_id"]
            if question_id in gt_lookup:
                aligned_predictions.append(pred)
                aligned_ground_truth.append(gt_lookup[question_id])

        metrics = calculate_metrics(aligned_predictions, aligned_ground_truth)
        self.performance_metrics = metrics

        results_df = pd.DataFrame([
            {
                "question_id": pred["question_id"],
                "predicted_topic": pred.get("predicted_topic", "Unknown"),
                "ground_truth_topic": pred.get("ground_truth_topic", "Unknown"),
                "topic_correct": pred.get("predicted_topic") == pred.get("ground_truth_topic"),
                "predicted_answer": pred.get("answer", ""),
                "ground_truth_answer": pred.get("ground_truth_answer", ""),
                "is_answerable": pred.get("is_answerable", True),
                "topic_confidence": pred.get("topic_confidence", 0.0),
                "status": pred.get("status", "unknown"),
                "processing_time_ms": pred.get("processing_time_ms", 0)
            }
            for pred in aligned_predictions
        ])

        self.results_df = results_df

        # Enhanced summary stats with timing and native engine metrics
        summary_stats = {
            "Total Questions Processed": len(results_df),
            "Topic Classification Accuracy": f"{metrics['topic_accuracy']:.3f}",
            "Answer Exact Match": f"{metrics['exact_match']:.3f}",
            "Answerability Accuracy": f"{metrics['answerability_accuracy']:.3f}",
            "Success Rate": f"{(results_df['status'] == 'success').mean():.3f}",
            "Average Topic Confidence": f"{results_df['topic_confidence'].mean():.3f}",
            "Total Processing Time": f"{self.total_processing_time:.2f}s",
            "Average Time per Question": f"{results_df['processing_time_ms'].mean():.0f}ms",
            "Questions per Second": f"{len(results_df)/self.total_processing_time:.2f}",
            "Engine Type": "Native vLLM with Guided Decoding"
        }

        current.card.append(Markdown("# Native vLLM Document Processing Results"))
        current.card.append(Markdown("## Summary Metrics"))
        for key, value in summary_stats.items():
            current.card.append(Markdown(f"**{key}**: {value}"))

        # Topic performance with timing
        topic_dist = results_df.groupby("ground_truth_topic").agg({
            "topic_correct": ["count", "sum", "mean"],
            "processing_time_ms": "mean"
        }).round(3)
        topic_dist.columns = ["Total", "Correct", "Accuracy", "Avg_Time_ms"]
        current.card.append(Markdown("## Topic Classification Performance"))
        current.card.append(Table.from_dataframe(topic_dist))

        # Sample results with performance info
        sample_results = results_df.sample(min(10, len(results_df)))[
            ["question_id", "predicted_topic", "ground_truth_topic", 
             "topic_correct", "status", "processing_time_ms"]
        ]
        current.card.append(Markdown("## Sample Results"))
        current.card.append(Table.from_dataframe(sample_results))

        self.next(self.end)

    @pypi(disabled=True)
    @step
    def end(self):
        """Workflow completion"""
        print("=" * 60)
        print("vLLM BATCH PROCESSING WORKFLOW COMPLETED")
        print("=" * 60)
        print(f"📊 Total Questions Processed: {len(self.all_results):,}")
        print(f"🎯 Topic Classification Accuracy: {self.performance_metrics['topic_accuracy']:.3f}")
        print(f"✅ Answer Exact Match: {self.performance_metrics['exact_match']:.3f}")
        print(f"⚡ Total Processing Time: {self.total_processing_time:.2f}s")
        print(f"🚀 Throughput: {len(self.all_results)/self.total_processing_time:.2f} questions/second")
        print(f"🔧 Engine: Native vLLM with Guided JSON Decoding")
        print("=" * 60)


if __name__ == "__main__":
    BatchDocumentProcessing()