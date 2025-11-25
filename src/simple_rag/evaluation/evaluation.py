"""
DeepEval RAG Evaluation Script
Evaluates RAG system using faithfulness, answer relevancy, and context relevancy metrics
"""

from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from .custom_llama import CustomLlama3_8B
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from typing import List, Optional
import json



class DeepEvalEvaluator:
    """
    DeepEval-based RAG Evaluator
    
    Evaluates RAG system outputs using DeepEval metrics:
    - Faithfulness: How faithful the answer is to the retrieved contexts
    - Answer Relevancy: How relevant the answer is to the question
    - Contextual Relevancy: How relevant the retrieved contexts are to the question
    """
    
    def __init__(
        self,
        model_name: str = "llama3.1:8b-instruct-q4_K_M",
        faithfulness_threshold: float = 0.6,
        answer_relevancy_threshold: float = 0.6,
        contextual_relevancy_threshold: float = 0.6,
        include_reason: bool = True
    ):
        """
        Initialize DeepEval Evaluator
        
        Args:
            model_name: Ollama model name for evaluation
            faithfulness_threshold: Threshold for faithfulness metric (0-1)
            answer_relevancy_threshold: Threshold for answer relevancy metric (0-1)
            contextual_relevancy_threshold: Threshold for contextual relevancy metric (0-1)
            include_reason: Whether to include reasoning in metric results
        """
        self.model_name = model_name
        self.faithfulness_threshold = faithfulness_threshold
        self.answer_relevancy_threshold = answer_relevancy_threshold
        self.contextual_relevancy_threshold = contextual_relevancy_threshold
        self.include_reason = include_reason
        
        # Initialize model
        self.model = CustomLlama3_8B(model_name=model_name)
        
        # Test cases and results
        self.test_cases: List[LLMTestCase] = []
        self.results = None
        
        print(f"DeepEval Evaluator initialized with model: {model_name}")
    
    def load_dataset(self, json_path: str):
        """
        Load dataset from JSON file
        
        Args:
            json_path: Path to JSON file with RAG outputs
            
        Expected JSON format:
            [
                {
                    "question": "...",
                    "answer": "...",
                    "contexts": ["...", "..."],
                    "ground_truth": "..."  # Optional
                },
                ...
            ]
            OR
            {
                "questions": [...],
                "metadata": {...}
            }
        """
        import os
        
        print(f"\nLoading dataset from: {json_path}")
        abs_path = os.path.abspath(json_path)
        print(f"Absolute path: {abs_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset file not found!\n"
                f"  Provided path: {json_path}\n"
                f"  Absolute path: {abs_path}\n"
                f"  Current directory: {os.getcwd()}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading dataset file: {e}")
        
        # Handle both list format and dict with 'questions' key
        if isinstance(data, list):
            questions_list = data
        elif isinstance(data, dict) and 'questions' in data:
            questions_list = data['questions']
        else:
            raise ValueError("JSON must be a list or dict with 'questions' key")
        
        self.test_cases = []
        for item in questions_list:
            # Ensure contexts is a list
            contexts = item.get('contexts', [])
            if not isinstance(contexts, list):
                contexts = [str(contexts)]
            
            test_case = LLMTestCase(
                input=item['question'],
                actual_output=item['answer'],
                retrieval_context=contexts,
                expected_output=item.get('ground_truth', None)  # Optional
            )
            self.test_cases.append(test_case)
        
        print(f"✓ Loaded {len(self.test_cases)} test cases")
        return self.test_cases
    
    def evaluate(
        self, 
        test_cases: Optional[List[LLMTestCase]] = None, 
        num_test_cases: Optional[int] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Evaluate RAG system with DeepEval metrics
        
        Args:
            test_cases: Optional list of test cases. If not provided, uses loaded test cases.
            num_test_cases: Optional number of test cases to evaluate. If None, evaluates all.
            metrics: Optional list of metric names to evaluate. 
                     Options: 'faithfulness', 'answer_relevancy', 'contextual_relevancy'
                     If None, evaluates all metrics.
        
        Returns:
            Evaluation results
        """
        if test_cases is not None:
            self.test_cases = test_cases
        
        if not self.test_cases:
            raise ValueError("No test cases to evaluate. Load dataset first or provide test_cases.")
        
        # Validate and set metrics to evaluate
        available_metrics = ['faithfulness', 'answer_relevancy', 'contextual_relevancy']
        if metrics is None:
            metrics_to_evaluate = available_metrics
        else:
            # Validate metric names
            invalid_metrics = [m for m in metrics if m not in available_metrics]
            if invalid_metrics:
                raise ValueError(
                    f"Invalid metric(s): {invalid_metrics}. "
                    f"Available metrics: {available_metrics}"
                )
            metrics_to_evaluate = metrics
        
        # Select subset of test cases if specified
        if num_test_cases is not None:
            if num_test_cases <= 0:
                raise ValueError(f"num_test_cases must be positive, got {num_test_cases}")
            if num_test_cases > len(self.test_cases):
                print(f"⚠ Warning: Requested {num_test_cases} test cases but only {len(self.test_cases)} available")
                num_test_cases = len(self.test_cases)
            
            test_cases_to_evaluate = self.test_cases[:num_test_cases]
            print(f"\n{'='*60}")
            print(f"Starting DeepEval Evaluation (SUBSET)")
            print(f"{'='*60}")
            print(f"  Model: {self.model_name}")
            print(f"  Total test cases available: {len(self.test_cases)}")
            print(f"  Evaluating: {num_test_cases} test cases")
            print(f"  Metrics: {', '.join(metrics_to_evaluate)}")
            print(f"{'='*60}\n")
        else:
            test_cases_to_evaluate = self.test_cases
            print(f"\n{'='*60}")
            print(f"Starting DeepEval Evaluation")
            print(f"{'='*60}")
            print(f"  Model: {self.model_name}")
            print(f"  Test cases: {len(self.test_cases)}")
            print(f"  Metrics: {', '.join(metrics_to_evaluate)}")
            print(f"{'='*60}\n")
        
        # Define metrics based on selection
        metrics_list = []
        
        if 'faithfulness' in metrics_to_evaluate:
            faithfulness_metric = FaithfulnessMetric(
                threshold=self.faithfulness_threshold,
                model=self.model,
                include_reason=self.include_reason
            )
            metrics_list.append(faithfulness_metric)
        
        if 'answer_relevancy' in metrics_to_evaluate:
            answer_relevancy_metric = AnswerRelevancyMetric(
                threshold=self.answer_relevancy_threshold,
                model=self.model,
                include_reason=self.include_reason
            )
            metrics_list.append(answer_relevancy_metric)
        
        if 'contextual_relevancy' in metrics_to_evaluate:
            contextual_relevancy_metric = ContextualRelevancyMetric(
                threshold=self.contextual_relevancy_threshold,
                model=self.model,
                include_reason=self.include_reason
            )
            metrics_list.append(contextual_relevancy_metric)
        
        # Run evaluation
        self.results = evaluate(
            test_cases=test_cases_to_evaluate,
            metrics=metrics_list
        )
        
        print(f"\n{'='*60}")
        print(f"Evaluation Complete!")
        print(f"{'='*60}\n")
        return self.results
    
    def save_results(self, output_file: str = 'deepeval_results.json'):
        """
        Save evaluation results to JSON file
        
        Args:
            output_file: Path to save results
        """
        if self.results is None:
            raise ValueError("No results to save. Run evaluate() first.")
        
        print(f"\nSaving results to: {output_file}")
        
        results_dict = {
            'metadata': {
                'model_name': self.model_name,
                'total_test_cases': len(self.test_cases),
                'faithfulness_threshold': self.faithfulness_threshold,
                'answer_relevancy_threshold': self.answer_relevancy_threshold,
                'contextual_relevancy_threshold': self.contextual_relevancy_threshold
            },
            'test_cases': [],
            'summary': {}
        }
        
        for result in self.results:
            test_case_result = {
                'input': result.input,
                'actual_output': result.actual_output,
                'metrics': {}
            }
            
            for metric_name, metric_data in result.metrics_data.items():
                test_case_result['metrics'][metric_name] = {
                    'score': metric_data.score,
                    'threshold': metric_data.threshold,
                    'success': metric_data.success,
                    'reason': metric_data.reason if hasattr(metric_data, 'reason') else None
                }
            
            results_dict['test_cases'].append(test_case_result)
        
        # Calculate averages
        if results_dict['test_cases']:
            metrics_names = list(results_dict['test_cases'][0]['metrics'].keys())
            for metric_name in metrics_names:
                scores = [tc['metrics'][metric_name]['score'] for tc in results_dict['test_cases']]
                success_count = sum(1 for tc in results_dict['test_cases'] if tc['metrics'][metric_name]['success'])
                
                results_dict['summary'][metric_name] = {
                    'average': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'success_rate': success_count / len(results_dict['test_cases'])
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✓ Results saved to {output_file}")
        
        # Print summary
        self._print_summary(results_dict['summary'])
    
    def _print_summary(self, summary: dict):
        """Print evaluation summary"""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        for metric_name, stats in summary.items():
            print(f"\n{metric_name}:")
            print(f"  Average:      {stats['average']:.4f}")
            print(f"  Min:          {stats['min']:.4f}")
            print(f"  Max:          {stats['max']:.4f}")
            print(f"  Success Rate: {stats['success_rate']:.2%}")
        
        print(f"\n{'='*60}\n")


