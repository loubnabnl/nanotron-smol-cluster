from llama_cpp import Llama
import json
import os
import spacy
import random
import torch
import argparse
from typing import Dict, List, Set, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from datasets import load_dataset


#==== rewrite SelfCheckLLMPrompt to use local gguf models and sample ad-hoc
class SelfCheckLLMPromptLocal:
    def __init__(
        self,
        model = None,
        device = None,
        prompt_template = None,
        local = True
    ):
        self.local = local
        if device is None:
            device = torch.device("mps")
        if not local:
            self.tokenizer = AutoTokenizer.from_pretrained(model, torch_dtype="auto", trust_remote_code=True)
            precision_type = 'auto' if torch.cuda.is_available() else torch.float16 # for mps
            self.model = AutoModelForCausalLM.from_pretrained(model, device_map='auto',trust_remote_code=True, torch_dtype=precision_type).to(device)
            self.model.eval()
            self.model.to(device)
        else:
            self.model = Llama(model_path=model, n_gpu_layers=1, n_ctx=4096)
        self.device = device
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: " if prompt_template is None else prompt_template
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt ({model}) initialized to device {device}")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    def generate(self, input_text):
        if self.local:
            # generate using local gguf model
            response = self.model(input_text, max_tokens=4096, echo=False)
            result = response["choices"][0]["text"]
        else:
            # generate using huggingface model
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(
                inputs.input_ids,
                max_new_tokens=5,
                do_sample=False, # hf's default for Llama2 is True
            )
            output_text = self.tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            result = output_text.replace(input_text, "")
        return result

    def generate_samples(self, sample_num, prompt):
        print("Ad-hoc sampling...")
        sample_list = []
        for i in tqdm(range(sample_num)):
            result = self.generate(prompt)
            sample_list.append(result.replace('\n', ' '))
        return sample_list

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
        prompt_to_generate_samples: str = None,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        if len(sampled_passages) == 0: # if no sampled passages, generate them ad-hoc
            if prompt_to_generate_samples is None:
                raise ValueError("prompt_to_generate_samples is None")
            sampled_passages = self.generate_samples(1, prompt_to_generate_samples)
            print("Sample 0:")
            print(sampled_passages[0])
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                sample = sample.replace("\n", " ") 
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                output_text = self.generate(prompt)
                score_ = self.text_postprocessing(output_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        # tested on Llama-2-chat (7B, 13B)
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]

#==== rewrite SelfCheckNLI to support local gguf models and sample ad-hoc
class SelfCheckNLILocal(SelfCheckLLMPromptLocal):
    def __init__(
        self,
        nli_model = 'potsawee/deberta-v3-large-mnli',
        device = None,
        local = True,
        model = 'mistralai/Mistral-7B-Instruct-v0.2',
    ):
        self.local = local
        if device is None:
            device = torch.device("mps")
        if not local:
            self.tokenizer = AutoTokenizer.from_pretrained(model, torch_dtype="auto",trust_remote_code=True)
            precision_type = 'auto' if torch.cuda.is_available() else torch.float16 # for mps
            self.model = AutoModelForCausalLM.from_pretrained(model, device_map='auto',trust_remote_code=True, torch_dtype=precision_type).to(device)
            self.model.eval()
        else:
            self.model = Llama(model_path=model, n_gpu_layers=1, n_ctx=4096)
        self.nli_tokenizer = DebertaV2Tokenizer.from_pretrained(nli_model, max_length=1024)
        self.nli_model = DebertaV2ForSequenceClassification.from_pretrained(nli_model)
        self.nli_model.eval()

        self.nli_model.to(device)
        self.device = device
        print("SelfCheck-NLI initialized to device", device)

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        prompt_to_generate_samples: str = None,
    ):
        num_sentences = len(sentences)
        if len(sampled_passages) == 0: # if no sampled passages, generate them ad-hoc
            if prompt_to_generate_samples is None:
                raise ValueError("prompt_to_generate_samples is None")
            sampled_passages = self.generate_samples(3, prompt_to_generate_samples)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(sampled_passages):
                print(f"Sample {sample_i} for sentence {sent_i}")
                print(sample)
                inputs = self.nli_tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=[(sentence, sample)],
                    add_special_tokens=True, padding="longest",
                    truncation=True, return_tensors="pt",
                    return_token_type_ids=True, return_attention_mask=True,
                )
                inputs = inputs.to(self.device)
                logits = self.nli_model(**inputs).logits # neutral is already removed
                probs = torch.softmax(logits, dim=-1)
                prob_ = probs[0][1].item() # prob(contradiction)
                scores[sent_i, sample_i] = prob_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for AutoValidator')
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--method', default='nli')
    parser.add_argument('--model_path', default=None, help="path to local gguf model or huggingface model")
    args = parser.parse_args()

    directory = "texts"
    if not os.path.exists(directory):
        os.makedirs(directory)

    nlp = spacy.load("en_core_web_sm")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    #==== initialize selfchecker, nli is faster but prompt is more accurate
    model_path = args.model_path
    if args.method == 'nli':
        selfchecker = SelfCheckNLILocal(device=device, model=model_path, local=args.local) # by default, use deberta-v3 the variant that has been fine-tuned on nli is used
    elif args.method == 'prompt':
        baseline_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        cot_template = prompt_template = """Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Think step by step and put your answer of Yes or No first.\n\nAnswer: """
        selfchecker = SelfCheckLLMPromptLocal(model=model_path, device=device, local=args.local, prompt_template=baseline_template)
    else:
        raise ValueError("method not recognized")

    #==== load a sample from HF dataset for verification
    dataset = load_dataset("HuggingFaceTB/cosmopedia", "openstax", split="train", num_proc=12)
    test_idx = random.randint(0, len(dataset)-1)
    for test_idx in tqdm(range(10)):
        test_passage = dataset[test_idx]['text']
        test_prompt = dataset[test_idx]['prompt']
        print(f"index {test_idx}")
        print(f"Prompt:\n{test_prompt}")
        text_file_path = os.path.join(directory, f"text_{test_idx}.txt")
        prompt_file_path = os.path.join(directory, f"prompt_{test_idx}.txt")
        
        # Save the text to its file
        with open(text_file_path, 'w') as file:
            file.write(test_passage)
        
        # Save the prompt to its file
        with open(prompt_file_path, 'w') as file:
            file.write(test_prompt)

        sentences = [sent for sent in nlp(test_passage).sents] # List[spacy.tokens.span.Span]
        sentences = [sent.text.strip() for sent in nlp(test_passage).sents if len(sent) > 3]
        print('Data loaded!')

        #==== validate with selfchecker
        sent_scores = selfchecker.predict(
            sentences = sentences,
            sampled_passages = [],
            prompt_to_generate_samples = test_prompt
        )
        avg_score = sent_scores.mean()
        print("Average validation score per sentence using {}: {}".format(args.method, avg_score)) # The lower the better, recommended to set a threshold at 0.5
