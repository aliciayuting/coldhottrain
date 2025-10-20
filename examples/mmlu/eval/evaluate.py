import argparse
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

choices = ["A", "B", "C", "D"]

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def get_token_logprobs(model, tokenizer, prompt, device):
    """Get log probabilities for answer tokens A, B, C, D"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Get logits for the last token position
    
    # Get log probabilities for each answer choice
    lprobs = []
    for ans in choices:
        # Try different token formats
        token_variants = [
            f" {ans}",
            ans,
            f"{ans}",
        ]
        
        best_lprob = -100
        for variant in token_variants:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            if len(tokens) > 0:
                token_id = tokens[0]
                log_softmax = torch.nn.functional.log_softmax(logits, dim=0)
                lprob = log_softmax[token_id].item()
                best_lprob = max(best_lprob, lprob)
        
        lprobs.append(best_lprob)
    
    return lprobs

def crop_prompt(prompt, tokenizer, max_length=2048):
    """Crop prompt if it exceeds max length"""
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_length:
        # Keep the instruction and crop the examples
        lines = prompt.split("\n\n")
        header = lines[0] + "\n\n"
        examples = lines[1:]
        
        # Remove examples from the beginning until it fits
        while len(tokenizer.encode(header + "\n\n".join(examples))) > max_length and len(examples) > 1:
            examples.pop(0)
        
        return header + "\n\n".join(examples)
    return prompt

def eval(args, subject, model, tokenizer, device, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # Crop prompt if needed
        original_prompt = prompt
        prompt = crop_prompt(prompt, tokenizer, max_length=args.max_length)
        
        # If cropped, reduce k
        while prompt != original_prompt and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            original_prompt = prompt
            prompt = crop_prompt(prompt, tokenizer, max_length=args.max_length)

        label = test_df.iloc[i, test_df.shape[1]-1]

        # Get log probabilities
        lprobs = get_token_logprobs(model, tokenizer, prompt, device)
        
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def main(args):
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    results_dir = os.path.join(args.save_dir, "results_{}".format(args.model_name.replace("/", "_")))
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    print(subjects)
    print(args)

    # Load model
    print(f"Loading model: {args.model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)
    
    model.eval()
    
    print(f"Model loaded on {device}")
    
    all_cors = []
    subject_cors = {}
    subcategory_cors = defaultdict(list)
    category_cors = defaultdict(list)

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = eval(args, subject, model, tokenizer, device, dev_df, test_df)
        
        # Store results
        subject_cors[subject] = cors
        all_cors.append(cors)
        
        # Map to subcategory
        if subject in subcategories:
            for subcat in subcategories[subject]:
                subcategory_cors[subcat].append(cors)
        
        # Map to category
        if subject in subcategories:
            for subcat in subcategories[subject]:
                for cat_name, cat_subcats in categories.items():
                    if subcat in cat_subcats:
                        category_cors[cat_name].append(cors)
        
        model_name_clean = args.model_name.replace("/", "_")
        test_df["{}_correct".format(model_name_clean)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(model_name_clean, choice)] = probs[:, j]
        test_df.to_csv(os.path.join(results_dir, "{}.csv".format(subject)), index=None)

    # Calculate and print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Overall accuracy
    weighted_acc = np.mean(np.concatenate(all_cors))
    print(f"\n{'OVERALL ACCURACY':.<50} {weighted_acc:.3f}")
    
    # Category accuracies
    print(f"\n{'CATEGORY ACCURACIES':.<50}")
    print("-"*70)
    for cat_name in sorted(categories.keys()):
        if cat_name in category_cors:
            cat_acc = np.mean(np.concatenate(category_cors[cat_name]))
            print(f"  {cat_name:.<48} {cat_acc:.3f}")
    
    # Subcategory accuracies
    print(f"\n{'SUBCATEGORY ACCURACIES':.<50}")
    print("-"*70)
    for subcat in sorted(subcategory_cors.keys()):
        subcat_acc = np.mean(np.concatenate(subcategory_cors[subcat]))
        print(f"  {subcat:.<48} {subcat_acc:.3f}")
    
    # Individual subject accuracies
    print(f"\n{'INDIVIDUAL SUBJECT ACCURACIES':.<50}")
    print("-"*70)
    for subject in sorted(subject_cors.keys()):
        subj_acc = np.mean(subject_cors[subject])
        print(f"  {subject:.<48} {subj_acc:.3f}")
    
    print("\n" + "="*70)
    
    # Save summary to file
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"\nModel: {args.model_name}\n")
        f.write(f"n-shot: {args.ntrain}\n")
        f.write(f"Max length: {args.max_length}\n")
        
        f.write(f"\n{'OVERALL ACCURACY':.<50} {weighted_acc:.3f}\n")
        
        f.write(f"\n{'CATEGORY ACCURACIES':.<50}\n")
        f.write("-"*70 + "\n")
        for cat_name in sorted(categories.keys()):
            if cat_name in category_cors:
                cat_acc = np.mean(np.concatenate(category_cors[cat_name]))
                f.write(f"  {cat_name:.<48} {cat_acc:.3f}\n")
        
        f.write(f"\n{'SUBCATEGORY ACCURACIES':.<50}\n")
        f.write("-"*70 + "\n")
        for subcat in sorted(subcategory_cors.keys()):
            subcat_acc = np.mean(np.concatenate(subcategory_cors[subcat]))
            f.write(f"  {subcat:.<48} {subcat_acc:.3f}\n")
        
        f.write(f"\n{'INDIVIDUAL SUBJECT ACCURACIES':.<50}\n")
        f.write("-"*70 + "\n")
        for subject in sorted(subject_cors.keys()):
            subj_acc = np.mean(subject_cors[subject])
            f.write(f"  {subject:.<48} {subj_acc:.3f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Hugging Face model name")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length for the model")
    args = parser.parse_args()
    main(args)