import os
from datasets import load_dataset
import pandas as pd

def download_mmlu(output_dir="data"):
    """
    Download MMLU dataset from Hugging Face and save as CSV files
    in the format expected by the evaluation script.
    """
    print("Downloading MMLU dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("cais/mmlu", "all")
    
    # Create directories
    os.makedirs(os.path.join(output_dir, "dev"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    
    # Get all subjects
    subjects = set()
    for split in ['dev', 'test', 'validation']:
        if split in dataset:
            for item in dataset[split]:
                subjects.add(item['subject'])
    
    subjects = sorted(list(subjects))
    print(f"Found {len(subjects)} subjects")
    
    # Process each subject
    for subject in subjects:
        print(f"Processing {subject}...")
        
        # Process dev split
        dev_data = []
        for item in dataset['dev']:
            if item['subject'] == subject:
                row = [item['question']] + item['choices'] + [item['answer']]
                dev_data.append(row)
        
        if dev_data:
            dev_df = pd.DataFrame(dev_data)
            dev_df.to_csv(
                os.path.join(output_dir, "dev", f"{subject}_dev.csv"),
                index=False,
                header=False
            )
        
        # Process test split
        test_data = []
        for item in dataset['test']:
            if item['subject'] == subject:
                row = [item['question']] + item['choices'] + [item['answer']]
                test_data.append(row)
        
        if test_data:
            test_df = pd.DataFrame(test_data)
            test_df.to_csv(
                os.path.join(output_dir, "test", f"{subject}_test.csv"),
                index=False,
                header=False
            )
        
        # Process validation split (optional)
        val_data = []
        for item in dataset['validation']:
            if item['subject'] == subject:
                row = [item['question']] + item['choices'] + [item['answer']]
                val_data.append(row)
        
        if val_data:
            val_df = pd.DataFrame(val_data)
            val_df.to_csv(
                os.path.join(output_dir, "val", f"{subject}_val.csv"),
                index=False,
                header=False
            )
    
    print(f"\nDataset downloaded successfully to '{output_dir}' directory!")
    print(f"Total subjects: {len(subjects)}")
    
    return subjects

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MMLU dataset")
    parser.add_argument(
        "--output_dir", 
        "-o", 
        type=str, 
        default="data",
        help="Output directory for the dataset (default: data)"
    )
    args = parser.parse_args()
    
    subjects = download_mmlu(args.output_dir)
    
    print("\nDataset structure:")
    print(f"  {args.output_dir}/")
    print(f"    ├── dev/")
    print(f"    │   ├── {subjects[0]}_dev.csv")
    print(f"    │   └── ...")
    print(f"    ├── test/")
    print(f"    │   ├── {subjects[0]}_test.csv")
    print(f"    │   └── ...")
    print(f"    └── val/")
    print(f"        ├── {subjects[0]}_val.csv")
    print(f"        └── ...")