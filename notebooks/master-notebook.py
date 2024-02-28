# Master Notebook

import subprocess

def execute_notebook(notebook_path):
    result = subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', r'C:\Users\eswar\f1\notebooks'],capture_output=True)
    print(result.stdout.decode('utf-8'))

def main():
    notebooks_to_execute = ['Data-pre-processing-checkpoint.ipynb', 'Model-Training-checkpoint.ipynb.ipynb']

    for notebook in notebooks_to_execute:
        print(f"Executing {notebook}...")
        execute_notebook(notebook)

if __name__ == "__main__":
    main()

