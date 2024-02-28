import subprocess
import os

def execute_script(script_path):
    # Get the full path to the script
    main_directory = os.getcwd()
    notebooks_directory = 'notebooks_final'
    
    full_script_path = os.path.join(main_directory, notebooks_directory,script_path)
    
    result = subprocess.run(['python', full_script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error executing {script_path}:\n{result.stderr}")

def main():
    scripts_to_execute = ['data-preprocessing.py', 'model-training.py']

    for script in scripts_to_execute:
        print(f"Executing {script}...")
        execute_script(script)
        print(f"Completed {script}...")

if __name__ == "__main__":
    main()
