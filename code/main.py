import subprocess
import sys
import os

def run_scripts_from_config(config_path):
    # Check if the configuration file exists
    if not os.path.exists(config_path):
        print(f"Error: the configuration file '{config_path}' does not exist.")
        return

    print(f"Reading configuration file: {config_path}...\n")
    
    # Read the files from the .cfg file
    with open(config_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Remove whitespace and newline characters
        script_path = line.strip()

        # Ignore empty lines and lines starting with '#' (comments)
        if not script_path or script_path.startswith('#'):
            continue

        # Check if the script to be executed actually exists
        if not os.path.exists(script_path):
            print(f"[!] Warning: the file '{script_path}' was not found. Skipping it.")
            continue

        print(f"{'='*50}")
        print(f"[*] Executing: {script_path}")
        print(f"{'-'*50}")
        print("OUTPUT:")
        
        try:
            # Popen allows us to read the output line by line in real time
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Combines errors with standard output
                text=True,
                bufsize=1 # Line buffered
            )
            
            # Read and print the output in real time
            for output_line in process.stdout:
                # Adds a visual prefix to the script's output
                print(f"    | {output_line}", end="")
            
            # Wait for the process to finish and get the return code
            process.wait()
            
            print(f"{'-'*50}")
            if process.returncode == 0:
                print(f"[✓] Successfully completed: {script_path}\n")
            else:
                print(f"[X] Error! '{script_path}' failed with exit code: {process.returncode}\n")
                # Uncomment the line below to stop execution on the first error
                # break
                
        except Exception as e:
            print(f"[X] An unexpected error occurred while running '{script_path}': {e}\n")

if __name__ == "__main__":
    CONFIG_PATH = os.path.join("config", "test.cfg")
    run_scripts_from_config(CONFIG_PATH)