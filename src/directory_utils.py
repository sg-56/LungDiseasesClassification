import os

def change_directory(parent_directory):
    """Change current working directory to parent directory."""
    print(f"Current Directory: {os.getcwd()}")
    os.chdir(parent_directory)
    print(f"Directory Changed to : {os.getcwd()}")

def check_directory(input_directory, max_files_to_display=50):
    """Check directory and list files that are not images."""
    files_displayed = 0
    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            if not any(filename.lower().endswith(ext) for ext in ['.jpg']):
                print(os.path.join(root, filename))
                files_displayed += 1
                if files_displayed >= max_files_to_display:
                    print("Files Checking Complete...")
        if files_displayed >= max_files_to_display:
            print("Files Checking Complete...")
