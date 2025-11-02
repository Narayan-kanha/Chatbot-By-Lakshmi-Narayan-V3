import os
import shutil

# The name of the file you want to copy from
source_filename = 'combined_pdf_text.txt'

# The name of the final destination file
destination_filename = 'ULTIMATE_DATSET.txt'

# A temporary filename to use during the copy process.
# We add a .tmp extension to signify it's a temporary file.
temporary_filename = destination_filename + '.tmp'

try:
    print(f"Starting to copy from '{source_filename}' to '{destination_filename}'...")

    # Open the source file for reading and the temporary file for writing
    # The 'with' statement ensures that files are properly closed even if an error occurs
    with open(source_filename, 'r', encoding='utf-8') as source_file, open(temporary_filename, 'w', encoding='utf-8') as temp_file:
        # This loop reads the source file line by line, which is memory-efficient
        for line in source_file:
            # Write the current line to the temporary file
            temp_file.write(line)
    
    # If the loop completes without any errors, we can now safely replace the
    # original destination file with our newly created temporary file.
    # The 'shutil.move' is an atomic operation on most modern operating systems,
    # meaning it's very unlikely to fail midway.
    shutil.move(temporary_filename, destination_filename)

    print("File copied successfully!")

except FileNotFoundError:
    print(f"Error: The source file '{source_filename}' was not found. No changes were made.")
    # If the source file doesn't exist, we ensure no temporary file is left behind
    if os.path.exists(temporary_filename):
        os.remove(temporary_filename)
        
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("An error occurred during the copy process. All changes have been reverted.")
    
    # If any other error occurs, we must remove the temporary file to avoid leaving partial data.
    if os.path.exists(temporary_filename):
        os.remove(temporary_filename)