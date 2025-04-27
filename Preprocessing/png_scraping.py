import os
from PIL import Image
import pytesseract
import pandas as pd
import re
from global_config import datasets_dir


output_dir = os.path.join(datasets_dir, os.getcwd())

# List of image file paths
image_paths = ["image1.jpg", "image2.jpg"]  # Make sure these are in the same directory or use full paths

excluded_keywords = [
    "list of abbreviations", "foreword", "paypal", "contributors",
    "declaration", "sources", "references", "subject index",
    "previous volumes"
]


def extract_cleaned_lines(image_path):
    image = Image.open(image_path)
    raw_text = pytesseract.image_to_string(image)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    cleaned_lines = []
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in excluded_keywords):
            continue
        if line.isdigit():  # Skip standalone numbers (page numbers)
            continue
        if line[:2].strip().isdigit():  # Skip numbered main section headers
            continue
        cleaned_lines.append(line)

    return cleaned_lines


# Step 1: Collect all cleaned lines from all images
all_cleaned_lines = []
for path in image_paths:
    all_cleaned_lines.extend(extract_cleaned_lines(path))

# Step 2: Extract tumor names, removing page numbers using regex
tumour_terms = []
for line in all_cleaned_lines:
    match = re.match(r'^(.*?)(?:\s+\d+)?$', line)
    if match:
        name = match.group(1).strip()
        if name:
            tumour_terms.append(name)

# Step 3: Build and export the final DataFrame
df = pd.DataFrame(tumour_terms, columns=["Tumour Type"])

# Print result
print("\n✅ Extracted Tumour Types:")
print(df)


output_file = os.path.join(output_dir, "scraped_data.csv")
df.to_csv(output_file, index=False)
print(f"File successfully saved to {output_file}")