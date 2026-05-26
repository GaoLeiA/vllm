import os
import re
import shutil

base_dir = r"c:\projects\vllm\study"
md_files = [f for f in os.listdir(base_dir) if f.endswith('.md')]

# Match standard markdown images ![alt](path)
image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

for md_file in md_files:
    md_path = os.path.join(base_dir, md_file)
    file_base = os.path.splitext(md_file)[0]
    assets_dir_name = f"{file_base}.assets"
    assets_dir_path = os.path.join(base_dir, assets_dir_name)
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all images
    images = image_pattern.findall(content)
    if not images:
        continue
        
    print(f"Processing {md_file}...")
    
    new_content = content
    modified = False
    
    for img_path in set(images):
        if img_path.startswith('http') or img_path.startswith('data:'):
            continue
            
        # Standardize path for looking up the actual file locally
        clean_img_path = img_path.strip()
        # Sometimes paths have leading "./"
        if clean_img_path.startswith("./"):
            clean_img_path = clean_img_path[2:]
            
        clean_img_path_os = clean_img_path.replace('/', os.sep).replace('\\', os.sep)
        
        # Avoid processing already migrated images that start with file_base.assets
        if clean_img_path_os.startswith(assets_dir_name):
            continue
        
        # Absolute path of the source image
        source_path = os.path.join(base_dir, clean_img_path_os)
        
        if os.path.exists(source_path):
            os.makedirs(assets_dir_path, exist_ok=True)
            img_filename = os.path.basename(source_path)
            dest_path = os.path.join(assets_dir_path, img_filename)
            
            # Copy file
            try:
                if os.path.abspath(source_path) != os.path.abspath(dest_path):
                    shutil.copy2(source_path, dest_path)
                
                # Update markdown content
                new_img_ref = f"{assets_dir_name}/{img_filename}"
                new_content = new_content.replace(f"]({img_path})", f"]({new_img_ref})")
                modified = True
            except Exception as e:
                print(f"Failed to copy {source_path}: {e}")
        else:
            print(f"Image not found: {source_path} (in {md_file})")
            
    if modified:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {md_file}.")

print("Markdown image migration complete for study directory.")
