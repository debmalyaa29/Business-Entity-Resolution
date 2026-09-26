"""
Submission packager module for Amazon ML Challenge 2026.
Builds the required <team_name>_submission.zip structure.
"""

import shutil
import zipfile
from pathlib import Path
from src import config

def package_submission(team_name: str = "AntigravityTeam"):
    """
    Creates the compliant final submission zip archive.
    """
    zip_filename = config.BASE_DIR / f"{team_name}_submission.zip"
    temp_dir = config.BASE_DIR / "temp_submission_pkg"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. output/ directory
        output_dst = temp_dir / "output"
        output_dst.mkdir(parents=True, exist_ok=True)
        if config.MATCHING_RESULTS_PATH.is_file():
            shutil.copy2(config.MATCHING_RESULTS_PATH, output_dst / "matching_results.tsv")
        if config.CANDIDATE_PAIRS_PATH.is_file():
            shutil.copy2(config.CANDIDATE_PAIRS_PATH, output_dst / "candidate_pairs.tsv")

        # 2. code/business_entity_resolution/
        code_dst = temp_dir / "code" / "business_entity_resolution"
        code_dst.mkdir(parents=True, exist_ok=True)

        # Copy src/
        shutil.copytree(config.BASE_DIR / "src", code_dst / "src", dirs_exist_ok=True)

        # Copy documentation and requirements
        req_src = config.BASE_DIR / "requirements.txt"
        if req_src.is_file():
            shutil.copy2(req_src, code_dst / "requirements.txt")

        readme_src = config.BASE_DIR / "README.md"
        if readme_src.is_file():
            shutil.copy2(readme_src, code_dst / "README.md")

        # 3. Documentation_template.md at zip root
        doc_src = config.BASE_DIR / "Documentation_template.md"
        if doc_src.is_file():
            shutil.copy2(doc_src, temp_dir / "Documentation_template.md")

        # Create zip archive
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arcname)

        print(f"[+] Submission package successfully created: {zip_filename}")
        print(f"    Size: {zip_filename.stat().st_size / (1024*1024):.2f} MB")
        return zip_filename

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    package_submission()
