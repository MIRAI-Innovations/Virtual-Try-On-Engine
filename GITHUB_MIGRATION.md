# GitHub Migration Manual (Fixed)

**IMPORTANT:** You previously pushed `venv` (libraries) and some large files. We need to clean this up so your friends can download it easily without errors.

### The Problem
-   **`venv` folder:** Should NOT be on GitHub. Your friends will generate their own using `REQUIREMENTS_GUIDE.md`.
-   **Large Datasets:** Use Google Drive or OneDrive for the 3GB training data. GitHub has a strict 100MB limit per file.

---

### Step 1: Clean Up Git Tracking (Undo the mess)
Run these commands to stop tracking the large files and the venv folder.
```bash
# 1. Stop tracking files (does not delete them from your disk)
git rm -r --cached .

# 2. Update .gitignore to strictly exclude large items
echo "venv/" > .gitignore
echo "venv_310/" >> .gitignore
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "archive/" >> .gitignore
echo "*.zip" >> .gitignore
echo "datasets/train/" >> .gitignore
echo "Mirror_Sessions/" >> .gitignore
```

### Step 2: Re-Add Only Necessary Files
Now we re-add everything, but `git` will respect the new ignore rules.
```bash
git add .
```

### Step 3: Commit the Clean Version
```bash
git commit -m "Fix: Remove venv and large datasets from git tracking"
```

### Step 4: Force Push
Since we are fixing the history on your new branch.
```bash
git push -u origin feature/unified-vton --force
```

---

## How specific files are handled

1.  **Dependencies (Libraries)**:
    -   Your friends will NOT download your `venv`.
    -   They will simply run: `pip install -r requirements.txt` (or follow the guide).
    
2.  **Datasets**:
    -   **`datasets/test`**: I have kept this included (as it's around 500MB, which is acceptable if no single file is >100MB). This allows them to run demos immediately.
    -   **`datasets/train` & `Zip files`**: These are ignored. You should upload the `Virtual tryon data.zip` to Google Drive/OneDrive and share the link in the `README.md` if they strictly need to re-train models.
