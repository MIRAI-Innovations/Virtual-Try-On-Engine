# GitHub Migration Manual

Follow these exact steps to push your current workspace to your existing GitHub repository on a new branch.

### Prerequisites
-   Open your terminal (Git Bash or PowerShell) in this folder: `C:\Users\dhaks\OneDrive\Documents\MIRAI\model`
-   Make sure you have your GitHub Repo URL ready (e.g., `https://github.com/YourName/YourRepo.git`)

### Step 1: Initialize Git
If you haven't already initialized git:
```bash
git init
```

### Step 2: Add Remote
Link your local folder to the remote repository. Replace `YOUR_REPO_URL` with your actual URL.
```bash
git remote add origin YOUR_REPO_URL
```
*Note: If it says "remote origin already exists", skip this step.*

### Step 3: Create .gitignore (Crucial for Cleanliness)
Before adding files, run this command to prevent the `archive` folder and other junk from being pushed.
```bash
echo "archive/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "venv/" >> .gitignore
echo "*.jpg" >> .gitignore
```
*Note: We ignore `*.jpg` generally to avoid pushing large test images, but you can remove that line if you want to keep them.*

### Step 4: Create & Switch to New Branch
Crucial Step: This creates a separate branch named `feature/unified-vton` so we don't touch the existing `main` code.
```bash
git checkout -b feature/unified-vton
```

### Step 5: Add Files
Stage all your files for the commit.
```bash
git add .
```

### Step 6: Commit
Save your changes locally.
```bash
git commit -m "Refactor: Unified VTON system with multi-model fallback"
```

### Step 7: Push
Upload the new branch to GitHub.
```bash
git push -u origin feature/unified-vton
```

**Success!** You can now check GitHub to see your new branch.
