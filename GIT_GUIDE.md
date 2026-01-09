# Git Walkthrough: Push Local Code to New Branch

This guide lists the exact terminal commands to initialize your local repository and push your current code to a **new branch** on your existing GitHub repository, ensuring the `main` branch remains untouched.

### Prerequisites
- You are in the root directory of your project (where `main_mirror.py` is located).
- You have the URL of your existing GitHub repository (e.g., `https://github.com/YourUsername/YourRepo.git`).

### Step-by-Step Commands

1.  **Initialize Git** (if not already done)
    ```bash
    git init
    ```

2.  **Add the Remote Repository**
    Replace `<YOUR_REPO_URL>` with your actual repository URL.
    ```bash
    git remote add origin <YOUR_REPO_URL>
    ```
    *(If you get "error: remote origin already exists", you can skip this step or check it with `git remote -v`)*

3.  **Create and Switch to a New Branch**
    This ensures you are working on a separate branch (e.g., `feature/api-version`) and not `main`.
    ```bash
    git checkout -b feature/api-version
    ```

4.  **Stage All Files**
    Prepare all current files for commit.
    ```bash
    git add .
    ```

5.  **Commit the Changes**
    Save the changes locally.
    ```bash
    git commit -m "Add unified entry script and VTON logic"
    ```

6.  **Push to the New Branch**
    Upload your code to the remote repository on the specific branch.
    ```bash
    git push -u origin feature/api-version
    ```

### Verification
After pushing, visit your GitHub repository URL in the browser. You should see a notification about the new branch `feature/api-version` and be able to switch to it to view your new code.
