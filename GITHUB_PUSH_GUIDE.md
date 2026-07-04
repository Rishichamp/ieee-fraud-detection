# ============================================================
# GITHUB_PUSH_GUIDE.md
# Step-by-step guide to push this project to GitHub
# (Windows PowerShell version)
# ============================================================

## Prerequisites

Install Git for Windows if not already installed:
  https://git-scm.com/download/win

Verify:
  git --version

---

## Step 1 — Create GitHub Repository

1. Go to: https://github.com/new
2. Fill in:
   - Repository name:  ieee-fraud-detection
   - Description:      Financial fraud detection system using IEEE-CIS dataset. XGBoost + Isolation Forest + Graph GNN ensemble with FastAPI + Docker.
   - Visibility:       Public
   - DO NOT tick:      "Add a README file" (we have our own)
   - DO NOT tick:      "Add .gitignore" (we have our own)
3. Click "Create repository"
4. Copy the repository URL shown — looks like:
   https://github.com/YOUR_USERNAME/ieee-fraud-detection.git

---

## Step 2 — Open PowerShell in Your Project Folder

In VS Code: Press Ctrl + ` (backtick)
Make sure you are in:  R:\Projects\fraud_detection

---

## Step 3 — Run These Commands One by One

### 3a — Configure Git (first time only)
```
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### 3b — Initialize repository
```
git init
```

### 3c — Check what will be committed
```
git status
```
VERIFY: You should NOT see any .csv, .npy, or .zip files listed.
If you do, check .gitignore is in the folder.

### 3d — Stage all files
```
git add .
```

### 3e — First commit
```
git commit -m "Initial commit: IEEE-CIS fraud detection system — XGBoost AUC 0.9589"
```

### 3f — Add remote (replace YOUR_USERNAME)
```
git remote add origin https://github.com/YOUR_USERNAME/ieee-fraud-detection.git
```

### 3g — Push to GitHub
```
git branch -M main
git push -u origin main
```

You will be asked for your GitHub username and password.
NOTE: Use a Personal Access Token as the password, not your GitHub password.

---

## Step 4 — Create Personal Access Token (if needed)

GitHub no longer accepts passwords for git push. Use a token:

1. Go to: https://github.com/settings/tokens/new
2. Note: "fraud-detection-push"
3. Expiration: 90 days
4. Scopes: tick "repo" (full control)
5. Click "Generate token"
6. COPY the token immediately (shown only once)
7. Use this token as the password when git push asks

---

## Step 5 — Verify on GitHub

Go to: https://github.com/YOUR_USERNAME/ieee-fraud-detection

You should see:
  ✅ README.md rendered with badges and documentation
  ✅ All source code folders (api/, dashboard/, preprocessing/, scripts/, models/)
  ✅ Docker files (Dockerfile, docker-compose.yml)
  ✅ .github/workflows/ci.yml (GitHub Actions CI badge)
  ❌ NO .csv files (excluded by .gitignore)
  ❌ NO .npy files (excluded by .gitignore)
  ❌ NO .env file (excluded by .gitignore)

---

## Step 6 — Future Updates

After making changes to the code:
```
git add .
git commit -m "Describe what you changed"
git push
```

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| "not a git repository" | Run: git init |
| "remote origin already exists" | Run: git remote set-url origin https://github.com/YOUR_USERNAME/ieee-fraud-detection.git |
| "Authentication failed" | Use Personal Access Token, not password |
| "large files rejected" | Check .gitignore excludes .npy and .csv |
| "nothing to commit" | Run: git status to check |
