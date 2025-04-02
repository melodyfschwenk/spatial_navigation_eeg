# Git Repository Setup for Spatial Navigation EEG Project

This guide explains how to move your Spatial Navigation EEG project to a Git repository.

## Prerequisites

- [Git](https://git-scm.com/downloads) installed on your computer
- Account on GitHub, GitLab, or another Git hosting service (if you want to push to a remote repository)

## Step 1: Create Empty Directories with .gitkeep

Since we're ignoring the data and logs directories but want to maintain the structure, create .gitkeep files:

```bash
# Navigate to your project directory
cd c:\Users\melod\OneDrive\Desktop\spatial_navigation_eeg

# Create .gitkeep files in empty directories
mkdir -p data logs
touch data/.gitkeep logs/.gitkeep
```

## Step 2: Initialize the Git Repository

```bash
# Navigate to your project directory (if not already there)
cd c:\Users\melod\OneDrive\Desktop\spatial_navigation_eeg

# Initialize the repository
git init
```

## Step 3: Add and Commit Files

```bash
# Add all files (except those in .gitignore)
git add .

# Make the initial commit
git commit -m "Initial commit of Spatial Navigation EEG project"
```

## Step 4: Create a Remote Repository

### GitHub
1. Go to [GitHub](https://github.com) and log in
2. Click the "+" in the top-right corner and select "New repository"
3. Name your repository (e.g., "spatial_navigation_eeg")
4. Choose public or private
5. Do NOT initialize with README, .gitignore, or license (we already have our files)
6. Click "Create repository"

### GitLab
1. Go to [GitLab](https://gitlab.com) and log in
2. Click "New project"
3. Choose "Create blank project"
4. Name your repository and configure visibility
5. Click "Create project"

## Step 5: Connect and Push to the Remote Repository

After creating the remote repository, you'll see instructions. Follow them to connect your local repository.

### For GitHub:

```bash
# Replace YOUR-USERNAME with your GitHub username
git remote add origin https://github.com/YOUR-USERNAME/spatial_navigation_eeg.git

# Push your code to the remote repository
git push -u origin master
# OR if the default branch is 'main'
git push -u origin main
```

### For GitLab:

```bash
# Replace YOUR-USERNAME with your GitLab username
git remote add origin https://gitlab.com/YOUR-USERNAME/spatial_navigation_eeg.git

# Push your code to the remote repository
git push -u origin master
# OR if the default branch is 'main'
git push -u origin main
```

## Step 6: Verify the Push

Go to your remote repository in your browser and verify that all files were pushed correctly.

## Additional Git Commands

### Checking Status

```bash
# See current status (modified/added/deleted files)
git status
```

### Creating a New Branch

```bash
# Create and switch to a new branch
git checkout -b new-feature
```

### Committing Changes

```bash
# Add specific files
git add file1.py file2.py

# Commit with a message
git commit -m "Add new feature X"
```

### Syncing with Remote

```bash
# Pull latest changes
git pull origin main

# Push your changes
git push origin main
```

## Best Practices for This Project

1. **Commit Often**: Make small, logical commits with clear messages
2. **Use Branches**: Create feature branches for new features/experiments
3. **Don't Commit Data**: Keep experimental data separate from code
4. **Document Changes**: Note important changes in commit messages
5. **Version Your Releases**: Tag stable versions with version numbers
