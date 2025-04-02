# Using Git with VS Code for Spatial Navigation EEG Project

This guide explains how to use VS Code's built-in Git features to manage the repository.

## Prerequisites

- VS Code installed
- Git installed
- GitHub account connected to VS Code

## Connect Your GitHub Account to VS Code

If you haven't already connected your GitHub account to VS Code:

1. Click on the accounts icon in the bottom left corner of VS Code
2. Select "Sign in to GitHub"
3. Follow the prompts to authorize VS Code

## Initialize the Repository (if not already done)

1. Open the Spatial Navigation EEG project folder in VS Code
2. Click on the Source Control icon in the sidebar (or press Ctrl+Shift+G)
3. Click "Initialize Repository"

## Stage and Commit Files

1. In the Source Control panel, you'll see a list of changes
2. Click the "+" (plus) icon next to each file to stage it, or click the "+" next to "Changes" to stage all files
3. Enter a commit message in the text box at the top of the Source Control panel
4. Click the checkmark icon (or press Ctrl+Enter) to commit the staged changes

## Push to GitHub

### Option 1: Create and Push to a New Repository

1. After committing, click the "Publish Branch" button in the Source Control panel
2. Choose either "Public" or "Private" repository
3. Give your repository a name (e.g., "spatial_navigation_eeg")
4. Click "OK" or "Publish"

### Option 2: Connect to an Existing Repository

If you've already created the repository on GitHub:

1. Open a terminal in VS Code (Terminal > New Terminal)
2. Run the following commands:
   ```
   git remote add origin https://github.com/YOUR-USERNAME/spatial_navigation_eeg.git
   git push -u origin main
   ```
3. You'll be prompted to log in to GitHub if you haven't already

## Working with Git in VS Code

### Viewing Changes

- The Source Control panel shows all modified files
- Click on a file to see a diff view comparing changes
- Use the "Changed Files" view to navigate through modifications

### Pull Changes

- Click the "Sync Changes" button (circular arrows) in the Source Control panel
- This performs both push and pull operations

### Branching

1. Click on the branch name in the status bar (bottom left)
2. Select "Create new branch"
3. Enter a branch name and press Enter

### Merge Branches

1. Switch to the target branch (e.g., main)
2. Click on the branch name in the status bar
3. Select "Merge branch..." and choose the source branch

## Using the Git History

1. In the Source Control panel, click the "View History" button (clock icon)
2. Browse through previous commits
3. Click on a commit to see its details

## Git Graph Extension (Recommended)

For better visualization of your repository:

1. Go to Extensions (Ctrl+Shift+X)
2. Search for "Git Graph"
3. Install the extension
4. Access it by right-clicking in the Source Control panel and selecting "View Git Graph"

## Resolving Merge Conflicts

When conflicts occur:

1. VS Code will highlight the conflicting files
2. Click on a file to see the conflicts marked in the editor
3. Edit the file to resolve conflicts
4. Once resolved, stage the file and commit

## Additional Tips

- Use the VS Code terminal for more advanced Git operations
- The status bar shows the current branch and sync status
- Use keyboard shortcut Ctrl+Shift+G to quickly access Git features
