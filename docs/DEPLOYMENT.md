# Deployment guide (Streamlit Community Cloud)

**Status: not yet deployed.** Publishing needs your own GitHub and Streamlit accounts and your approval. Everything else is ready.

Before you publish: the repository contains no student ID, name or email (the `report/` and `submission/` folders are git-ignored). Commits use the placeholder author `student@example.invalid`.

## 1. Install Git and set your identity (once)
Install Git from https://git-scm.com/download/win, then in PowerShell, inside the project folder:

```powershell
git config user.name "Your GitHub username"
```
```powershell
git config user.email "YOUR-ID+username@users.noreply.github.com"
```
GitHub shows your private noreply address under Settings > Emails. Using it keeps your real email out of the public history.

## 2. Push to GitHub
Create an **empty** repository (no README) named `bnpl-intelligence` on github.com. A private repository also works with Streamlit; check that your lecturer can open it. Then:

```powershell
git branch -M main
```
```powershell
git remote add origin https://github.com/YOUR-USERNAME/bnpl-intelligence.git
```
```powershell
git push -u origin main
```

## 3. Deploy
1. Open https://share.streamlit.io and sign in with GitHub.
2. **Create app**, choose the repository, branch `main`, main file `app.py`.
3. **Advanced settings > Python 3.12**. This matters: the pinned packages are built for 3.12.
4. Deploy. The first build takes several minutes.

## 4. Verify before submitting
- Open the URL in a private window (no login) and visit all seven pages.
- BNPL predictor > Predict with the starting values should show the same percentage as on your computer.
- Put the URL in the report (section D.9) and README. Free apps sleep when unused; open yours shortly before marking and the viva.

## Weekly commits
The rubric gives 3 marks for weekly commits. The history so far is real and all from 24 November. Commit your own review work (report edits, notebook comments, README URL) in the following days; do not change commit dates.
