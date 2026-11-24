# Deployment guide (GitHub Desktop + Streamlit Community Cloud)

**Status: deployed on 24 November 2026** at https://bnpl-intelligence.streamlit.app (Streamlit Community Cloud, branch `main`, `app.py`, Python 3.12, no secrets). All seven pages were opened without signing in and the predictor returned 4.0%. The steps below describe how to repeat it. The repository contains no student ID, name or email: `report/` and `submission/` are git-ignored, and existing commits use the placeholder author `student@example.invalid`. The locked packages were checked to resolve for Linux / Python 3.12, which Streamlit uses.

## A. Put the code on GitHub
1. Create a free account at https://github.com/signup. Choose a username that does not reveal your name or ID.
2. On GitHub: Settings > Emails, tick **Keep my email addresses private**. Copy the `...@users.noreply.github.com` address shown there.
3. Install **GitHub Desktop** from https://desktop.github.com and sign in.
4. In GitHub Desktop: File > Options > Git. Name = your username, Email = the noreply address.
5. File > Add local repository, choose the `BNPL_Intelligence` folder, **Add repository**. If it says the folder is not trusted, choose to trust it.
6. Check the **History** tab (real commits) and the **Changes** tab (should be empty; `report` and `submission` must not appear).
7. **Publish repository**, name `bnpl-intelligence`. Untick *Keep this code private* so the lecturer can open it.

Do not use "Upload files" on the GitHub website: it would lose the commit history and could publish `report/`.

## B. Deploy on Streamlit
1. Open https://share.streamlit.io and choose **Continue with GitHub**. Read and accept the permissions yourself.
2. **Create app** > deploy from GitHub. Repository `YOUR-USERNAME/bnpl-intelligence`, branch `main`, main file `app.py`.
3. **Advanced settings > Python 3.12**. Keep Secrets empty. Deploy; the first build takes several minutes.
4. If the app is private, set Settings > Sharing to public.

## C. Verify before submitting
- Open the URL in a private window (not signed in) and visit all seven pages.
- BNPL predictor > Predict with the starting values must show **4.0%**, "Unlikely BNPL user".
- ML laboratory: train one Logistic Regression.
- Add the URL to the report (D.10) and README. Free apps sleep when unused: open it before marking and the viva.

## D. Later commits
Edit a file, open GitHub Desktop, write a short summary, **Commit to main**, **Push origin**. Streamlit updates by itself. Commit real work on the days you do it; do not change dates.

## If something fails
- *Build error about Python or a package:* the app was created without Python 3.12. Delete it in Streamlit and create it again.
- *Any other error:* Manage app (bottom right) shows the log; copy the last lines.
