# Online, behavior-only version (no EEG)

This folder contains a web-based version of the task using [jsPsych](https://www.jspsych.org/) to collect keyboard responses and reaction times. It can be hosted on GitHub Pages and saves data to a Google Sheet via a small Google Apps Script endpoint.

## Quick start

1) Enable GitHub Pages (Settings > Pages) with source "main" and folder "/docs".

2) Create a Google Sheet and Apps Script:
- Create a Google Sheet, add a tab named `data`.
- Copy the Sheet ID from its URL.
- Open https://script.new and paste the contents of `tools/google_apps_script.gs`.
- Set `SPREADSHEET_ID` to your Sheet ID.
- Deploy > New deployment > Type: Web app > Execute as: Me > Who has access: Anyone.
- Copy the Web App URL.

3) Configure:
- In `docs/config.js`, set `DATA_ENDPOINT` to your Web App URL.
- Commit/push.

4) Visit your Pages URL:
- https://<your-username>.github.io/<repo-name>/
- Enter Participant ID and complete the task.
- Data appends to your Google Sheet.
- If network submission fails, a CSV is downloaded locally as a fallback.

## Customization

- Stimuli: Edit the `trials` generation in `docs/experiment.js`. Replace arrows with your stimuli (text or images).
- Keys: Update `choices` and `correct_response` mapping.
- Trial count/duration: Modify `trial_duration` and number of trials.
- Metadata: Add properties via `jsPsych.data.addProperties`.

## Notes

- This version intentionally excludes all EEG components and only collects behavior.
- For PsychoPy-based tasks, you can alternatively export to PsychoJS and host on Pavlovia, which also handles data storage.