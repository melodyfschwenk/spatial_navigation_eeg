// Google Apps Script for receiving jsPsych POSTs and appending to a Google Sheet.
// 1) Set SPREADSHEET_ID to your Sheet's ID (from its URL).
// 2) Make sure a sheet/tab named SHEET_NAME exists (e.g., 'data').
// 3) Deploy as a Web App: Deploy > New deployment > Web app > Execute as: Me; Who has access: Anyone.
// 4) Copy the deployment URL and paste it into docs/config.js as DATA_ENDPOINT.

const SPREADSHEET_ID = 'PUT_YOUR_SHEET_ID_HERE';
const SHEET_NAME = 'data';

// Define the columns/fields you wish to store (add/remove as needed).
// These keys should match properties present in jsPsych trial data objects.
const COLUMNS = [
  'timestamp',         // when we append (server time)
  'participant_id',
  'trial_index',
  'time_elapsed',
  'rt',
  'stimulus',
  'response',
  'correct',
  'correct_response',
  'task',
  'label',
  'start_time'         // from jsPsych.data.addProperties
];

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents);
    const trials = body.trials || [];

    const ss = SpreadsheetApp.openById(SPREADSHEET_ID);
    const sheet = ss.getSheetByName(SHEET_NAME) || ss.insertSheet(SHEET_NAME);

    const rows = trials.map(t => {
      const now = new Date();
      return [
        now.toISOString(),
        safe(t.participant_id),
        safe(t.trial_index),
        safe(t.time_elapsed),
        safe(t.rt),
        safe(t.stimulus),
        safe(t.response),
        safe(t.correct),
        safe(t.correct_response),
        safe(t.task),
        safe(t.label),
        safe(t.start_time)
      ];
    });

    if (rows.length > 0) {
      sheet.getRange(sheet.getLastRow() + 1, 1, rows.length, COLUMNS.length).setValues(rows);
    }

    return jsonResponse({ result: 'success', rows_appended: rows.length });
  } catch (err) {
    return jsonResponse({ result: 'error', message: String(err) }, 500);
  }
}

function safe(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'object') return JSON.stringify(v);
  return v;
}

function jsonResponse(obj, status) {
  const out = ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
  // Basic CORS header so browser fetch can read response
  out.setHeader('Access-Control-Allow-Origin', '*');
  out.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  out.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  return out;
}