// Basic behavior-only experiment using jsPsych that records keyboard responses and RTs,
// then posts the data to a Google Apps Script endpoint (Google Sheet).

function downloadCSV(filename, csv) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const a = document.createElement('a');
  const url = URL.createObjectURL(blob);
  a.href = url;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function saveData(jsPsych) {
  const allData = jsPsych.data.get().values(); // array of trial objects
  const payload = {
    sheet: SHEET_NAME,
    trials: allData
  };

  if (!DATA_ENDPOINT || DATA_ENDPOINT.includes('PUT_YOUR_APPS_SCRIPT_WEB_APP_URL_HERE')) {
    // If endpoint not configured, fallback to CSV immediately
    const csv = jsPsych.data.get().csv();
    downloadCSV(`spatial_nav_data_${Date.now()}.csv`, csv);
    return;
  }

  try {
    const resp = await fetch(DATA_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    let ok = false;
    try {
      const j = await resp.json();
      ok = j && (j.result === 'success' || j.status === 'ok');
    } catch (e) {
      ok = resp.ok; // fallback if response body isn't JSON
    }

    if (!ok) {
      const csv = jsPsych.data.get().csv();
      downloadCSV(`spatial_nav_data_${Date.now()}.csv`, csv);
    }
  } catch (e) {
    const csv = jsPsych.data.get().csv();
    downloadCSV(`spatial_nav_data_${Date.now()}.csv`, csv);
  }
}

// Initialize jsPsych and register finish handler
const jsPsych = initJsPsych({
  display_element: 'jspsych-target',
  on_finish: async () => {
    await saveData(jsPsych);
    const code = Math.random().toString(36).slice(2, 8).toUpperCase();
    document.querySelector('#jspsych-target').innerHTML = `
      <div style="max-width:700px;margin:40px auto;font-family:system-ui,Arial,sans-serif;">
        <h2>Thank you!</h2>
        <p>Your responses have been recorded.</p>
        <p><strong>Completion code:</strong> ${code}</p>
        <p>You can close this window now.</p>
      </div>`;
  }
});

const timeline = [];

// Collect participant ID
timeline.push({
  type: jsPsychSurveyText,
  preamble: '<h2>Welcome</h2><p>Please enter your Participant ID to continue.</p>',
  questions: [{ prompt: 'Participant ID:', name: 'participant_id', required: true, placeholder: 'e.g., P001' }],
  button_label: 'Begin',
  on_finish: (data) => {
    const responses = JSON.parse(data.responses);
    const pid = (responses.participant_id || '').trim();
    jsPsych.data.addProperties({
      participant_id: pid,
      start_time: new Date().toISOString()
    });
  }
});

// Instructions
timeline.push({
  type: jsPsychHtmlButtonResponse,
  stimulus: `
    <div style="max-width:700px;margin:0 auto;text-align:left;">
      <h3>Instructions</h3>
      <p>You will see arrows pointing left or right.</p>
      <p>Press the F key if the arrow points LEFT.</p>
      <p>Press the J key if the arrow points RIGHT.</p>
      <p>Please respond as quickly and accurately as possible.</p>
    </div>
  `,
  choices: ['Start']
});

// Trial definitions (simple arrow task as a placeholder for your spatial navigation stimuli)
const trials = [];
const arrows = [
  { stimulus: '<div style="font-size:64px;">←</div>', correct_response: 'f', label: 'left' },
  { stimulus: '<div style="font-size:64px;">→</div>', correct_response: 'j', label: 'right' }
];
// Build 20 trials randomized
for (let i = 0; i < 10; i++) {
  trials.push(...arrows);
}
const shuffled = jsPsych.randomization.shuffle(trials);

// Trial procedure
timeline.push({
  timeline: [{
    type: jsPsychHtmlKeyboardResponse,
    stimulus: jsPsych.timelineVariable('stimulus'),
    choices: ['f', 'j'],
    trial_duration: 2000, // ms
    data: {
      task: 'arrow_choice',
      label: jsPsych.timelineVariable('label'),
      correct_response: jsPsych.timelineVariable('correct_response')
    },
    on_finish: (data) => {
      data.correct = data.response === data.correct_response;
    }
  }],
  timeline_variables: shuffled,
  randomize_order: false
});

// Debrief screen (the on_finish of jsPsych will replace this after saving)
timeline.push({
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<p>Submitting your responses...</p><p>Please wait.</p>',
  choices: 'NO_KEYS',
  trial_duration: 800
});

// Start
jsPsych.run(timeline);