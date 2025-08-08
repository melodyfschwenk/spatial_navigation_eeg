/*****************************************************************
 * Spatial Navigation (Behavior-only) – jsPsych 7 implementation *
 *****************************************************************/

const jsPsych = initJsPsych({
  on_finish: () => {
    // Auto-download data as CSV when experiment ends
    jsPsych.data.get().localSave('csv', 'spatial_navigation_data.csv');
  }
});

/* ----------------------------------------------------------------
   1. Preload images
---------------------------------------------------------------- */
const image_files = [];

// CHANGE this number to match your total stimulus count
const N_STIM = 20;

for (let i = 1; i <= N_STIM; i++) {
  image_files.push(`images/maze_${String(i).padStart(2,'0')}.png`);
}

const preload = {
  type: jsPsychPreload,
  images: image_files
};

/* ----------------------------------------------------------------
   2. Welcome / instructions
---------------------------------------------------------------- */
const welcome = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: `
    <h2>Welcome to the Spatial Navigation Task!</h2>
    <p>You will see pictures of mazes.</p>
    <p>Press the arrow key that indicates the correct exit direction<br>
       as quickly and accurately as possible.</p>
    <p><strong>Use the arrow keys: ↑ ↓ ← →</strong></p>
    <p>Press any key to begin.</p>`
};

/* ----------------------------------------------------------------
   3. Trial factory
---------------------------------------------------------------- */
function make_trial(stim_path) {
  return {
    type: jsPsychImageKeyboardResponse,
    stimulus: stim_path,
    prompt: '<p>Use the arrow keys (↑ ↓ ← →)</p>',
    choices: ['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'],
    data: {stimulus: stim_path},
    on_finish: data => {
      // Extract answer from file name, e.g. maze_01_left.png
      const fname = stim_path.split('/').pop().toLowerCase();
      if (fname.includes('_up'))    data.correct_key = 'ArrowUp';
      if (fname.includes('_down'))  data.correct_key = 'ArrowDown';
      if (fname.includes('_left'))  data.correct_key = 'ArrowLeft';
      if (fname.includes('_right')) data.correct_key = 'ArrowRight';
      data.correct = data.response === data.correct_key;
    }
  };
}

const main_trials = jsPsych.randomization.shuffle(
  image_files.map(f => make_trial(f))
);

/* ----------------------------------------------------------------
   4. Goodbye
---------------------------------------------------------------- */
const goodbye = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: `
      <h3>Thank you for completing the task!</h3>
      <p>Your data has been downloaded automatically as
         <code>spatial_navigation_data.csv</code>.</p>
      <p>Please email this file to the researcher.</p>
      <p>Press any key to exit.</p>`
};

/* ----------------------------------------------------------------
   5. Run experiment
---------------------------------------------------------------- */
jsPsych.run([preload, welcome, ...main_trials, goodbye]);
