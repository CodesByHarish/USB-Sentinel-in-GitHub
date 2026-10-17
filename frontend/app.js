let currentSessionId = null;
let currentBRDState = {
  objective: null,
  stakeholders: [],
  scope: { in: [], out: [] },
  functional_reqs: [],
  non_functional_reqs: [],
  constraints: [],
  open_questions: []
};

// Global audio player for Sarvam Bulbul v3
let currentSarvamAudio = null;

// MediaRecorder setup for Sarvam Saaras v3 ASR
let mediaRecorder = null;
let audioChunks = [];
let isRecordingAudio = false;

// Web Speech Recognition as client-side fallback
let recognition = null;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-IN';

  recognition.onresult = (event) => {
    let transcript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    const input = document.getElementById('user-input');
    input.value = transcript;
  };

  recognition.onerror = (event) => {
    console.warn('Speech recognition warning:', event.error);
  };
}

async function toggleMicrophone() {
  const micStatus = document.getElementById('mic-status');

  if (isRecordingAudio) {
    // Stop recording
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
    isRecordingAudio = false;
    updateMicUI(false);
    micStatus.textContent = 'Processing audio via Sarvam Saaras v3 ASR...';
    return;
  }

  // Start recording using MediaRecorder for Sarvam Saaras v3
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        audioChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(track => track.stop());
      if (audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        await sendAudioTurn(audioBlob);
      }
    };

    mediaRecorder.start();
    isRecordingAudio = true;
    updateMicUI(true);
    micStatus.textContent = '🎙️ Recording audio for Sarvam Saaras v3... Click mic again to send.';

    // Start recognition as a live preview in the text box if supported
    if (recognition) {
      const langSelect = document.getElementById('lang-select');
      if (langSelect && langSelect.value === 'ta-IN') {
        recognition.lang = 'ta-IN';
      } else if (langSelect && langSelect.value === 'hi-IN') {
        recognition.lang = 'hi-IN';
      } else {
        recognition.lang = 'en-IN';
      }
      try { recognition.start(); } catch (e) {}
    }
  } catch (err) {
    console.warn('Microphone access issue, using Web Speech API fallback:', err);
    if (recognition) {
      try {
        recognition.start();
        updateMicUI(true);
        micStatus.textContent = 'Listening via browser mic...';
      } catch (e) {
        micStatus.textContent = 'Mic access denied. Please type your requirements.';
      }
    } else {
      alert('Could not access microphone: ' + err.message);
    }
  }
}

function updateMicUI(listening) {
  const micBtn = document.getElementById('mic-btn');
  if (listening) {
    micBtn.classList.add('listening');
  } else {
    micBtn.classList.remove('listening');
  }
}

// Sarvam Bulbul v3 Audio Playback (Voice Service Output)
function playSarvamAudio(base64Audio) {
  const ttsEnabled = document.getElementById('tts-toggle').checked;
  if (!ttsEnabled || !base64Audio) return;

  try {
    if (currentSarvamAudio) {
      currentSarvamAudio.pause();
      currentSarvamAudio.currentTime = 0;
    }
    currentSarvamAudio = new Audio(`data:audio/wav;base64,${base64Audio}`);
    document.getElementById('mic-status').textContent = '🔊 Speaking via Sarvam Bulbul v3...';

    currentSarvamAudio.onended = () => {
      document.getElementById('mic-status').textContent = 'Mic idle. Click microphone or press Enter.';
    };

    currentSarvamAudio.play().catch(e => {
      console.warn('Playback error (check user interaction):', e);
      document.getElementById('mic-status').textContent = 'Click to enable audio playback.';
    });
  } catch (err) {
    console.error('Sarvam audio playback error:', err);
  }
}

// Send Audio Turn to Sarvam Saaras v3 ASR endpoint
async function sendAudioTurn(audioBlob) {
  const sendBtn = document.getElementById('send-btn');
  const input = document.getElementById('user-input');
  const langSelect = document.getElementById('lang-select');
  const selectedLang = langSelect ? langSelect.value : 'auto';

  sendBtn.disabled = true;
  input.disabled = true;
  document.getElementById('mic-status').textContent = 'Transcribing via Sarvam Saaras v3...';

  try {
    const formData = new FormData();
    formData.append('file', audioBlob, 'speech.webm');
    if (currentSessionId) {
      formData.append('session_id', currentSessionId);
    }
    formData.append('target_language_code', selectedLang);
    formData.append('speaker', 'meera');

    const res = await fetch('/api/interview/turn/audio', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Server returned ${res.status}: ${errText}`);
    }

    const data = await res.json();
    handleTurnResponse(data, data.transcribed_text || 'User Voice Input');

  } catch (err) {
    console.error('Error sending audio turn:', err);
    appendMessage('system', 'Error', `Failed to process audio turn: ${err.message}`);
    document.getElementById('mic-status').textContent = 'Error during audio transcription.';
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
  }
}

// Send Text Turn to Backend
async function sendTurn(text) {
  if (!text || !text.trim()) return;

  const sendBtn = document.getElementById('send-btn');
  const input = document.getElementById('user-input');
  const langSelect = document.getElementById('lang-select');
  const selectedLang = langSelect ? langSelect.value : 'auto';

  sendBtn.disabled = true;
  input.disabled = true;

  // Add user bubble
  appendMessage('user', 'User', text);
  input.value = '';
  document.getElementById('mic-status').textContent = 'Reasoning via Groq (with Gemini Fallback) & Sarvam synthesizing voice...';

  try {
    const payload = {
      session_id: currentSessionId,
      user_text: text,
      target_language_code: selectedLang,
      speaker: 'meera'
    };

    const res = await fetch('/api/interview/turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }

    const data = await res.json();
    handleTurnResponse(data);

  } catch (err) {
    console.error('Error sending turn:', err);
    appendMessage('system', 'Error', `Failed to process turn: ${err.message}`);
    document.getElementById('mic-status').textContent = 'Error connecting to backend.';
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

function handleTurnResponse(data, userAudioPrompt = null) {
  currentSessionId = data.session_id;
  document.getElementById('session-badge').textContent = `Session: ${currentSessionId.substring(0, 8)}...`;

  if (userAudioPrompt) {
    appendMessage('user', 'User (Voice via Sarvam Saaras v3)', userAudioPrompt);
  }

  // Identify active brain (Groq or Gemini Fallback)
  const brainName = (data.active_brain === 'gemini') ? 'GEMINI 2.5 FLASH (FALLBACK)' : 'GROQ (LLAMA-3.3-70B)';
  const senderTitle = `Business Analyst [${brainName}]`;

  // Render assistant question with metadata chips
  let metaNotes = `[${data.detected_language_code ? data.detected_language_code.toUpperCase() : 'AUTO'}] Brain: ${brainName}`;
  if (data.updated_fields && Object.keys(data.updated_fields).length > 0) {
    const fieldNames = Object.keys(data.updated_fields).join(', ');
    metaNotes += ` • Updated: ${fieldNames}`;
  }

  appendMessage('assistant', senderTitle, data.next_question, metaNotes, data.audio_base64);

  // Update BRD State & Visuals
  updateActionIndicator(data.next_action);
  updateBRDState(data.brd_state);

  // Play Voice Audio strictly originated from Sarvam Bulbul v3
  if (data.audio_base64) {
    playSarvamAudio(data.audio_base64);
  } else {
    document.getElementById('mic-status').textContent = 'Turn completed. Ready for input.';
  }
}

function appendMessage(type, sender, text, tagText = null, audioBase64 = null) {
  const container = document.getElementById('chat-messages');
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${type}-msg`;

  const metaDiv = document.createElement('div');
  metaDiv.className = 'msg-meta';
  metaDiv.textContent = sender;
  msgDiv.appendChild(metaDiv);

  const bodyDiv = document.createElement('div');
  bodyDiv.className = 'msg-body';
  bodyDiv.textContent = text;

  if (tagText) {
    const chip = document.createElement('div');
    chip.className = 'implied-chip';
    chip.textContent = `⚡ ${tagText}`;
    bodyDiv.appendChild(chip);
  }

  // If audio is available from Sarvam, add a replay button
  if (audioBase64) {
    const audioCtrl = document.createElement('button');
    audioCtrl.className = 'btn btn-outline';
    audioCtrl.style.marginTop = '8px';
    audioCtrl.style.fontSize = '11px';
    audioCtrl.style.padding = '3px 8px';
    audioCtrl.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="5 3 19 12 5 21 5 3"/>
      </svg>
      Replay Sarvam Bulbul Audio
    `;
    audioCtrl.onclick = () => playSarvamAudio(audioBase64);
    bodyDiv.appendChild(audioCtrl);
  }

  msgDiv.appendChild(bodyDiv);
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function updateActionIndicator(action) {
  const el = document.getElementById('action-indicator');
  el.className = `action-pill ${action}`;
  el.textContent = `ACTION: ${action.replace('_', ' ').toUpperCase()}`;
}

function updateBRDState(state) {
  currentBRDState = state;

  // 1. Objective
  const objEl = document.getElementById('val-objective');
  if (state.objective) {
    objEl.innerHTML = `<span>${escapeHtml(state.objective)}</span>`;
  } else {
    objEl.innerHTML = `<span class="placeholder-text">Awaiting initial business goal...</span>`;
  }

  // 2. Stakeholders
  renderList('val-stakeholders', 'count-stakeholders', state.stakeholders, 'No stakeholders identified yet');

  // 3. Scope
  const scopeIn = (state.scope && state.scope.in) || [];
  const scopeOut = (state.scope && state.scope.out) || [];
  renderList('val-scope-in', 'count-scope-in', scopeIn, 'No items in scope');
  renderList('val-scope-out', 'count-scope-out', scopeOut, 'No exclusions specified');

  // 4. Functional Reqs
  renderList('val-functional', 'count-functional', state.functional_reqs, 'Awaiting functional workflows...');

  // 5. Non-Functional Reqs
  renderList('val-non-functional', 'count-non-functional', state.non_functional_reqs, 'Implied scalability, latency, SLA requirements will appear here...');

  // 6. Constraints
  renderList('val-constraints', 'count-constraints', state.constraints, 'No compliance, budget or timeline constraints recorded');

  // 7. Open Questions
  renderList('val-open-questions', 'count-questions', state.open_questions, 'No unresolved questions');

  // Update Raw JSON Viewer
  document.getElementById('json-pre').textContent = JSON.stringify(state, null, 2);
}

function renderList(listId, countId, items, placeholder) {
  const listEl = document.getElementById(listId);
  const countEl = countId ? document.getElementById(countId) : null;

  if (countEl) {
    countEl.textContent = items ? items.length : 0;
  }

  if (!items || items.length === 0) {
    listEl.innerHTML = `<li class="placeholder-item">${placeholder}</li>`;
    return;
  }

  listEl.innerHTML = items
    .map(item => `<li>${escapeHtml(item)}</li>`)
    .join('');
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const micBtn = document.getElementById('mic-btn');
  const resetBtn = document.getElementById('reset-btn');
  const exportBtn = document.getElementById('export-btn');
  const copyJsonBtn = document.getElementById('copy-json-btn');

  // Send turn triggers
  sendBtn.addEventListener('click', () => {
    sendTurn(userInput.value.trim());
  });

  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendTurn(userInput.value.trim());
    }
  });

  // Mic trigger with Sarvam Saaras v3 ASR recording
  micBtn.addEventListener('click', toggleMicrophone);

  // Quick Prompt Chips
  document.querySelectorAll('.prompt-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      userInput.value = chip.dataset.text;
      sendTurn(userInput.value);
    });
  });

  // View Switcher (Structured vs Raw JSON)
  document.querySelectorAll('.view-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');

      const target = tab.dataset.tab;
      if (target === 'cards') {
        document.getElementById('cards-view').classList.add('active');
      } else {
        document.getElementById('json-view').classList.add('active');
      }
    });
  });

  // Reset Session
  resetBtn.addEventListener('click', async () => {
    if (!confirm('Start a fresh BRD interview session? This clears current conversation & progress.')) return;
    try {
      const url = currentSessionId ? `/api/interview/reset?session_id=${currentSessionId}` : '/api/interview/reset';
      const res = await fetch(url, { method: 'POST' });
      const data = await res.json();
      currentSessionId = data.session_id;
      document.getElementById('session-badge').textContent = `Session: ${currentSessionId.substring(0, 8)}...`;
      updateBRDState(data.brd_state);
      updateActionIndicator('probe');

      const chatContainer = document.getElementById('chat-messages');
      chatContainer.innerHTML = `
        <div class="message system-msg">
          <div class="msg-meta">System</div>
          <div class="msg-body">Session reset. Ready for a new BRD discovery conversation.</div>
        </div>
        <div class="message assistant-msg">
          <div class="msg-meta">Business Analyst Agent [Groq / Gemini Fallback]</div>
          <div class="msg-body">Namaste / Vanakkam / Hello! Main aapka Business Requirements Document (BRD) prepare karne me help karunga. What project are we building today?</div>
        </div>
      `;
      document.getElementById('mic-status').textContent = 'Session reset. Ready.';
    } catch (e) {
      alert('Failed to reset session: ' + e.message);
    }
  });

  // Export BRD
  exportBtn.addEventListener('click', async () => {
    if (!currentSessionId) {
      alert('Please conduct at least one interview turn before exporting.');
      return;
    }
    try {
      const res = await fetch(`/api/interview/export/${currentSessionId}`);
      if (!res.ok) throw new Error('Failed to export');
      const md = await res.text();

      const blob = new Blob([md], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `BRD_${currentSessionId.substring(0, 8)}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      alert('Export failed: ' + e.message);
    }
  });

  // Copy JSON
  copyJsonBtn.addEventListener('click', () => {
    const jsonStr = JSON.stringify(currentBRDState, null, 2);
    navigator.clipboard.writeText(jsonStr).then(() => {
      copyJsonBtn.textContent = 'Copied!';
      setTimeout(() => {
        copyJsonBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          Copy JSON
        `;
      }, 1500);
    });
  });

  // Initial State Render
  updateBRDState(currentBRDState);
});
