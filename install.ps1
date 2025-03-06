Param(
    [int]$NumInstances = 10
)

# Dynamically calculate number of instances if default value is used
if ($NumInstances -eq 10) {
    $cpuCount = $env:NUMBER_OF_PROCESSORS
    $calculated = [math]::Floor($cpuCount / 2)
    if ($calculated -lt 1) { $calculated = 1 }
    $NumInstances = $calculated
    Write-Host "Dynamic instance allocation based on CPU cores: $NumInstances instance(s) allocated (CPU cores: $cpuCount)"
} else {
    Write-Host "Using provided instance count: $NumInstances"
}

# Set fixed short working directory
$workingDir = "C:\0N3-app"
if (!(Test-Path $workingDir)) { New-Item -ItemType Directory -Path $workingDir | Out-Null }
Set-Location $workingDir

# Remove all items except the persistent folder
Get-ChildItem -Force | Where-Object { $_.Name -ne "persistent" } | Remove-Item -Recurse -Force

# Copy background.png from script directory if exists
if (Test-Path "$PSScriptRoot\background.png") {
    Copy-Item "$PSScriptRoot\background.png" -Destination "$workingDir\background.png"
}

Write-Host "Checking for Chocolatey..."
$chocoCheck = Get-Command choco -ErrorAction SilentlyContinue
if (-not $chocoCheck) {
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-WebRequest "https://chocolatey.org/install.ps1" -UseBasicParsing | Invoke-Expression
}

Write-Host "Checking for Python..."
$pythonCheck = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCheck) { choco install python -y }

Write-Host "Checking for Node.js..."
$nodeCheck = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCheck) { choco install nodejs-lts -y }

Write-Host "Removing any existing MongoDB installation (not needed)..."
try {
    $existingMongo = Get-Service -Name "MongoDB" -ErrorAction SilentlyContinue
    if ($existingMongo) {
        Write-Host "Existing MongoDB service detected. Uninstalling..."
        choco uninstall mongodb -y
        Start-Sleep -Seconds 10
    }
} catch {
    Write-Host "No existing MongoDB service found."
}

@"
import os
import time
import threading
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, Request
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTENT_DIR = os.path.join(BASE_DIR, "persistent")
os.makedirs(PERSISTENT_DIR, exist_ok=True)
MODEL_CHECKPOINT = os.path.join(PERSISTENT_DIR, "student_model.pt")
TRAINING_DATA_FILE = os.path.join(PERSISTENT_DIR, "training_data.txt")

class TuningProfile:
    def __init__(self,
                 teacher_name="gpt2",
                 student_name="distilgpt2",
                 lr=1e-5,
                 batch_size=2,
                 distill_epochs=1,
                 gen_max_length=80,
                 gen_temperature=0.9,
                 gen_repetition_penalty=1.2,
                 gen_top_k=50,
                 gen_top_p=0.95):
        self.teacher_name = teacher_name
        self.student_name = student_name
        self.lr = lr
        self.batch_size = batch_size
        self.distill_epochs = distill_epochs
        self.gen_max_length = gen_max_length
        self.gen_temperature = gen_temperature
        self.gen_repetition_penalty = gen_repetition_penalty
        self.gen_top_k = gen_top_k
        self.gen_top_p = gen_top_p

default_profile = TuningProfile()

class Distiller:
    def __init__(self, tuning_profile=default_profile, device=None):
        self.tp = tuning_profile
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_tokenizer = GPT2Tokenizer.from_pretrained(self.tp.teacher_name)
        self.teacher_model = GPT2LMHeadModel.from_pretrained(self.tp.teacher_name).to(self.device)
        self.teacher_model.eval()
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.teacher_model.resize_token_embeddings(len(self.teacher_tokenizer))
        self.student_tokenizer = AutoTokenizer.from_pretrained(self.tp.student_name)
        self.student_model = AutoModelForCausalLM.from_pretrained(self.tp.student_name).to(self.device)
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.student_model.resize_token_embeddings(len(self.student_tokenizer))
        self.student_optimizer = optim.AdamW(self.student_model.parameters(), lr=self.tp.lr)
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.batch_size = self.tp.batch_size
        self.distill_epochs = self.tp.distill_epochs
        self.is_trainer = os.environ.get("IS_TRAINER")=="1"
        self.bg_thread = None
        self.bg_running = False
        self.model_lock = threading.Lock()
        self.last_model_update = 0
        if not self.is_trainer:
            self.start_model_monitor()

    def start_model_monitor(self):
        def monitor():
            while True:
                if os.path.exists(MODEL_CHECKPOINT):
                    mod_time = os.path.getmtime(MODEL_CHECKPOINT)
                    if mod_time > self.last_model_update:
                        with self.model_lock:
                            state_dict = torch.load(MODEL_CHECKPOINT, map_location=self.device)
                            self.student_model.load_state_dict(state_dict)
                        self.last_model_update = mod_time
                        logging.info("Model updated from persistent storage.")
                time.sleep(5)
        threading.Thread(target=monitor, daemon=True).start()

    def add_training_text(self, text):
        with open(TRAINING_DATA_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _read_training_data(self):
        if os.path.exists(TRAINING_DATA_FILE):
            with open(TRAINING_DATA_FILE, "r", encoding="utf-8") as f:
                data = f.read().strip().splitlines()
            return data
        return []

    def _clear_training_data(self):
        open(TRAINING_DATA_FILE, "w", encoding="utf-8").close()

    def _distill_step(self, teacher_batch, student_batch):
        with torch.no_grad():
            t_out = self.teacher_model(**teacher_batch)
        s_out = self.student_model(**student_batch)
        t_logits = t_out.logits
        s_logits = s_out.logits
        seq_len = min(t_logits.size(1), s_logits.size(1))
        t_logits = t_logits[:, :seq_len, :]
        s_logits = s_logits[:, :seq_len, :]
        t_probs = torch.nn.functional.log_softmax(t_logits, dim=-1)
        s_probs = torch.nn.functional.log_softmax(s_logits, dim=-1)
        loss = self.kl_div_loss(s_probs, t_probs.exp())
        self.student_optimizer.zero_grad()
        loss.backward()
        self.student_optimizer.step()
        return loss.item()

    def background_training_loop(self):
        while self.bg_running:
            texts = self._read_training_data()
            if not texts:
                time.sleep(2)
                continue
            for _ in range(self.distill_epochs):
                idx = 0
                while idx < len(texts):
                    batch = texts[idx: idx+self.batch_size]
                    idx += self.batch_size
                    if not batch: break
                    t_in = self.teacher_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    s_in = self.student_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    self._distill_step(t_in, s_in)
            self._clear_training_data()
            with self.model_lock:
                torch.save(self.student_model.state_dict(), MODEL_CHECKPOINT)
            time.sleep(2)

    def start_background(self):
        if self.is_trainer and not self.bg_thread:
            self.bg_running = True
            self.bg_thread = threading.Thread(target=self.background_training_loop, daemon=True)
            self.bg_thread.start()

    def stop_background(self):
        self.bg_running = False
        if self.bg_thread:
            self.bg_thread.join()
        self.bg_thread = None

    def generate(self, prompt, max_length=None, temperature=None, repetition_penalty=None):
        with self.model_lock:
            if os.path.exists(MODEL_CHECKPOINT):
                state_dict = torch.load(MODEL_CHECKPOINT, map_location=self.device)
                self.student_model.load_state_dict(state_dict)
        max_length = max_length if max_length is not None else self.tp.gen_max_length
        temperature = temperature if temperature is not None else self.tp.gen_temperature
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.tp.gen_repetition_penalty
        shaped_prompt = f"User: {prompt}\nAI:"
        enc = self.student_tokenizer(shaped_prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.student_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=self.tp.gen_top_k,
                top_p=self.tp.gen_top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.student_tokenizer.pad_token_id
            )
        txt = self.student_tokenizer.decode(out[0], skip_special_tokens=True)
        if "AI:" in txt: txt = txt.split("AI:",1)[-1].strip()
        return txt

app = FastAPI()
distiller = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global distiller
    distiller = Distiller(tuning_profile=default_profile)
    distiller.start_background()
    logging.info("Application startup complete.")
    yield
    distiller.stop_background()

app.router.lifespan_context = lifespan

@app.post("/chat")
async def chat_endpoint(request: Request):
    global distiller
    data = await request.json()
    user_text = data.get("message", "")
    if not user_text.strip():
        return {"reply": "I didn't catch that."}
    distiller.add_training_text(user_text)
    reply = distiller.generate(user_text)
    return {"reply": reply}

@app.post("/reason")
async def reason_endpoint(request: Request):
    global distiller
    data = await request.json()
    prompt = data.get("message", "")
    if not prompt.strip():
        return {"reply": "No prompt."}
    reply = distiller.generate(f"Reasoning steps about: {prompt}", max_length=100, temperature=1.0, repetition_penalty=1.1)
    return {"reply": reply}

@app.post("/set_tuning")
async def set_tuning_endpoint(request: Request):
    global distiller
    data = await request.json()
    for key, value in data.items():
        if hasattr(distiller.tp, key):
            setattr(distiller.tp, key, value)
    return {"status": "ok", "tuning": data}

@app.get("/status")
async def status_endpoint():
    return {"status": "ready"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

if __name__=="__main__":
    main()
"@ | Set-Content -Encoding UTF8 -Path "model.py"

@"
<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <title>0N3 Distillation + Multi-Instance Reasoning</title>
  <link rel='stylesheet' href='style.css'>
</head>
<body>
  <div class='container'>
    <h1>0N3 Distillation (Multi-Instance Reasoning)</h1>
    <div id='status-bar'></div>
    <div id='toggle-container'>
      <button id='toggle-tuning-btn'>Show Tuning Options</button>
      <div id='tuning-panel'>
        <div class='tuning-field'><label for='gen_max_length'>Max Length:</label><input type='number' id='gen_max_length' value='80' /></div>
        <div class='tuning-field'><label for='gen_temperature'>Temperature:</label><input type='number' step='0.1' id='gen_temperature' value='0.9' /></div>
        <div class='tuning-field'><label for='gen_repetition_penalty'>Repetition Penalty:</label><input type='number' step='0.1' id='gen_repetition_penalty' value='1.2' /></div>
        <div class='tuning-field'><label for='gen_top_k'>Top K:</label><input type='number' id='gen_top_k' value='50' /></div>
        <div class='tuning-field'><label for='gen_top_p'>Top P:</label><input type='number' step='0.01' id='gen_top_p' value='0.95' /></div>
        <button id='set-tuning-btn'>Set Tuning</button>
      </div>
    </div>
    <div id='chat-window'>
      <div id='messages'></div>
    </div>
    <div id='input-area'>
      <input type='text' id='user-input' placeholder='Type a message...' autocomplete='off' autocorrect='off' autocapitalize='off' spellcheck='false' />
      <button id='send-btn'>Send</button>
    </div>
  </div>
  <script src='script.js'></script>
</body>
</html>
"@ | Set-Content -Encoding UTF8 -Path "index.html"

@"
body {
  margin: 0;
  padding: 0;
  background: url('background.png') no-repeat center center fixed;
  background-size: cover;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #fff;
  text-shadow: 0 0 3px #00fcff;
}
.container {
  width: 600px;
  margin: 50px auto;
  background: rgba(17, 17, 17, 0.85);
  border-radius: 10px;
  box-shadow: 0 0 20px rgba(0,255,255,0.2);
  overflow: hidden;
  border: 1px solid rgba(0,255,255,0.2);
  padding-bottom: 10px;
}
h1 {
  margin: 0;
  padding: 20px;
  font-size: 1.5rem;
  text-align: center;
  background: linear-gradient(90deg, #00fcff, #00a2ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: none;
}
#toggle-container {
  padding: 10px;
  text-align: center;
}
#toggle-tuning-btn {
  padding: 5px 10px;
  border: none;
  background: #00fcff;
  color: #111;
  font-weight: bold;
  cursor: pointer;
  margin-bottom: 10px;
}
#tuning-panel {
  display: none;
  background: rgba(0,0,0,0.5);
  padding: 10px;
  border-radius: 5px;
  max-width: 500px;
  margin: 0 auto;
}
.tuning-field {
  margin: 5px 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.tuning-field label {
  flex: 1;
  text-align: left;
  padding-right: 10px;
}
.tuning-field input {
  flex: 1;
  padding: 5px;
  border: none;
  border-radius: 4px;
}
#status-bar {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 10px;
  flex-wrap: wrap;
}
.status-item {
  display: flex;
  align-items: center;
  margin: 5px;
  padding: 5px;
  border: 1px solid rgba(0,255,255,0.3);
  border-radius: 5px;
  background: rgba(0,0,0,0.4);
}
.status-circle {
  width: 15px;
  height: 15px;
  border-radius: 50%;
  margin-right: 5px;
}
.status-ready { background-color: green; }
.status-busy { background-color: yellow; }
.status-error { background-color: red; }
#chat-window {
  height: 300px;
  overflow-y: auto;
  padding: 15px;
  display: flex;
  flex-direction: column;
  scrollbar-color: #00fcff #0b0c0d;
  scrollbar-width: thin;
}
#chat-window::-webkit-scrollbar { width: 8px; }
#chat-window::-webkit-scrollbar-track { background-color: #0b0c0d; }
#chat-window::-webkit-scrollbar-thumb { background-color: #00fcff; }
.message {
  margin: 10px 0;
  padding: 10px;
  border-radius: 6px;
  max-width: 80%;
  word-wrap: break-word;
  box-shadow: 0 0 3px rgba(0,255,255,0.3);
}
.user {
  background: linear-gradient(90deg, #006aff, #00fcff);
  color: #fff;
  margin-left: auto;
  align-self: flex-end;
  text-shadow: none;
}
.ai {
  background: #222;
  color: #00fcff;
  margin-right: auto;
  align-self: flex-start;
  text-shadow: none;
  border: 1px solid rgba(0,255,255,0.3);
}
#input-area {
  display: flex;
  border-top: 1px solid rgba(0,255,255,0.2);
}
#user-input {
  flex: 1;
  padding: 15px;
  border: none;
  outline: none;
  background: #0b0c0d;
  color: #00fcff;
  font-size: 1rem;
  text-shadow: 0 0 3px #00fcff;
}
#send-btn {
  width: 80px;
  border: none;
  background: #00fcff;
  color: #111;
  font-weight: bold;
  cursor: pointer;
  transition: background 0.2s ease;
  text-shadow: none;
}
#send-btn:hover { background: #00a2ff; }
"@ | Set-Content -Encoding UTF8 -Path "style.css"

@"
const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const setTuningBtn = document.getElementById('set-tuning-btn');
const toggleTuningBtn = document.getElementById('toggle-tuning-btn');
const tuningPanel = document.getElementById('tuning-panel');
const statusBar = document.getElementById('status-bar');
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keyup', (e) => { if(e.key === 'Enter') { sendMessage(); } });
toggleTuningBtn.addEventListener('click', () => {
  if(tuningPanel.style.display === 'none' || tuningPanel.style.display === '') { tuningPanel.style.display = 'block'; toggleTuningBtn.textContent = 'Hide Tuning Options'; }
  else { tuningPanel.style.display = 'none'; toggleTuningBtn.textContent = 'Show Tuning Options'; }
});
setTuningBtn.addEventListener('click', setTuning);
let lastAIMessageElem = null;
function sendMessage() {
  const message = userInput.value.trim();
  if(!message) return;
  addMessageToUI(message, 'user');
  lastAIMessageElem = addMessageToUI("Thinking...", 'ai', true);
  fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: message })
  })
  .then(res => res.json())
  .then(data => {
    fetch('/api/reason', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: message })
    })
    .then(res => res.json())
    .then(reasonData => { if(lastAIMessageElem) { lastAIMessageElem.textContent = reasonData.reply; } else { addMessageToUI(reasonData.reply, 'ai'); } })
    .catch(err => { if(lastAIMessageElem) { lastAIMessageElem.textContent = "Error in reasoning: " + err.toString(); } else { addMessageToUI("Error in reasoning: " + err.toString(), 'ai'); } });
  })
  .catch(err => { addMessageToUI("Error: " + err.toString(), 'ai'); });
  userInput.value = '';
}
function addMessageToUI(text, sender, returnElement=false) {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('message');
  msgDiv.classList.add(sender);
  msgDiv.textContent = text;
  messagesDiv.appendChild(msgDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  if(returnElement) return msgDiv;
}
function setTuning() {
  const tuning = {
    gen_max_length: parseInt(document.getElementById('gen_max_length').value),
    gen_temperature: parseFloat(document.getElementById('gen_temperature').value),
    gen_repetition_penalty: parseFloat(document.getElementById('gen_repetition_penalty').value),
    gen_top_k: parseInt(document.getElementById('gen_top_k').value),
    gen_top_p: parseFloat(document.getElementById('gen_top_p').value)
  };
  fetch('/api/set_tuning', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(tuning)
  })
  .then(res => res.json())
  .then(data => { alert("Tuning parameters updated"); })
  .catch(err => { alert("Error updating tuning: " + err.toString()); });
}
function updateStatusBar() {
  fetch('/api/status')
  .then(res => res.json())
  .then(data => {
    statusBar.innerHTML = '';
    for(let i=0;i<data.statuses.length;i++){
      const item = document.createElement('div');
      item.classList.add('status-item');
      const circle = document.createElement('div');
      circle.classList.add('status-circle');
      let stat = data.statuses[i].status;
      if(stat==="ready") { circle.classList.add('status-ready'); }
      else if(stat==="busy") { circle.classList.add('status-busy'); }
      else { circle.classList.add('status-error'); }
      const label = document.createElement('span');
      label.textContent = "Inst " + data.statuses[i].port;
      item.appendChild(circle);
      item.appendChild(label);
      statusBar.appendChild(item);
    }
  })
  .catch(err => { console.error("Error fetching status: ", err); });
}
setInterval(updateStatusBar, 5000);
"@ | Set-Content -Encoding UTF8 -Path "script.js"

@"
const express = require('express');
const fetch = require('node-fetch');
const { spawn } = require('child_process');
const numInstances = $NumInstances;
const aggregatorPort = 3000;
const startPyPorts = 8100;
const pythonProcesses = [];
for (let i = 0; i < numInstances; i++) {
  const pPort = startPyPorts + i;
  const env = Object.assign({}, process.env, { IS_TRAINER: i === 0 ? "1" : "0" });
  const pyProc = spawn('python', ['model.py', '--port=' + pPort], { cwd: __dirname, env: env });
  pyProc.stdout.on('data', (data) => { console.log("PYTHON " + pPort + ": " + data.toString()); });
  pyProc.stderr.on('data', (data) => { console.error("PYTHON ERR " + pPort + ": " + data.toString()); });
  pythonProcesses.push(pyProc);
}
const expressApp = require('express');
const app = expressApp();
app.use(express.json());
app.use(express.static(__dirname));
app.post('/api/chat', async (req, res) => {
  const userMsg = req.body.message || '';
  try {
    const primaryPort = startPyPorts;
    const response = await fetch('http://localhost:' + primaryPort + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userMsg })
    });
    const result = await response.json();
    return res.json({ reply: result.reply });
  } catch (err) {
    return res.json({ reply: 'Error calling main python instance: ' + err.toString() });
  }
});
app.post('/api/reason', async (req, res) => {
  const userMsg = req.body.message || '';
  try {
    const reasonReplies = [];
    for (let i = 0; i < numInstances; i++) {
      const pPort = startPyPorts + i;
      let success = false;
      let attempt = 0;
      let resp = null;
      while (!success && attempt < 5) {
        try {
          resp = await fetch('http://localhost:' + pPort + '/reason', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userMsg })
          });
          success = true;
        } catch (error) {
          attempt++;
          await new Promise(r => setTimeout(r, 500));
        }
      }
      if (resp) {
        const j = await resp.json();
        reasonReplies.push(j.reply);
      } else {
        reasonReplies.push("Instance " + pPort + " unavailable.");
      }
    }
    let final = "";
    let freq = {};
    let bestCount = 0;
    for (const r of reasonReplies) {
      freq[r] = (freq[r] || 0) + 1;
      if (freq[r] > bestCount) { bestCount = freq[r]; final = r; }
    }
    final = final || reasonReplies[0] || "";
    return res.json({ reply: final, reasonReplies: reasonReplies });
  } catch (err) {
    return res.json({ reply: '', reasonReplies: [], error: err.toString() });
  }
});
app.post('/api/set_tuning', async (req, res) => {
  const tuning = req.body || {};
  try {
    const responses = [];
    for (let i = 0; i < numInstances; i++) {
      const pPort = startPyPorts + i;
      const resp = await fetch('http://localhost:' + pPort + '/set_tuning', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tuning)
      });
      responses.push(await resp.json());
    }
    return res.json({ status: "ok", responses: responses });
  } catch (err) {
    return res.json({ status: "error", error: err.toString() });
  }
});
app.get('/api/status', async (req, res) => {
  const statuses = [];
  for (let i = 0; i < numInstances; i++) {
    const pPort = startPyPorts + i;
    try {
      const resp = await fetch('http://localhost:' + pPort + '/status');
      const j = await resp.json();
      statuses.push({ port: pPort, status: j.status });
    } catch (err) {
      statuses.push({ port: pPort, status: "error" });
    }
  }
  return res.json({ statuses: statuses });
});
app.listen(aggregatorPort, () => {
  console.log("Aggregator Node server on port " + aggregatorPort);
  console.log("Front-end available at http://localhost:" + aggregatorPort + "/index.html");
  console.log("Spawned " + $NumInstances + " python processes on ports " + $startPyPorts + " to " + ($startPyPorts + $NumInstances - 1));
});
"@ | Set-Content -Encoding UTF8 -Path "server.js"

@"
fastapi
uvicorn
transformers
torch
numpy
"@ | Set-Content -Encoding UTF8 -Path "requirements.txt"

if (!(Test-Path "package.json")) { npm init -y | Out-Null }
npm install express --save | Out-Null
npm install node-fetch@2 --save | Out-Null

Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Deployment complete."
Write-Host "Aggregator will run on port 3000 and spawn $NumInstances python instances on ports 8100 to " + (8100 + $NumInstances - 1)
Write-Host "Press Ctrl+C to stop the aggregator."
Write-Host "Launching aggregator Node server now..."
node server.js

Pop-Location
