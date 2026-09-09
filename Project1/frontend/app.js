const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const attachButton = document.querySelector("#attach-button");
const fileInput = document.querySelector("#file-input");
const attachmentList = document.querySelector("#attachment-list");
const sidebar = document.querySelector(".sidebar");

let selectedFiles = [];
let isSending = false;

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  sendButton.disabled = isSending || !input.value.trim();
}

function appendMessage(role, text = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);
  messages.appendChild(wrapper);
  document.querySelector("#conversation").scrollTop = document.querySelector("#conversation").scrollHeight;
  return bubble;
}

function renderAttachments() {
  attachmentList.replaceChildren();
  selectedFiles.forEach((file, index) => {
    const item = document.createElement("div");
    item.className = "attachment";
    const label = document.createElement("span");
    label.textContent = file.name;
    label.title = file.name;
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.setAttribute("aria-label", `Remove ${file.name}`);
    removeButton.textContent = "×";
    item.append(label, removeButton);
    removeButton.addEventListener("click", () => {
      selectedFiles.splice(index, 1);
      renderAttachments();
    });
    attachmentList.appendChild(item);
  });
}

async function streamResponse(message, assistantBubble) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream, text/plain" },
    body: JSON.stringify({ message }),
  });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  if (!response.body) {
    assistantBubble.textContent = await response.text();
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const payload = line.startsWith("data:") ? line.slice(5).trimStart() : line;
      if (!payload || payload === "[DONE]") continue;
      try {
        const parsed = JSON.parse(payload);
        assistantBubble.textContent += parsed.delta ?? parsed.message ?? parsed.text ?? payload;
      } catch {
        assistantBubble.textContent += payload;
      }
      document.querySelector("#conversation").scrollTop = document.querySelector("#conversation").scrollHeight;
    }
  }
  if (buffer.trim() && buffer.trim() !== "[DONE]") assistantBubble.textContent += buffer.replace(/^data:\s*/, "");
}

async function sendMessage(message) {
  isSending = true;
  resizeInput();
  welcome.hidden = true;
  appendMessage("user", message);
  const assistantBubble = appendMessage("assistant");
  try {
    await streamResponse(message, assistantBubble);
  } catch (error) {
    assistantBubble.parentElement.classList.add("error");
    assistantBubble.textContent = "Unable to reach the RAG service. Please make sure the backend is running and try again.";
    console.error(error);
  } finally {
    isSending = false;
    resizeInput();
  }
}

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || isSending) return;
  input.value = "";
  resizeInput();
  sendMessage(message);
});
attachButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  selectedFiles = [...selectedFiles, ...fileInput.files];
  renderAttachments();
  fileInput.value = "";
});
document.querySelector("#mobile-menu").addEventListener("click", () => sidebar.classList.toggle("open"));
document.querySelector("#collapse-sidebar").addEventListener("click", () => sidebar.classList.remove("open"));
document.querySelector("#voice-button").addEventListener("click", () => {
  input.placeholder = input.placeholder === "Listening…" ? "Ask anything" : "Listening…";
  if (input.placeholder === "Listening…") input.focus();
});
document.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.action === "new-chat") {
      messages.replaceChildren();
      welcome.hidden = false;
      input.focus();
      sidebar.classList.remove("open");
    }
  });
});
resizeInput();
