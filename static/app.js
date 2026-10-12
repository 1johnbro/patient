const messages = document.querySelector("#messages");
const errorBox = document.querySelector("#error");
const confirmation = document.querySelector("#confirmation");
const choices = document.querySelector("#choices");
const form = document.querySelector("#composer");
const input = document.querySelector("#text");
const sendButton = document.querySelector("#send");
const statusLine = document.querySelector("#status");

function addMessage(role, text) {
  const item = document.createElement("div");
  item.className = "message " + role;
  const who = document.createElement("strong");
  who.textContent = role === "user" ? "You" : "Assistant";
  const body = document.createElement("p");
  body.textContent = text;
  item.append(who, body);
  messages.append(item);
  messages.scrollTop = messages.scrollHeight;
}

function showError(text) {
  if (!text) {
    errorBox.hidden = true;
    errorBox.textContent = "";
    return;
  }
  errorBox.hidden = false;
  errorBox.textContent = text;
}

function setBusy(busy) {
  input.disabled = busy;
  sendButton.disabled = busy;
  sendButton.textContent = busy ? "Sending…" : "Send";
  statusLine.textContent = busy ? "Waiting for the assistant…" : "";
  for (const button of document.querySelectorAll("#choices button, #confirmation button")) {
    button.disabled = busy;
  }
}

function showConfirmation(data) {
  if (!data) {
    confirmation.hidden = true;
    return;
  }
  document.querySelector("#confirm-provider").textContent = data.provider;
  document.querySelector("#confirm-date").textContent = data.date;
  document.querySelector("#confirm-time").textContent = data.time;
  document.querySelector("#confirm-place").textContent = [data.specialty, data.location].filter(Boolean).join(" · ");
  confirmation.hidden = false;
}

function showChoices(items) {
  choices.replaceChildren();
  for (const item of items || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.label;
    button.addEventListener("click", function () {
      send(item.text);
    });
    choices.append(button);
  }
}

async function send(text) {
  const message = (text || "").trim();
  if (!message) {
    showError("Type a message first.");
    return;
  }
  showError("");
  addMessage("user", message);
  input.value = "";
  setBusy(true);
  try {
    const response = await fetch("/api/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message }),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "The request failed. Nothing was booked.");
      return;
    }
    addMessage("assistant", data.reply || "");
    if (data.reply && data.reply.indexOf("Nothing was booked") !== -1) {
      showError("The scheduling service could not finish that request. Nothing was booked.");
    }
    showConfirmation(data.confirmation);
    showChoices(data.choices);
  } catch (err) {
    showError("The page could not reach the assistant. Nothing was booked.");
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", function (event) {
  event.preventDefault();
  send(input.value);
});

document.querySelector("#confirm-yes").addEventListener("click", function () {
  send("yes");
});

document.querySelector("#confirm-no").addEventListener("click", function () {
  send("no");
});

input.focus();
