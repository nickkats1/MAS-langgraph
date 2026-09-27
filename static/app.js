const STORAGE_KEY = "chat-messages";
const HISTORY_LIMIT = 10;

const form = document.getElementById("composer");
const list = document.getElementById("messages");
const empty = document.getElementById("empty");
const chips = document.getElementById("chips");
const input = form.elements.user_request;
const send = document.getElementById("send");
const fileInputs = [form.elements.image, form.elements.audio];

let messages = load();

function load() {
    try {
        return JSON.parse(sessionStorage.getItem(STORAGE_KEY)) || [];
    } catch {
        return [];
    }
}

function save() {
    try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {}
}

function bubble(role, text, extra = "") {
    const node = document.createElement("div");
    node.className = `msg ${role} ${extra}`.trim();
    node.textContent = text;
    list.append(node);
    list.scrollTop = list.scrollHeight;
    return node;
}

function render() {
    list.querySelectorAll(".msg").forEach((node) => node.remove());
    empty.hidden = messages.length > 0;
    messages.forEach((m) => bubble(m.role, m.content));
}

function renderChips() {
    chips.replaceChildren();
    for (const fileInput of fileInputs) {
        const file = fileInput.files[0];
        if (!file) continue;
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = file.name;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "×";
        remove.onclick = () => { fileInput.value = ""; renderChips(); };
        chip.append(remove);
        chips.append(chip);
    }
}

function grow() {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight}px`;
}

async function ask() {
    const text = input.value.trim();
    if (!text || send.disabled) return;

    const body = new FormData(form);
    for (const name of ["image", "audio"]) {
        if (!body.get(name)?.size) body.delete(name);
    }
    body.set("history", JSON.stringify(messages.slice(-HISTORY_LIMIT)));

    messages.push({ role: "user", content: text });
    save();
    render();
    form.reset();
    renderChips();
    grow();

    send.disabled = true;
    const thinking = bubble("assistant", "", "thinking");
    try {
        const response = await fetch("/ask", { method: "POST", body });
        const data = await response.json();
        if (!response.ok) throw new Error(JSON.stringify(data));
        messages.push({ role: "assistant", content: data.report });
        save();
        render();
    } catch (error) {
        thinking.remove();
        bubble("assistant", `Something went wrong: ${error.message}`, "error");
    } finally {
        send.disabled = false;
        input.focus();
    }
}

form.addEventListener("submit", (event) => { event.preventDefault(); ask(); });
input.addEventListener("input", grow);
input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        ask();
    }
});
fileInputs.forEach((fileInput) => fileInput.addEventListener("change", renderChips));
document.getElementById("new-chat").addEventListener("click", () => {
    messages = [];
    save();
    render();
    input.focus();
});

render();
