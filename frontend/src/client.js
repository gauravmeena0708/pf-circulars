export const session = crypto.randomUUID().replaceAll("-", "");
let token = "";
export const setToken = (value) => {
  token = value.trim();
};
export function headers(extra = {}) {
  return {
    "X-Workspace-Session": session,
    ...(token ? { "X-HF-Token": token } : {}),
    ...extra,
  };
}
export async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: headers(options.headers),
  });
  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {}
    throw Error(
      typeof detail === "string"
        ? detail
        : `Request failed (${response.status}).`,
    );
  }
  return response;
}
export async function json(path, options = {}) {
  return (await request(path, options)).json();
}
export const post = (value) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(value),
});
export async function saveResponse(response, fallback = "download") {
  const disposition = response.headers.get("content-disposition") || "";
  const match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  const name = match ? decodeURIComponent(match[1]) : fallback;
  const url = URL.createObjectURL(await response.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export async function stream(path, body, onText, signal) {
  const response = await request(path, { ...post(body), signal });
  const reader = response.body.getReader(),
    decoder = new TextDecoder();
  let pending = "";
  while (true) {
    const { done, value } = await reader.read();
    pending += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = pending.split("\n");
    pending = lines.pop();
    for (const line of lines) {
      if (!line) continue;
      const item = JSON.parse(line);
      if (item.error) throw Error(item.error);
      if (item.text) onText(item.text);
    }
    if (done) break;
  }
}
