import { hindi } from "./messages.hi.js";

// Only interface copy is translated. Source documents and generated answers keep
// their own language; the answer-language preference is independent.
export function translate(locale, text, values = {}) {
  if (typeof text !== "string") return text;
  const key = text.trim();
  const translated = locale === "hi" ? hindi[key] : undefined;
  const copy =
    translated === undefined ? text : text.replace(key, () => translated);
  return copy.replace(/\{(\w+)\}/g, (match, name) =>
    Object.hasOwn(values, name) ? String(values[name]) : match,
  );
}
