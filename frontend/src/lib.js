// Small helpers shared by the page components.
export { json as api, post } from "./client";

export const dateText = (v) => {
  if (!v) return "Not established";
  const value = /^\d{2}\/\d{2}\/\d{4}$/.test(v)
    ? v.split("/").reverse().join("-")
    : v;
  const d = new Date(value + "T00:00:00");
  return Number.isNaN(d.getTime())
    ? v
    : d.toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      });
};

export function download(name, text, type = "text/markdown") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// The API already rewrites other citation styles to [n]; this turns [n] into a link
// that the answer renderer shows as a source button. A marker that is already a
// link, such as [1](#source-1) in a table, is left alone.
export const linkCitations = (answer) =>
  (answer || "").replace(/\[(\d+)\](?!\()/g, "[$1](#source-$1)");
