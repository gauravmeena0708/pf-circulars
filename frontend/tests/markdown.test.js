import { expect, test } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "../src/Markdown.js";
import { dateText, linkCitations } from "../src/lib.js";

test("answers render GFM tables with headers, aligned cells and source links", () => {
  const text =
    "| Matter | Limit |\n| :--- | ---: |\n| Briefcase | See [1](#source-1) |\n| Laptop | As specified |";
  const html = renderToStaticMarkup(createElement(Markdown, { children: text }));
  expect(html).toMatch(/<table>/);
  expect(html).toMatch(/<thead>/);
  expect(html).toMatch(/<th[^>]*>Matter<\/th>/);
  expect((html.match(/<tr>/g) || []).length).toBe(3);
  expect(html).toMatch(/text-align:right/);
  expect(html).toMatch(/href="#source-1"/);
});

test("table cells preserve escaped pipes and Hindi; raw HTML remains escaped", () => {
  const text =
    "| विषय | विवरण |\n| --- | --- |\n| पेंशन | A \\| B |\n\n<script>alert(1)</script>";
  const html = renderToStaticMarkup(createElement(Markdown, { children: text }));
  expect(html).toMatch(/<td>पेंशन<\/td>/);
  expect(html).toMatch(/<td>A \| B<\/td>/);
  expect(html).not.toMatch(/<script>/);
});

test("numbered citations become source links, once", () => {
  expect(linkCitations("Relief is 58% [2] and [12].")).toBe(
    "Relief is 58% [2](#source-2) and [12](#source-12).",
  );
  // A marker that is already a link is not wrapped again.
  expect(linkCitations("See [1](#source-1)")).toBe("See [1](#source-1)");
  expect(linkCitations("A list [a, b] and a [link](https://example.com)")).toBe(
    "A list [a, b] and a [link](https://example.com)",
  );
  expect(linkCitations("")).toBe("");
  expect(linkCitations(undefined)).toBe("");
});

test("dates are shown the same way whichever format the source used", () => {
  expect(dateText("2025-04-03")).toBe("03 Apr 2025");
  expect(dateText("03/04/2025")).toBe("03 Apr 2025");
  expect(dateText("")).toBe("Not established");
  expect(dateText("undated")).toBe("undated");
});
