import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import Markdown from '../src/Markdown.js';

test('answers render GFM tables with headers, aligned cells and source links', () => {
  const text = '| Matter | Limit |\n| :--- | ---: |\n| Briefcase | See [1](#source-1) |\n| Laptop | As specified |';
  const html = renderToStaticMarkup(createElement(Markdown, {children: text}));
  assert.match(html, /<table>/);
  assert.match(html, /<thead>/);
  assert.match(html, /<th[^>]*>Matter<\/th>/);
  assert.equal((html.match(/<tr>/g) || []).length, 3);
  assert.match(html, /text-align:right/);
  assert.match(html, /href="#source-1"/);
});

test('table cells preserve escaped pipes and Hindi; raw HTML remains escaped', () => {
  const text = '| विषय | विवरण |\n| --- | --- |\n| पेंशन | A \\| B |\n\n<script>alert(1)</script>';
  const html = renderToStaticMarkup(createElement(Markdown, {children: text}));
  assert.match(html, /<td>पेंशन<\/td>/);
  assert.match(html, /<td>A \| B<\/td>/);
  assert.doesNotMatch(html, /<script>/);
});
