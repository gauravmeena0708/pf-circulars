import { createElement } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Keep research answers and uploaded-file responses on the same Markdown dialect.
export default function Markdown({remarkPlugins = [], ...props}) {
  return createElement(ReactMarkdown, {...props, remarkPlugins: [remarkGfm, ...remarkPlugins]});
}
