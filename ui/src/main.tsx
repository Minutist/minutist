import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// Local-first font bundling via Fontsource — NO CDN (offline-capable).
// Fraunces (display): the `full` axis CSS carries opsz + wght + SOFT + WONK so
// headings can take on a touch of optical-size and soft/wonk character.
// Newsreader (reading body + UI chrome). Italic faces back blockquotes / em.
import "@fontsource-variable/fraunces/full.css";
import "@fontsource-variable/fraunces/full-italic.css";
import "@fontsource-variable/newsreader/index.css";
import "@fontsource-variable/newsreader/standard-italic.css";

// Editorial Ink design tokens + base layer (must precede component CSS).
import "./styles/theme.css";
import "./styles/global.css";

import { App } from "./App";

const rootEl = document.getElementById("root");
if (!rootEl) {
  throw new Error("Root element #root not found");
}

createRoot(rootEl).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
