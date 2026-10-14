/**
 * The entry point. The stylesheet order is the design system's dependency
 * order: tokens define the values, base applies them to elements, ui styles the
 * primitives, app lays out the shell.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/fonts.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./ui/ui.css";
import "./styles/app.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
