/**
 * A person's act from this page: pause, resume or stop an agent, approve or deny an inbox item.
 *
 * The act is a tool marked `human` (agent_control, inbox_decide): the server refuses it unless the request
 * carries the reviewer secret from the URL `openlab app` printed (`client.ts` sends it whenever this page has
 * it), which the agent terminal never receives.
 *
 * THE PASSPHRASE HOOK. A later version signs a person's act with the person's key: the server then refuses a
 * call without the passphrase (an envelope whose `error` or `meaning` says "passphrase", or `needs:
 * "passphrase"`). This helper answers that by asking for the passphrase in a second, hidden field and sending
 * the same call again with `passphrase` added; a wrong one is refused the same way and asked again, at most
 * three times. Nothing here stores it.
 */
import { callTool, type Args, type Envelope, type Fail } from "../api/tools";
import { form, toast, type FormField } from "../ui";
import type { ReactNode } from "react";

export interface PersonActSpec {
  title: ReactNode;
  body?: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  /** the reason's placeholder */
  placeholder?: string;
  /** more answers the act needs (a screen conflict's pick), asked before the reason; a number-like answer is
   *  sent as a number */
  fields?: FormField[];
}

/** The server wants the person's passphrase for this act. */
export function needsPassphrase(env: Fail): boolean {
  const said = `${env.error} ${env.meaning ?? ""}`.toLowerCase();
  return (env as Fail & { needs?: string }).needs === "passphrase" || said.includes("passphrase");
}

/**
 * Ask for the reason (and the passphrase when the server asks for one), then call `tool` with `args` and the
 * reason. Returns the envelope, or null when the person cancelled. A refusal is shown as a toast.
 */
export async function personAct<T>(tool: string, args: Args, spec: PersonActSpec): Promise<Envelope<T> | null> {
  const asked = await form({
    title: spec.title,
    body: spec.body,
    fields: [...(spec.fields ?? []), { key: "reason", label: "Reason (recorded with the act)", placeholder: spec.placeholder }],
    confirmLabel: spec.confirmLabel,
    danger: spec.danger,
  });
  const reason = asked?.reason?.trim();
  if (!asked || !reason) return null;
  for (const f of spec.fields ?? []) {
    const v = asked[f.key];
    if (v !== undefined && v !== "") args = { ...args, [f.key]: /^-?\d+(\.\d+)?$/.test(v) ? Number(v) : v };
  }
  let env = await callTool<T>(tool, { ...args, reason });
  for (let tries = 0; !env.ok && needsPassphrase(env) && tries < 3; tries++) {
    const pass = await form({
      title: "Your passphrase",
      body: env.meaning ?? "This act is signed with your key: give its passphrase.",
      fields: [{ key: "passphrase", label: "Passphrase", secret: true }],
      confirmLabel: spec.confirmLabel,
      danger: spec.danger,
    });
    if (!pass?.passphrase) return null;
    env = await callTool<T>(tool, { ...args, reason, passphrase: pass.passphrase });
  }
  if (!env.ok) toast.err(env, { context: `${tool} was refused` });
  return env;
}
