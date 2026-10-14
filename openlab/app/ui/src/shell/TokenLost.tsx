/**
 * The one failure the app cannot recover from by itself.
 *
 * The bearer token is minted per boot unless `OPENLAB_APP_TOKEN` pins it, so
 * restarting the server rotates it and every request 401s with no way for this
 * page to re-acquire it. Without this panel that shows up as a screen full of
 * identical failures and no explanation; with it, the user is told exactly where
 * the new URL is and can paste the token instead of losing the tab.
 */
import { useEffect, useState } from "react";
import { adoptToken, isTokenLost, onTokenLost } from "../api/client";
import { Button, Field } from "../ui";

export default function TokenLost() {
  const [lost, setLost] = useState(isTokenLost());
  const [token, setToken] = useState("");
  useEffect(() => onTokenLost(() => setLost(true)), []);
  if (!lost) return null;

  return (
    <div className="ui-scrim">
      <div className="ui-dialog" role="dialog" aria-modal="true" aria-label="token expired">
        <h3 className="ui-dialog-title">This page's token is no longer valid</h3>
        <div className="ui-dialog-body">
          The workspace server mints a token each time it starts, so a restart invalidates the one this tab
          captured. The terminal that is running <code>openlab app</code> printed a fresh URL — open it, or
          paste its token here.
        </div>
        <Field label="token" hint="the ?t=… part of the printed URL">
          <input value={token} onChange={(e) => setToken(e.target.value)} autoFocus />
        </Field>
        <div className="ui-dialog-actions">
          <Button size="sm" onClick={() => window.location.reload()}>
            Reload
          </Button>
          <Button size="sm" variant="primary" disabled={!token.trim()} onClick={() => adoptToken(token)}>
            Use this token
          </Button>
        </div>
      </div>
    </div>
  );
}
