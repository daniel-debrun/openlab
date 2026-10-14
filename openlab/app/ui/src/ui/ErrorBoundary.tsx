/**
 * A render error must not blank the app.
 *
 * One bad payload — a `null` where a number was assumed, a cell kind the screen
 * did not expect — would otherwise take the whole shell to a white page with the
 * detail only in the console. This is a local-first app, so the user is also the
 * operator: the message they need is the stack, shown to them.
 */
import { Component, ErrorInfo, ReactNode } from "react";
import { Button } from "./basics";

interface Props {
  children: ReactNode;
  /** Shown above the message, e.g. which panel failed. */
  where?: string;
  /** Remount key: changing it clears the error. */
  resetKey?: unknown;
}

interface State {
  error: Error | null;
  stack?: string;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[openlab] render error", error, info.componentStack);
    this.setState({ stack: info.componentStack ?? undefined });
  }

  componentDidUpdate(prev: Props) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null, stack: undefined });
    }
  }

  render() {
    const { error, stack } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="pane-pad">
        <div className="ui-error" role="alert">
          <div className="ui-error-context">
            this panel failed to render{this.props.where ? ` (${this.props.where})` : ""}
          </div>
          <div className="ui-error-msg">{error.message}</div>
          <div className="ui-error-line">
            <span className="ui-error-tag">meaning</span>a bug in the interface, not in your
            mathematics — nothing was computed or written, and the ledger is untouched
          </div>
          <div className="ui-error-actions">
            <Button
              size="sm"
              variant="primary"
              onClick={() => this.setState({ error: null, stack: undefined })}
            >
              Try again
            </Button>
            <Button size="sm" onClick={() => window.location.reload()}>
              Reload
            </Button>
          </div>
          {stack && (
            <details className="ui-error-stack">
              <summary>component stack</summary>
              <pre>{stack}</pre>
            </details>
          )}
        </div>
      </div>
    );
  }
}
