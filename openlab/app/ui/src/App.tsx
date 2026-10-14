/**
 * The app: a query client, the two hosts that live outside the tree (dialogs and
 * toasts), and the shell.
 *
 * `retry: false` is deliberate. Every query here is a tool call against a local
 * server: if it refused, it refused for a reason the envelope already explains,
 * and retrying three times only delays showing that reason.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DialogHost, ErrorBoundary, ToastHost } from "./ui";
import Shell from "./shell/Shell";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchOnWindowFocus: false,
      // the socket tells us when something changed; polling would be noise
      staleTime: 10_000,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary where="app">
        <Shell />
      </ErrorBoundary>
      <DialogHost />
      <ToastHost />
    </QueryClientProvider>
  );
}
