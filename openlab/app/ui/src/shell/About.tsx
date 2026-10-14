/**
 * "What is this?": one paragraph on what openlab is, and the words the interface uses before it has earned them.
 * A sheet rather than a page, reachable from the header on every screen, because the question arrives the first
 * time you meet each word, not once at the start.
 */
import { GLOSSARY, Sheet } from "../ui";

export default function About({ open, onClose, version }: { open: boolean; onClose: () => void; version?: string }) {
  return (
    <Sheet open={open} onClose={onClose} title="What is openlab?" width={520}>
      <div className="about">
        <p>
          openlab is a research framework and a control plane for the agents doing the research. The work is a{" "}
          <strong>study</strong>: a folder with its question, its plan, its model and a record of every run that
          nobody can quietly rewrite. Your agents and you act on it through the same tools, and every write says
          who made it.
        </p>
        <p>
          The <strong>Agents</strong> view is the tree of every agent working here, with its budget, its spend,
          what it last did and whether it waits on you; the <strong>inbox</strong> holds what only a person may
          decide. <code>openlab watch</code> shows the same tree in a terminal.
        </p>
        {version && <p className="about-version">openlab v{version}</p>}

        <h4 className="panel-h">the words this interface will use at you</h4>
        <dl className="about-glossary">
          {GLOSSARY.map((g) => (
            <div className="about-term" key={g.term}>
              <dt>{g.term}</dt>
              <dd>{g.gloss}</dd>
            </div>
          ))}
        </dl>
      </div>
    </Sheet>
  );
}
