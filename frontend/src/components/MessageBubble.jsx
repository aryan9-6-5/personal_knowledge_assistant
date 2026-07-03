import { useState } from 'react';
import { BrainCircuit, FileText, ChevronDown, ChevronRight } from 'lucide-react';
import Markdown from 'react-markdown';

export default function MessageBubble({ message }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const { role, content, sources, isStreaming, timestamp } = message;

  const timeStr = timestamp
    ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';

  return (
    <div className={`message-row ${role}`}>
      <div className="message-wrapper">
        {role === 'assistant' && (
          <div className="avatar assistant">
            <BrainCircuit size={16} />
          </div>
        )}

        <div className="message-content-wrapper">
          <div className={`message-bubble ${role}`}>
            {content ? (
              role === 'assistant' ? (
                <Markdown>{content}</Markdown>
              ) : (
                content
              )
            ) : (
              isStreaming && (
                <div className="typing-indicator">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
              )
            )}
          </div>

          {/* Sources */}
          {sources && sources.length > 0 && !isStreaming && (
            <div className="sources-section">
              <div className="sources-header" onClick={() => setSourcesOpen(!sourcesOpen)}>
                <FileText size={14} />
                Sources ({sources.length})
                {sourcesOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              </div>
              {sourcesOpen && (
                <div className="sources-list">
                  {sources.map((s, i) => {
                    const pct = Math.round(s.relevance_score * 100);
                    const level = pct >= 70 ? 'high' : pct >= 40 ? 'medium' : 'low';
                    return (
                      <div key={i} className="source-card">
                        <div className="source-card-title">
                          {s.source} <span className="source-card-page">p.{s.page}</span>
                        </div>
                        <div className="source-card-content">{s.content}</div>
                        <div className="relevance-bar">
                          <div className={`relevance-fill ${level}`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {timeStr && <div className="message-timestamp">{timeStr}</div>}
        </div>
      </div>
    </div>
  );
}
