import { useState, useRef, useEffect } from 'react';
import { Menu, BrainCircuit, Send } from 'lucide-react';
import MessageBubble from './MessageBubble';

const SUGGESTIONS = [
  'Summarize the key findings',
  'What are the main conclusions?',
  'Explain the methodology used',
  'Compare the different approaches',
];

export default function ChatArea({
  messages, isTyping, health, sidebarOpen,
  onToggleSidebar, onSendMessage,
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '48px';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px';
    }
  }, [input]);

  function handleSubmit() {
    if (!input.trim() || isTyping) return;
    onSendMessage(input.trim());
    setInput('');
  }

  function handleSuggestion(q) {
    onSendMessage(q);
  }

  const isConnected = health?.status === 'ok';

  return (
    <div className="main-area">
      {/* Top Bar */}
      <div className="topbar">
        <div className="topbar-left">
          <button className="btn icon" onClick={onToggleSidebar} title="Toggle sidebar (Ctrl+B)">
            <Menu size={18} />
          </button>
          <span className="topbar-title">Chat</span>
        </div>
        <div className="connection-status">
          <div className={`connection-dot ${isConnected ? 'connected' : 'disconnected'}`} />
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>

      {/* Messages */}
      <div className="messages-area">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">
              <BrainCircuit size={28} />
            </div>
            <h2 className="empty-title">How can I help you today?</h2>
            <p className="empty-desc">Ask questions about the documents you've uploaded.</p>
            <div className="suggested-questions">
              {SUGGESTIONS.map((q) => (
                <button key={q} className="suggestion-chip" onClick={() => handleSuggestion(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages-container">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Bar */}
      <div className="input-bar">
        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              id="chat-input"
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your documents... (Ctrl+K)"
              rows={1}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />
          </div>
          <button
            className={`send-btn ${input.trim() ? 'active' : ''}`}
            onClick={handleSubmit}
            disabled={!input.trim() || isTyping}
            title="Send message"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
