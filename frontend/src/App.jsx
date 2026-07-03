import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import { getHealth, getDocuments, getStats, uploadDocuments, ingestUrl, deleteDocument, streamChat, clearChatHistory } from './lib/api';

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('ka-theme') || 'dark');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [toasts, setToasts] = useState([]);

  // Theme persistence
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('ka-theme', theme);
  }, [theme]);

  // Load initial data
  useEffect(() => {
    loadHealth();
    loadDocuments();
    loadStats();
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        document.getElementById('chat-input')?.focus();
      }
      if (e.key === 'b' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setSidebarOpen(s => !s);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  function addToast(title, desc, type = 'info') {
    const id = Date.now();
    setToasts(t => [...t, { id, title, desc, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000);
  }

  async function loadHealth() {
    try {
      const data = await getHealth();
      setHealth(data);
    } catch { setHealth(null); }
  }

  async function loadDocuments() {
    try {
      const data = await getDocuments();
      setDocuments(data);
    } catch { /* silent */ }
  }

  async function loadStats() {
    try {
      const data = await getStats();
      setStats(data);
    } catch { /* silent */ }
  }

  async function handleUpload(files) {
    setUploading(true);
    try {
      const result = await uploadDocuments(files);
      addToast('Upload Complete', `Processed ${result.processed} document(s).`);
      loadDocuments();
      loadStats();
    } catch (err) {
      addToast('Upload Failed', err.message, 'error');
    } finally {
      setUploading(false);
    }
  }

  async function handleUrlIngest(url) {
    try {
      await ingestUrl(url);
      addToast('URL Ingested', 'Content has been processed.');
      loadDocuments();
      loadStats();
    } catch (err) {
      addToast('URL Failed', err.message, 'error');
    }
  }

  async function handleDeleteDoc(docId) {
    try {
      await deleteDocument(docId);
      loadDocuments();
      loadStats();
    } catch (err) {
      addToast('Delete Failed', err.message, 'error');
    }
  }

  async function handleSendMessage(content) {
    const userMsg = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    const assistantMsgId = crypto.randomUUID();
    const assistantMsg = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
      sources: [],
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setIsTyping(true);

    try {
      for await (const event of streamChat(content, conversationId)) {
        if (event.event === 'token') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsgId
                ? { ...m, content: m.content + event.data.token }
                : m
            )
          );
        } else if (event.event === 'sources') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsgId
                ? { ...m, sources: event.data.sources }
                : m
            )
          );
        } else if (event.event === 'done') {
          setConversationId(event.data.conversation_id);
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsgId ? { ...m, isStreaming: false } : m
            )
          );
        } else if (event.event === 'error') {
          throw new Error(event.data.error);
        }
      }
    } catch (err) {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMsgId
            ? { ...m, isStreaming: false, content: m.content || 'Sorry, an error occurred.' }
            : m
        )
      );
      addToast('Chat Error', err.message, 'error');
    } finally {
      setIsTyping(false);
    }
  }

  async function handleClearChat() {
    try {
      await clearChatHistory();
      setMessages([]);
      setConversationId(null);
      addToast('Chat Cleared', 'Conversation history removed.');
    } catch { /* silent */ }
  }

  return (
    <div className="app-layout">
      <Sidebar
        open={sidebarOpen}
        documents={documents}
        stats={stats}
        uploading={uploading}
        theme={theme}
        onToggleTheme={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
        onUpload={handleUpload}
        onUrlIngest={handleUrlIngest}
        onDeleteDoc={handleDeleteDoc}
        onClearChat={handleClearChat}
      />
      <ChatArea
        messages={messages}
        isTyping={isTyping}
        health={health}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(s => !s)}
        onSendMessage={handleSendMessage}
      />

      {/* Toasts */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.type === 'error' ? 'error' : ''}`}>
            <div className="toast-title">{t.title}</div>
            {t.desc && <div className="toast-desc">{t.desc}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
