import { useState, useRef } from 'react';
import { BrainCircuit, UploadCloud, Link, Trash2, FileText, Sun, Moon, Activity } from 'lucide-react';

export default function Sidebar({
  open, documents, stats, uploading, theme,
  onToggleTheme, onUpload, onUrlIngest, onDeleteDoc, onClearChat,
}) {
  const [urlVal, setUrlVal] = useState('');
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  function handleFiles(fileList) {
    if (!fileList || fileList.length === 0) return;
    onUpload(Array.from(fileList));
  }

  function handleUrlSubmit() {
    if (!urlVal.trim()) return;
    onUrlIngest(urlVal.trim());
    setUrlVal('');
  }

  return (
    <div className={`sidebar ${open ? '' : 'collapsed'}`}>
      {/* Header */}
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <BrainCircuit size={18} />
        </div>
        <div>
          <div className="sidebar-title">Knowledge Assistant</div>
          <div className="sidebar-subtitle">AI-powered document Q&A</div>
        </div>
      </div>

      {/* Content */}
      <div className="sidebar-content">
        {/* Upload Zone */}
        <div>
          <div className="sidebar-section-title">Ingest Knowledge</div>
          <div
            className={`upload-zone ${dragging ? 'dragging' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              handleFiles(e.dataTransfer.files);
            }}
          >
            <div className="upload-zone-icon">
              <UploadCloud size={24} />
            </div>
            <div className="upload-zone-text">
              <strong>Click to upload</strong> or drag & drop
            </div>
            <div className="upload-zone-formats">PDF, TXT, MD</div>
            <input
              type="file"
              multiple
              accept=".pdf,.txt,.md,.markdown,.rst,.csv"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={(e) => handleFiles(e.target.files)}
            />
          </div>

          {uploading && (
            <div className="upload-progress">
              <div className="upload-file-item">
                <div className="spinner" style={{ width: 14, height: 14 }}>
                  <Activity size={14} />
                </div>
                Processing documents...
              </div>
            </div>
          )}

          <div className="url-input-row">
            <input
              className="input"
              placeholder="https://..."
              value={urlVal}
              onChange={(e) => setUrlVal(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleUrlSubmit()}
            />
            <button className="btn icon" onClick={handleUrlSubmit} title="Add URL">
              <Link size={16} />
            </button>
          </div>
        </div>

        {/* Document List */}
        <div>
          <div className="sidebar-section-title">
            Documents ({documents.length})
            {uploading && <span className="status-dot processing" style={{ marginLeft: 6, display: 'inline-block' }} />}
          </div>

          {documents.length === 0 ? (
            <div className="empty-docs">No documents yet. Upload files to get started.</div>
          ) : (
            <div className="doc-list">
              {documents.map((doc) => (
                <div key={doc.id} className="doc-item">
                  <FileText size={16} className="doc-icon" />
                  <div className="doc-info">
                    <div className="doc-name" title={doc.name}>{doc.name}</div>
                    <div className="doc-meta">
                      <span className={`status-dot ${doc.status}`} />
                      <span>{doc.status}</span>
                      <span>•</span>
                      <span>{doc.chunks} chunks</span>
                    </div>
                  </div>
                  <button
                    className="doc-delete-btn"
                    onClick={() => onDeleteDoc(doc.id)}
                    title="Delete document"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        {stats && (
          <div className="sidebar-stats">
            <span>Chunks: {stats.total_chunks}</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Activity size={12} /> {stats.status}
            </span>
          </div>
        )}
        <div className="sidebar-actions">
          <button className="btn flex-1" onClick={onClearChat}>
            <Trash2 size={14} /> Clear Chat
          </button>
          <button className="btn icon" onClick={onToggleTheme} title="Toggle theme">
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>
    </div>
  );
}
