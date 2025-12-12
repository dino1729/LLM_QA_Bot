import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api';

interface RetrievedMemory {
  content: string;
  score: number;
  metadata: any;
}

export function MemoryPalace() {
  const [provider, setProvider] = useState('litellm');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [model, setModel] = useState('');
  const [loadingModels, setLoadingModels] = useState(false);
  
  const [history, setHistory] = useState<string[][]>([]);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [retrievedMemories, setRetrievedMemories] = useState<RetrievedMemory[]>([]);
  const [resetting, setResetting] = useState(false);

  // Fetch models
  useEffect(() => {
    let cancelled = false;
    let cancelFn: (() => void) | null = null;
    
    const fetchModels = async () => {
      setLoadingModels(true);
      const apiCall = api.models.list(provider);
      cancelFn = apiCall.cancel;
      
      try {
        const response = await apiCall.promise;
        if (!cancelled) {
          setAvailableModels(response.models || []);
          if (response.models && response.models.length > 0) {
            setModel(response.models[0]);
          }
        }
      } catch (error) {
        if (error instanceof Error && (error as any).aborted) return;
        if (!cancelled) {
          console.error('Error fetching models:', error);
          setAvailableModels([]);
        }
      } finally {
        if (!cancelled) setLoadingModels(false);
        cancelFn = null;
      }
    };
    
    fetchModels();
    
    return () => {
      cancelled = true;
      if (cancelFn) cancelFn();
    };
  }, [provider]);

  const handleProviderChange = (newProvider: string) => {
    setProvider(newProvider);
    setModel('');
  };

  const handleSend = async () => {
    if (!message.trim() || !model) return;
    
    const userMessage = message;
    setHistory(prev => [...prev, [userMessage, 'Searching memories...']]);
    setMessage('');
    setLoading(true);
    setRetrievedMemories([]);
    
    try {
      const modelName = `${provider.toUpperCase()}:${model}`;
      
      // 1. Search memories first
      const searchCall = api.memoryPalace.search(userMessage, modelName);
      const searchRes = await searchCall.promise;
      setRetrievedMemories(searchRes.results);
      
      // 2. Stream answer
      setHistory(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = [userMessage, '']; // Clear "Searching..."
        return updated;
      });
      
      let currentAnswer = '';
      const { promise } = api.memoryPalace.askStream(userMessage, history, modelName, (chunk) => {
        currentAnswer += chunk;
        setHistory(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = [userMessage, currentAnswer];
          return updated;
        });
      });
      
      await promise;
      
    } catch (e) {
      if (e instanceof Error && (e as any).aborted) return;
      setHistory(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = [userMessage, 'Error: ' + e];
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (!model || !confirm("Are you sure you want to delete all memories for this model?")) return;
    setResetting(true);
    try {
      const modelName = `${provider.toUpperCase()}:${model}`;
      const res = await api.memoryPalace.reset(modelName).promise;
      alert(res.status);
      setHistory([]);
      setRetrievedMemories([]);
    } catch (e) {
      alert("Error resetting: " + e);
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="memory-palace">
      <div className="card">
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 className="card-title" style={{ margin: 0 }}>🧠 Memory Palace</h2>
          <div style={{ display: 'flex', gap: '8px' }}>
            <select 
              value={provider} 
              onChange={e => handleProviderChange(e.target.value)} 
              style={{ minWidth: '120px', fontSize: '14px' }}
            >
              <option value="litellm">LiteLLM</option>
              <option value="ollama">Ollama</option>
            </select>
            <select 
              value={model} 
              onChange={e => setModel(e.target.value)} 
              style={{ minWidth: '180px', fontSize: '14px' }}
              disabled={loadingModels || availableModels.length === 0}
            >
              {loadingModels ? <option>Loading...</option> : availableModels.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <button 
              className="btn btn-secondary"
              onClick={handleReset}
              disabled={resetting}
              style={{ fontSize: '12px', padding: '8px 12px' }}
            >
              Reset Memory
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '20px', height: '600px' }}>
          {/* Chat Area */}
          <div style={{ flex: 2, display: 'flex', flexDirection: 'column' }}>
            <div className="chat-window" style={{ flex: 1, overflowY: 'auto', marginBottom: '20px', padding: '20px' }}>
              {history.map((msg, i) => (
                <div key={i} style={{ marginBottom: '24px' }}>
                  <div style={{ textAlign: 'right', marginBottom: '8px' }}>
                    <span style={{ 
                      background: 'var(--accent-gold)', 
                      color: 'black', 
                      padding: '8px 16px', 
                      borderRadius: '16px 16px 2px 16px', 
                      fontWeight: 500,
                      display: 'inline-block'
                    }}>
                      {msg[0]}
                    </span>
                  </div>
                  <div style={{ textAlign: 'left' }}>
                    <div style={{ 
                      background: 'rgba(255,255,255,0.05)', 
                      padding: '12px 16px', 
                      borderRadius: '16px 16px 16px 2px',
                      border: '1px solid var(--border-subtle)'
                    }}>
                      <div className="prose"><ReactMarkdown>{msg[1]}</ReactMarkdown></div>
                    </div>
                  </div>
                </div>
              ))}
              {history.length === 0 && (
                <div className="empty-state" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                  Ask questions about your saved memories...
                </div>
              )}
            </div>
            
            <div style={{ display: 'flex', gap: '12px' }}>
              <input 
                type="text" 
                value={message}
                onChange={e => setMessage(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !loading && handleSend()}
                placeholder="Search your memories..."
                disabled={loading}
                style={{ flex: 1 }}
              />
              <button 
                className="btn btn-primary"
                onClick={handleSend}
                disabled={loading || !message.trim()}
              >
                {loading ? 'Thinking...' : 'Ask'}
              </button>
            </div>
          </div>

          {/* Retrieved Context Area */}
          <div style={{ 
            flex: 1, 
            background: 'rgba(0,0,0,0.2)', 
            borderRadius: 'var(--border-radius-md)', 
            padding: '16px',
            overflowY: 'auto',
            border: '1px solid var(--border-subtle)'
          }}>
            <h3 style={{ marginTop: 0, fontSize: '16px', color: 'var(--text-secondary)' }}>Retrieved Context</h3>
            {retrievedMemories.length === 0 ? (
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                Relevant memories will appear here when you ask a question.
              </p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {retrievedMemories.map((mem, i) => (
                  <div key={i} style={{ 
                    padding: '12px', 
                    background: 'rgba(255,255,255,0.03)', 
                    borderRadius: '8px',
                    borderLeft: `2px solid var(--accent-gold)`
                  }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                      <span>{mem.metadata?.source_type?.toUpperCase()}</span>
                      <span>Score: {mem.score?.toFixed(2)}</span>
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--text-primary)', maxHeight: '100px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {mem.content}
                    </div>
                    {mem.metadata?.source_ref && (
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        Source: {mem.metadata.source_ref}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

