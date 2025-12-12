import { useState } from 'react';
import { api } from '../api';

export function ImageStudio() {
  const [mode, setMode] = useState('generate'); // generate | edit
  const [prompt, setPrompt] = useState('A futuristic cityscape at sunset, synthwave style');
  const [enhancedPrompt, setEnhancedPrompt] = useState('');
  const [size, setSize] = useState('1024x1024');
  const [provider, setProvider] = useState('nvidia');
  const [imgUrl, setImgUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [enhancing, setEnhancing] = useState(false);
  const [surprising, setSurprising] = useState(false);
  const [uploading, setUploading] = useState(false);
  
  /**
   * Extract a readable error message from various error types.
   * Handles Error objects, API response errors, and unknown error types.
   */
  const getErrorMessage = (e: unknown): string => {
    if (typeof e === 'string') return e;
    if (e instanceof Error) return e.message;
    if (e && typeof e === 'object') {
      const errorObj = e as any;
      // Try multiple paths to extract error info
      return errorObj.response?.data?.error || 
             errorObj.response?.data?.message ||
             errorObj.message || 
             JSON.stringify(e);
    }
    return String(e);
  };
  
  // Edit mode
  const [editFile, setEditFile] = useState<File | null>(null);
  const [editFileUrl, setEditFileUrl] = useState('');

  const presetPrompts = [
    { emoji: '🌆', text: 'A futuristic cityscape at sunset, synthwave style' },
    { emoji: '🐶', text: 'A golden retriever wearing a space helmet, digital art' },
    { emoji: '🍔', text: 'A giant cheeseburger resting on a mountaintop' },
    { emoji: '🎨', text: 'A dense forest painted in watercolor style' }
  ];

  const editPresets = [
    { emoji: '🎨', text: 'Convert to Studio Ghibli animation style' },
    { emoji: '📺', text: 'Turn the subjects into Simpsons characters' },
    { emoji: '☃️', text: 'Turn the subjects into South Park characters' },
    { emoji: '💥', text: 'Convert to comic book style with dramatic effects' }
  ];

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const { promise } = api.image.generate(prompt, enhancedPrompt, size, provider);
      const res = await promise;
      setImgUrl(`/api/files/${res.image_path}`);
    } catch (e) {
      // Ignore abort errors (request cancelled)
      if (e instanceof Error && (e as any).aborted) {
        return;
      }
      console.error('Error generating image:', e);
      alert('Error generating image: ' + getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const handleEnhance = async () => {
    setEnhancing(true);
    try {
      const { promise } = api.image.enhance(prompt, provider);
      const res = await promise;
      setEnhancedPrompt(res.enhanced_prompt);
    } catch (e) {
      // Ignore abort errors (request cancelled)
      if (e instanceof Error && (e as any).aborted) {
        return;
      }
      console.error('Error enhancing prompt:', e);
      alert('Error enhancing prompt: ' + getErrorMessage(e));
    } finally {
      setEnhancing(false);
    }
  };

  const handleSurprise = async () => {
    setSurprising(true);
    try {
      const { promise } = api.image.surprise(provider);
      const res = await promise;
      setPrompt(res.prompt);
      setEnhancedPrompt('');
    } catch (e) {
      // Ignore abort errors (request cancelled)
      if (e instanceof Error && (e as any).aborted) {
        return;
      }
      console.error('Error getting surprise prompt:', e);
      alert('Error getting surprise prompt: ' + getErrorMessage(e));
    } finally {
      setSurprising(false);
    }
  };
  
  const handleUpload = async (file: File) => {
      setUploading(true);
      try {
        const { promise } = api.image.upload(file);
        const res = await promise;
        // Only update state after successful upload
        setEditFile(file);
        setEditFileUrl(res.file_path);
      } catch (e) {
        // Ignore abort errors (request cancelled)
        if (e instanceof Error && (e as any).aborted) {
          return;
        }
        // Surface error to user and reset state to prevent inconsistency
        console.error('Error uploading file:', e);
        alert('Error uploading file: ' + getErrorMessage(e));
        setEditFile(null);
        setEditFileUrl('');
      } finally {
        setUploading(false);
      }
  }

  const handleEdit = async () => {
      if (!editFileUrl) return alert("Upload an image first");
      setLoading(true);
      try {
          const { promise } = api.image.edit(editFileUrl, prompt, enhancedPrompt, size, provider);
          const res = await promise;
          setImgUrl(`/api/files/${res.image_path}`);
      } catch(e) {
          // Ignore abort errors (request cancelled)
          if (e instanceof Error && (e as any).aborted) {
            return;
          }
          console.error('Error editing image:', e);
          alert('Error editing image: ' + getErrorMessage(e)); 
      }
      finally { 
          setLoading(false); 
      }
  }

  return (
    <div className="image-studio">
        <div style={{ marginBottom: '32px', textAlign: 'center' }}>
          <div className="pill-tabs" style={{ display: 'inline-flex' }}>
            <button className={`pill-btn ${mode === 'generate' ? 'active' : ''}`} onClick={() => setMode('generate')}>
              ✨ Generate
            </button>
            <button className={`pill-btn ${mode === 'edit' ? 'active' : ''}`} onClick={() => setMode('edit')}>
              🎨 Edit
            </button>
          </div>
        </div>
        
        <div className="card">
            <div className="card-header">
                <h2 className="card-title">{mode === 'generate' ? '✨ Generate Image' : '🎨 Edit Image'}</h2>
                <select value={provider} onChange={e => setProvider(e.target.value)} style={{ width: 'auto' }}>
                    <option value="nvidia">NVIDIA (Stable Diffusion 3)</option>
                </select>
            </div>
            
            {mode === 'edit' && (
                <div style={{ 
                  marginBottom: '24px'
                }}>
                    <label style={{ 
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      fontSize: '12px', 
                      fontWeight: 600,
                      marginBottom: '12px',
                      color: 'var(--text-primary)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px'
                    }}>
                      <span style={{ fontSize: '16px' }}>🖼️</span>
                      <span>Upload Image to Edit</span>
                    </label>
                    
                    <label style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '16px',
                      padding: '40px 24px',
                      background: editFile 
                        ? 'linear-gradient(135deg, rgba(212, 175, 55, 0.08) 0%, rgba(22, 25, 32, 0.6) 100%)'
                        : 'rgba(0, 0, 0, 0.2)',
                      border: editFile 
                        ? '2px solid rgba(212, 175, 55, 0.3)'
                        : '2px dashed var(--border-medium)',
                      borderRadius: 'var(--border-radius-md)',
                      cursor: uploading ? 'wait' : 'pointer',
                      transition: 'all var(--transition-base)',
                      opacity: uploading ? 0.6 : 1,
                    }}
                    onMouseOver={e => {
                      if (!editFile) {
                        e.currentTarget.style.borderColor = 'var(--accent-gold)';
                        e.currentTarget.style.background = 'rgba(0, 0, 0, 0.3)';
                      }
                    }}
                    onMouseOut={e => {
                      if (!editFile) {
                        e.currentTarget.style.borderColor = 'var(--border-medium)';
                        e.currentTarget.style.background = 'rgba(0, 0, 0, 0.2)';
                      }
                    }}
                    >
                      <div style={{ fontSize: '48px', opacity: uploading ? 0.6 : (editFile ? 1 : 0.4) }}>
                        {uploading ? '⏳' : (editFile ? '✓' : '📤')}
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        {uploading ? (
                          <>
                            <p style={{ 
                              fontSize: '15px', 
                              color: 'var(--accent-gold)', 
                              margin: 0,
                              fontWeight: 600
                            }}>
                              Uploading...
                            </p>
                            <p style={{
                              fontSize: '13px',
                              color: 'var(--text-muted)',
                              marginTop: '8px',
                              fontStyle: 'italic'
                            }}>
                              Please wait
                            </p>
                          </>
                        ) : editFile ? (
                          <>
                            <p style={{ 
                              fontSize: '15px', 
                              color: 'var(--accent-gold)', 
                              margin: 0,
                              fontWeight: 600
                            }}>
                              {editFile.name}
                            </p>
                            <p style={{
                              fontSize: '13px',
                              color: 'var(--text-muted)',
                              marginTop: '8px',
                              fontStyle: 'italic'
                            }}>
                              Click to change image
                            </p>
                          </>
                        ) : (
                          <>
                            <p style={{ 
                              fontSize: '15px', 
                              color: 'var(--text-primary)', 
                              margin: 0,
                              fontWeight: 500
                            }}>
                              Click to upload an image
                            </p>
                            <p style={{
                              fontSize: '13px',
                              color: 'var(--text-muted)',
                              marginTop: '8px'
                            }}>
                              PNG, JPG, WEBP up to 10MB
                            </p>
                          </>
                        )}
                      </div>
                      <input 
                        type="file" 
                        onChange={e => e.target.files && handleUpload(e.target.files[0])}
                        accept="image/*"
                        disabled={uploading}
                        style={{ display: 'none' }}
                      />
                    </label>
                </div>
            )}
            
            <div style={{ marginBottom: '20px' }}>
                <label style={{ 
                  display: 'block', 
                  fontSize: '14px', 
                  fontWeight: 500,
                  marginBottom: '10px',
                  color: 'var(--text-secondary)'
                }}>
                  Prompt
                </label>
                <textarea 
                    value={prompt} 
                    onChange={e => setPrompt(e.target.value)} 
                    rows={3} 
                    placeholder="Describe the image you want to create..."
                />
            </div>
            
            <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
                <select value={size} onChange={e => setSize(e.target.value)} style={{ width: 'auto', minWidth: '140px' }}>
                    <option value="1024x1024">1024×1024 (Square)</option>
                    <option value="1024x1536">1024×1536 (Portrait)</option>
                    <option value="1536x1024">1536×1024 (Landscape)</option>
                </select>
                <div className="tooltip-container" style={{ flex: 1, minWidth: '140px' }}>
                  <button className="btn btn-secondary" onClick={handleEnhance} disabled={enhancing} style={{ width: '100%' }}>
                    {enhancing ? <div className="loader"></div> : '✨ Enhance Prompt'}
                  </button>
                  <span className="tooltip-text">Use AI to expand and improve your prompt with more detail</span>
                </div>
                <div className="tooltip-container" style={{ flex: 1, minWidth: '140px' }}>
                  <button className="btn btn-secondary" onClick={handleSurprise} disabled={surprising} style={{ width: '100%' }}>
                    {surprising ? <div className="loader"></div> : '🎁 Surprise Me'}
                  </button>
                  <span className="tooltip-text">Generate a random creative prompt to inspire you</span>
                </div>
            </div>
            
            {/* Preset Buttons */}
            {mode === 'generate' && (
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', 
                gap: '8px', 
                marginBottom: '20px' 
              }}>
                {presetPrompts.map((preset, i) => (
                  <div key={i} className="tooltip-container" style={{ width: '100%' }}>
                    <button
                      onClick={() => setPrompt(preset.text)}
                      className="btn btn-secondary"
                      style={{ 
                        fontSize: '20px',
                        padding: '12px 16px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: '100%'
                      }}
                    >
                      {preset.emoji}
                    </button>
                    <span className="tooltip-text">{preset.text}</span>
                  </div>
                ))}
              </div>
            )}
            
            {mode === 'edit' && (
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', 
                gap: '8px', 
                marginBottom: '20px' 
              }}>
                {editPresets.map((preset, i) => (
                  <div key={i} className="tooltip-container" style={{ width: '100%' }}>
                    <button
                      onClick={() => setPrompt(preset.text)}
                      className="btn btn-secondary"
                      style={{ 
                        fontSize: '20px',
                        padding: '12px 16px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: '100%'
                      }}
                    >
                      {preset.emoji}
                    </button>
                    <span className="tooltip-text">{preset.text}</span>
                  </div>
                ))}
              </div>
            )}
            
            {enhancedPrompt && (
                <div style={{ 
                  marginBottom: '20px',
                  padding: '16px',
                  background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.08) 0%, rgba(22, 25, 32, 0.6) 100%)',
                  borderRadius: 'var(--border-radius-md)',
                  border: '1px solid rgba(212, 175, 55, 0.2)'
                }}>
                    <label style={{ 
                      display: 'block',
                      fontSize: '13px',
                      fontWeight: 600,
                      marginBottom: '8px',
                      color: 'var(--accent-gold)',
                      textTransform: 'uppercase',
                      letterSpacing: '1px'
                    }}>
                      Enhanced Prompt
                    </label>
                    <textarea 
                        value={enhancedPrompt} 
                        onChange={e => setEnhancedPrompt(e.target.value)} 
                        rows={3}
                        style={{ borderColor: 'var(--accent-gold)' }}
                    />
                </div>
            )}
            
            <button 
                className="btn btn-primary" 
                style={{ width: '100%', fontSize: '16px', padding: '16px' }} 
                onClick={mode === 'generate' ? handleGenerate : handleEdit}
                disabled={loading}
            >
                {loading ? (
                  <>
                    <div className="loader"></div>
                    <span style={{ marginLeft: '12px' }}>
                      {mode === 'generate' ? 'Generating...' : 'Editing...'}
                    </span>
                  </>
                ) : (
                  mode === 'generate' ? '✨ Generate Image' : '🎨 Edit Image'
                )}
            </button>
            
            {imgUrl && (
                <div style={{ 
                  marginTop: '32px', 
                  textAlign: 'center',
                  padding: '24px',
                  background: 'rgba(255, 255, 255, 0.02)',
                  borderRadius: 'var(--border-radius-md)',
                  border: '1px solid var(--border-subtle)'
                }}>
                    <img 
                      src={imgUrl} 
                      alt="Result" 
                      style={{ 
                        maxWidth: '100%', 
                        borderRadius: 'var(--border-radius-md)', 
                        boxShadow: '0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px var(--border-subtle)',
                        marginBottom: '16px'
                      }} 
                    />
                    <a 
                      href={imgUrl} 
                      download 
                      className="btn btn-secondary" 
                      style={{ minWidth: '160px' }}
                    >
                      ⬇️ Download Image
                    </a>
                </div>
            )}
        </div>
    </div>
  );
}
