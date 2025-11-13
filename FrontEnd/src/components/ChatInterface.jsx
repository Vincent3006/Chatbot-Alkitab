// ChatInterface.jsx
import React, { useState, useRef, useEffect } from 'react';
import { FaPaperPlane, FaTimes } from 'react-icons/fa'; 
import styles from './ChatInterface.module.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { id: 1, text: "Selamat datang! Silakan ajukan pertanyaan Anda tentang Alkitab.", sender: 'ai', sources: [] },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [modalSources, setModalSources] = useState(null);
  const [expandedSourceIndex, setExpandedSourceIndex] = useState(null);
  const chatMessagesRef = useRef(null);

  useEffect(() => {
    chatMessagesRef.current?.scrollTo(0, chatMessagesRef.current.scrollHeight);
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (input.trim() === '' || isLoading) return;
    if (modalSources) closeModal();

    const userMessage = { id: Date.now(), text: input, sender: 'human' };
    setMessages(prev => [...prev, userMessage]);
    
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ question: currentInput }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      
      const data = await response.json();
      const aiMessage = {
        id: Date.now() + 1,
        text: data.answer || "Maaf, saya tidak dapat menemukan jawaban.",
        sender: 'ai',
        sources: data.sources || [],
      };
      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error("Error fetching chat response:", error);
      const errorMessage = {
        id: Date.now() + 1,
        text: "Maaf, terjadi kesalahan saat menyambungkan ke server.",
        sender: 'ai',
        sources: [],
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const openModal = (sources) => {
    setModalSources(sources);
  };

  const closeModal = () => {
    setModalSources(null);
    setExpandedSourceIndex(null); 
  };

  const toggleSourceDetail = (index) => {
    setExpandedSourceIndex(prevIndex => (prevIndex === index ? null : index));
  };

  return (
    <div className={styles.chatContainer}>
      {modalSources && (
        <div className={styles.modalOverlay} onClick={closeModal}>
          <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>Sumber Referensi Lengkap</h3>
              <button onClick={closeModal} className={styles.modalCloseButton}>
                <FaTimes />
              </button>
            </div>
            <ul className={styles.sourceList}>
              {modalSources.map((source, index) => (
                <li key={index} className={styles.sourceItem}>
                  <div className={styles.sourceItemHeader} onClick={() => toggleSourceDetail(index)}>
                    <strong>{source.source_range}</strong>
                    <span className={styles.arrow}>{expandedSourceIndex === index ? '▲' : '▼'}</span>
                  </div>
                  {expandedSourceIndex === index && (
                    <div className={styles.sourceDetails}>
                      <p><strong>Isi:</strong> "{source.content}"</p>
                      <p><strong>Skor Relevansi:</strong> {source.score.toFixed(4)}</p>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className={styles.chatMessages} ref={chatMessagesRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`${styles.messageBubble} ${styles[msg.sender]}`}>
            <p>{msg.text}</p>
            {msg.sender === 'ai' && msg.sources && msg.sources.length > 0 && (
              <>
                <div className={styles.messageSource}>
                  <strong>Sumber:</strong> {msg.sources.slice(0, 2).map(s => s.source_range.split(':')[0]).join(', ')}
                  {msg.sources.length > 2 ? '...' : ''}
                </div>
                <div className={styles.sourceExpandIcon} onClick={() => openModal(msg.sources)}>
                  &gt;
                </div>
              </>
            )}
          </div>
        ))}
        {isLoading && (
          <div className={`${styles.messageBubble} ${styles.ai} ${styles.loading}`}>
            <p>
              AI sedang mengetik
              <span className={styles.loadingDot}>.</span>
              <span className={styles.loadingDot}>.</span>
              <span className={styles.loadingDot}>.</span>
            </p>
          </div>
        )}
      </div>

      <div className={styles.chatInputArea}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          // --- INI BAGIAN YANG DIUBAH ---
          placeholder={isLoading ? '' : "Ketik pertanyaan Anda di sini..."}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          disabled={isLoading}
        />
        <button onClick={handleSend} disabled={isLoading}>
          <FaPaperPlane />
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;