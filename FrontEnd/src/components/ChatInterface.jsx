import React, { useState, useRef, useEffect } from 'react';
import { FaPaperPlane } from 'react-icons/fa';
import styles from './ChatInterface.module.css'; // Impor CSS Module

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { id: 1, text: "Selamat datang! Silakan ajukan pertanyaan Anda tentang Alkitab.", sender: 'ai', sources: [] },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatMessagesRef = useRef(null);

  useEffect(() => {
    chatMessagesRef.current?.scrollTo(0, chatMessagesRef.current.scrollHeight);
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (input.trim() === '' || isLoading) return;
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

  return (
    <div className={styles.chatContainer}>
      <div className={styles.chatMessages} ref={chatMessagesRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`${styles.messageBubble} ${styles[msg.sender]}`}>
            <p>{msg.text}</p>
            {msg.sender === 'ai' && msg.sources && msg.sources.length > 0 && (
              <div className={styles.messageSource}>
                <strong>Sumber:</strong> {msg.sources.map(s => s.source_range.split(':')[0]).join(', ')}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className={`${styles.messageBubble} ${styles.ai} ${styles.loading}`}>
            <p>AI sedang mengetik...</p>
          </div>
        )}
      </div>
      <div className={styles.chatInputArea}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ketik pertanyaan Anda di sini..."
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