import React, {useState, useRef, useEffect} from 'react';
import clsx from 'clsx';
import styles from './ChatbotWidget.module.css';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
}

const ChatbotWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {id: 1, text: 'Hello! 👋 I\'m your AI assistant for the Physical AI Textbook. How can I help you today?', sender: 'bot'}
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isOpen]);

  // --- BACKEND CONNECTION LOGIC ---
  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now(),
      text: inputValue,
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputValue;
    setInputValue('');
    setIsTyping(true);

    try {
      const response = await fetch('http://localhost:8000/chat/full', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: currentInput,  // ✅ 'message' ki jagah 'query' kar diya
          session_id: null,      // Backend session handle kar lega
          temperature: 0.1
        }),
      });

      if (!response.ok) {
        throw new Error('Server error');
      }

      const data = await response.json();

      const botResponse: Message = {
        id: Date.now() + 1,
        text: data.message || "I couldn't find an answer.", // ✅ Backend 'message' key bhej raha hai
        sender: 'bot'
      };
      setMessages(prev => [...prev, botResponse]);

    } catch (error) {
      console.error("Connection Error:", error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: "Server se rabta nahi ho pa raha. Check karein ke backend chal raha hai.",
        sender: 'bot'
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  };

  return (
    <div className={styles.chatbotContainer}>
      {isOpen ? (
        <div className={clsx(styles.chatbot, styles.chatbotOpen)}>
          <div className={styles.chatHeader}>
            <div className={styles.headerTitle}>
              <div className={styles.botAvatar}>🤖</div>
              <span>AI Assistant</span>
            </div>
            <button
              className={styles.closeButton}
              onClick={toggleChat}
              aria-label="Close chat"
            >
              ×
            </button>
          </div>

          <div className={styles.chatMessages}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={clsx(
                  styles.message,
                  styles[`${message.sender}Message`]
                )}
              >
                {message.sender === 'bot' && (
                  <div className={styles.botAvatarSmall}>🤖</div>
                )}
                <div className={styles.messageText}>{message.text}</div>
              </div>
            ))}

            {isTyping && (
              <div className={clsx(styles.message, styles.botMessage)}>
                <div className={styles.botAvatarSmall}>🤖</div>
                <div className={styles.typingIndicator}>
                  <div className={styles.typingDot}></div>
                  <div className={styles.typingDot}></div>
                  <div className={styles.typingDot}></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSend} className={styles.chatInputForm}>
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about Physical AI, ROS, or Robotics..."
              className={styles.chatInput}
              aria-label="Type your message"
            />
            <button
              type="submit"
              className={styles.sendButton}
              disabled={!inputValue.trim() || isTyping}
              aria-label="Send message"
            >
              ➤
            </button>
          </form>
        </div>
      ) : (
        <button
          className={clsx(styles.chatbotButton, styles.futuristicGlow)}
          onClick={toggleChat}
          aria-label="Open chatbot"
        >
          <div className={styles.chatIcon}>💬</div>
        </button>
      )}
    </div>
  );
}

export default ChatbotWidget;